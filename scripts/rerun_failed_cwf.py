#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "quadResult"
DEFAULT_CWF_EXE = REPO_ROOT / "bin" / "cwf"

ITER_RE = re.compile(r"\[QuadCoverLike\]\[Adam\] iter=(\d+)")
HINGE_PREV_CURR_RE = re.compile(r"hingeRaw\(prev/curr\)=([^\s|]+)/([^\s|]+)")
HINGE_SINGLE_RE = re.compile(r"hingeRaw=([^\s|]+)")
BATCH_FIELD_RE = re.compile(r"^\[Batch\]\s+([^=]+)=(.*)$")


@dataclass
class Candidate:
    model_name: str
    log_path: Path
    result_dir: Path
    normalized_mesh: Path | None
    sample_points: Path | None
    max_quadcover_iter: int
    last_hinge_raw: float | None
    stop_reason: str | None


@dataclass
class RerunResult:
    model_name: str
    status: str
    reason: str
    log_path: Path
    rerun_dir: Path
    rerun_log_path: Path
    max_quadcover_iter: int
    last_hinge_raw: float | None
    rvd_path: Path | None = None
    remesh_path: Path | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect models in quadResult that reached 800 QuadCover iterations "
            "without driving hinge loss to zero/feasible, then rerun 50-step CWF "
            "and export the final RVD/remesh."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="quadResult root to scan. Default: ./quadResult",
    )
    parser.add_argument(
        "--cwf-exe",
        type=Path,
        default=DEFAULT_CWF_EXE,
        help="Path to the cwf executable. Default: ./bin/cwf",
    )
    parser.add_argument(
        "--iteration-threshold",
        type=positive_int,
        default=800,
        help="Detect runs whose QuadCover Adam iteration reaches at least this value. Default: 800",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=50,
        help="CWF rerun iteration count. Default: 50",
    )
    parser.add_argument(
        "--hinge-epsilon",
        type=float,
        default=0.0,
        help="Treat hingeRaw <= this value as zero. Default: 0.0",
    )
    parser.add_argument(
        "--output-subdir",
        default="cwf50",
        help="Subdirectory under each model result dir for rerun outputs. Default: cwf50",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of reruns to execute in parallel. Default: min(4, cpu_count)",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only process model names containing this substring.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Only process the first N detected candidates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if the standardized CWF output files already exist.",
    )
    parser.add_argument(
        "--keep-raw-cwf-files",
        action="store_true",
        help="Keep raw Ours_*/Temp.* files produced by cwf. Default: clean them after packaging final outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print detected candidates without rerunning cwf.",
    )
    return parser.parse_args()


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def find_model_logs(result_root: Path) -> list[Path]:
    logs: list[Path] = []
    for child in sorted(result_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "quadCover":
            continue
        direct_log = child / f"{child.name}.log"
        if direct_log.exists():
            logs.append(direct_log)
            continue
        fallback_logs = sorted(child.glob("*.log"))
        if fallback_logs:
            logs.append(fallback_logs[0])
    return logs


def parse_candidate(log_path: Path) -> Candidate | None:
    fields: dict[str, str] = {}
    max_iter = 0
    last_hinge_raw: float | None = None
    stop_reason: str | None = None
    next_quadcover_command = False

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if next_quadcover_command and line:
                next_quadcover_command = False
                try:
                    parts = shlex.split(line)
                except ValueError:
                    parts = line.split()
                for idx, token in enumerate(parts):
                    if token == "--surface" and idx + 1 < len(parts):
                        fields["normalized_mesh"] = parts[idx + 1]
                    elif token == "--input" and idx + 1 < len(parts):
                        fields["sample_points"] = parts[idx + 1]
                    elif token == "--output" and idx + 1 < len(parts):
                        fields["result_dir"] = parts[idx + 1]

            batch_match = BATCH_FIELD_RE.match(line)
            if batch_match:
                fields[batch_match.group(1).strip()] = batch_match.group(2).strip()
                continue

            if line == "[Batch] QuadCover command:":
                next_quadcover_command = True

            iter_match = ITER_RE.search(line)
            if iter_match:
                max_iter = max(max_iter, int(iter_match.group(1)))

            if "STOP (hinge zero)" in line:
                stop_reason = "hinge_zero"
            elif "STOP (hinge feasible)" in line:
                stop_reason = "hinge_feasible"

            hinge_prev_curr = HINGE_PREV_CURR_RE.search(line)
            if hinge_prev_curr:
                parsed = safe_float(hinge_prev_curr.group(2))
                if parsed is not None:
                    last_hinge_raw = parsed
                continue

            hinge_single = HINGE_SINGLE_RE.search(line)
            if hinge_single:
                parsed = safe_float(hinge_single.group(1))
                if parsed is not None:
                    last_hinge_raw = parsed

    result_dir = Path(fields.get("result_dir", log_path.parent))
    normalized_mesh = fields.get("normalized_mesh")
    sample_points = fields.get("sample_points")

    return Candidate(
        model_name=log_path.parent.name,
        log_path=log_path,
        result_dir=result_dir,
        normalized_mesh=Path(normalized_mesh) if normalized_mesh else None,
        sample_points=Path(sample_points) if sample_points else None,
        max_quadcover_iter=max_iter,
        last_hinge_raw=last_hinge_raw,
        stop_reason=stop_reason,
    )


def is_failed_at_threshold(candidate: Candidate, threshold: int, hinge_epsilon: float) -> bool:
    if candidate.max_quadcover_iter < threshold:
        return False
    if candidate.stop_reason in {"hinge_zero", "hinge_feasible"}:
        return False
    if candidate.last_hinge_raw is None:
        return True
    return candidate.last_hinge_raw > hinge_epsilon


def latest_matching(paths: Iterable[Path]) -> Path | None:
    sorted_paths = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
    return sorted_paths[0] if sorted_paths else None


def collect_raw_outputs(rerun_dir: Path) -> tuple[Path | None, Path | None]:
    rvd_candidates = [
        path
        for path in rerun_dir.iterdir()
        if path.is_file() and path.name.endswith("_RVD.obj") and "_Iter" not in path.name
    ]
    remesh_candidates = [
        path
        for path in rerun_dir.iterdir()
        if path.is_file() and path.name.endswith("Remesh.obj") and "Iter" not in path.name
    ]
    return latest_matching(rvd_candidates), latest_matching(remesh_candidates)


def cleanup_raw_files(rerun_dir: Path, keep_names: set[str]) -> None:
    removable_prefixes = ("Ours_", "\\Ours_", "Temp.")
    for path in rerun_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in keep_names:
            continue
        if path.name.startswith(removable_prefixes) or path.name == "Temp.obj" or path.name == "Temp.off":
            path.unlink(missing_ok=True)


def rerun_one(
    candidate: Candidate,
    cwf_exe: Path,
    max_iterations: int,
    output_subdir: str,
    force: bool,
    keep_raw_cwf_files: bool,
) -> RerunResult:
    rerun_dir = candidate.result_dir / output_subdir
    rerun_log_path = rerun_dir / "cwf_rerun.log"
    standardized_prefix = f"CWF{max_iterations}_{candidate.model_name}"
    final_rvd_path = rerun_dir / f"{standardized_prefix}_RVD.obj"
    final_remesh_path = rerun_dir / f"{standardized_prefix}_Remesh.obj"

    if candidate.normalized_mesh is None or candidate.sample_points is None:
        return RerunResult(
            model_name=candidate.model_name,
            status="failed",
            reason="missing normalized_mesh or sample_points in batch log",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    if not candidate.normalized_mesh.exists():
        return RerunResult(
            model_name=candidate.model_name,
            status="failed",
            reason=f"normalized mesh does not exist: {candidate.normalized_mesh}",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    if not candidate.sample_points.exists():
        return RerunResult(
            model_name=candidate.model_name,
            status="failed",
            reason=f"sample points do not exist: {candidate.sample_points}",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    rerun_dir.mkdir(parents=True, exist_ok=True)
    if not force and final_rvd_path.exists() and final_remesh_path.exists():
        return RerunResult(
            model_name=candidate.model_name,
            status="skipped",
            reason="standardized CWF outputs already exist",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
            rvd_path=final_rvd_path,
            remesh_path=final_remesh_path,
        )

    if force:
        for path in rerun_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    cmd = [
        str(cwf_exe),
        str(candidate.normalized_mesh),
        str(candidate.sample_points),
        str(max_iterations),
    ]

    with rerun_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[Rerun] model={candidate.model_name}\n")
        log_file.write(f"[Rerun] source_log={candidate.log_path}\n")
        log_file.write(f"[Rerun] normalized_mesh={candidate.normalized_mesh}\n")
        log_file.write(f"[Rerun] sample_points={candidate.sample_points}\n")
        log_file.write(f"[Rerun] max_iterations={max_iterations}\n")
        log_file.write(f"[Rerun] command={' '.join(cmd)}\n\n")
        log_file.flush()

        proc = subprocess.run(
            cmd,
            cwd=rerun_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        return RerunResult(
            model_name=candidate.model_name,
            status="failed",
            reason=f"cwf exited with code {proc.returncode}",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    raw_rvd, raw_remesh = collect_raw_outputs(rerun_dir)
    if raw_rvd is None or raw_remesh is None:
        return RerunResult(
            model_name=candidate.model_name,
            status="failed",
            reason="cwf finished but final RVD/Remesh files were not found",
            log_path=candidate.log_path,
            rerun_dir=rerun_dir,
            rerun_log_path=rerun_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    shutil.copy2(raw_rvd, final_rvd_path)
    shutil.copy2(raw_remesh, final_remesh_path)

    if not keep_raw_cwf_files:
        keep_names = {rerun_log_path.name, final_rvd_path.name, final_remesh_path.name}
        cleanup_raw_files(rerun_dir, keep_names)

    return RerunResult(
        model_name=candidate.model_name,
        status="success",
        reason="rerun finished",
        log_path=candidate.log_path,
        rerun_dir=rerun_dir,
        rerun_log_path=rerun_log_path,
        max_quadcover_iter=candidate.max_quadcover_iter,
        last_hinge_raw=candidate.last_hinge_raw,
        rvd_path=final_rvd_path,
        remesh_path=final_remesh_path,
    )


def write_summary(result_root: Path, results: list[RerunResult], max_iterations: int) -> Path:
    summary_path = result_root / f"cwf{max_iterations}_rerun_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "status",
                "reason",
                "max_quadcover_iter",
                "last_hinge_raw",
                "log_path",
                "rerun_dir",
                "rerun_log_path",
                "rvd_path",
                "remesh_path",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.model_name,
                    item.status,
                    item.reason,
                    item.max_quadcover_iter,
                    "" if item.last_hinge_raw is None else f"{item.last_hinge_raw:.17g}",
                    item.log_path,
                    item.rerun_dir,
                    item.rerun_log_path,
                    item.rvd_path or "",
                    item.remesh_path or "",
                ]
            )
    return summary_path


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    cwf_exe = args.cwf_exe.resolve()

    if not result_root.is_dir():
        print(f"result root does not exist: {result_root}", file=sys.stderr)
        return 1
    if not cwf_exe.exists():
        print(f"cwf executable does not exist: {cwf_exe}", file=sys.stderr)
        return 1

    logs = find_model_logs(result_root)
    candidates = [parse_candidate(path) for path in logs]
    filtered = [
        candidate
        for candidate in candidates
        if candidate is not None
        and is_failed_at_threshold(candidate, args.iteration_threshold, args.hinge_epsilon)
        and (not args.name_contains or args.name_contains in candidate.model_name)
    ]

    filtered.sort(key=lambda item: item.model_name)
    if args.limit is not None:
        filtered = filtered[: args.limit]

    print(f"Detected {len(filtered)} candidate models.")
    for candidate in filtered:
        hinge_text = (
            "unknown"
            if candidate.last_hinge_raw is None
            else f"{candidate.last_hinge_raw:.6e}"
        )
        print(
            f"  {candidate.model_name}: iter={candidate.max_quadcover_iter}, "
            f"hingeRaw={hinge_text}, log={candidate.log_path}"
        )

    if args.dry_run:
        return 0

    if not filtered:
        return 0

    results: list[RerunResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(
                rerun_one,
                candidate,
                cwf_exe,
                args.max_iterations,
                args.output_subdir,
                args.force,
                args.keep_raw_cwf_files,
            ): candidate
            for candidate in filtered
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"[{result.status}] {result.model_name} | {result.reason}"
            )

    results.sort(key=lambda item: item.model_name)
    summary_path = write_summary(result_root, results, args.max_iterations)
    success_count = sum(1 for item in results if item.status == "success")
    skipped_count = sum(1 for item in results if item.status == "skipped")
    failed_count = sum(1 for item in results if item.status == "failed")
    print(
        f"Finished rerun: success={success_count} skipped={skipped_count} "
        f"failed={failed_count} summary={summary_path}"
    )
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
