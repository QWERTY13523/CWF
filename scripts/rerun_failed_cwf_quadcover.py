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
DEFAULT_QUADCOVER_EXE = REPO_ROOT / "bin" / "quadcover_main"
DEFAULT_SKIP_MODELS = {
    "35_embossed_logo_plate",
    "03_quarter_round_relief_block",
    "04_star_gear_plate",
    "07_star_gear_disk",
}

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
    logged_threads: int | None
    max_quadcover_iter: int
    last_hinge_raw: float | None
    stop_reason: str | None


@dataclass
class PipelineResult:
    model_name: str
    status: str
    reason: str
    source_log_path: Path
    pipeline_dir: Path
    init_remesh_path: Path
    quadcover_log_path: Path
    max_quadcover_iter: int
    last_hinge_raw: float | None
    cwf50_remesh_path: Path | None = None
    quadcover_remesh_path: Path | None = None
    quadcover_csv_path: Path | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect models in quadResult that reached 800 QuadCover iterations "
            "without driving hinge loss to zero/feasible, then reuse cwf50 "
            "final remesh as init input for QuadCover rerun and export final remesh/csv/log files."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="quadResult root to scan. Default: ./quadResult",
    )
    parser.add_argument(
        "--quadcover-exe",
        type=Path,
        default=DEFAULT_QUADCOVER_EXE,
        help="Path to the quadcover_main executable. Default: ./bin/quadcover_main",
    )
    parser.add_argument(
        "--iteration-threshold",
        type=positive_int,
        default=800,
        help="Detect runs whose QuadCover Adam iteration reaches at least this value. Default: 800",
    )
    parser.add_argument(
        "--hinge-epsilon",
        type=float,
        default=0.0,
        help="Treat hingeRaw <= this value as zero. Default: 0.0",
    )
    parser.add_argument(
        "--cwf-iterations",
        type=positive_int,
        default=50,
        help="Use cwf<iters>/CWF<iters>_<model>_Remesh.obj as QuadCover init input. Default: 50",
    )
    parser.add_argument(
        "--output-subdir",
        default="rerun_quadcover_from_cwf50",
        help="Subdirectory under each model result dir for rerun outputs. Default: rerun_quadcover_from_cwf50",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of models rerun in parallel. Default: min(4, cpu_count)",
    )
    parser.add_argument(
        "--threads-per-job",
        type=positive_int,
        default=None,
        help="Threads passed to quadcover_main. Default: reuse logged --threads, else auto(cpu_count/jobs).",
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
        help="Rerun even if standardized output files already exist.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Model name to skip. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print detected candidates without rerunning anything.",
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
    logged_threads: int | None = None

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
                    elif token == "--threads" and idx + 1 < len(parts):
                        try:
                            logged_threads = int(parts[idx + 1])
                        except ValueError:
                            logged_threads = None

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
        logged_threads=logged_threads,
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


def cleanup_raw_quadcover_files(stage_dir: Path, keep_names: set[str]) -> None:
    for path in stage_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in keep_names:
            continue
        if path.name.startswith("QuadCover_"):
            path.unlink(missing_ok=True)


def collect_quadcover_outputs(stage_dir: Path) -> tuple[Path | None, Path | None]:
    remesh_candidates = [
        path for path in stage_dir.iterdir()
        if path.is_file() and path.name.endswith("_Remesh.obj") and "_Iter" not in path.name
    ]
    csv_candidates = [
        path for path in stage_dir.iterdir()
        if path.is_file() and path.name.endswith("_Spheres.csv") and "_Iter" not in path.name
    ]
    return latest_matching(remesh_candidates), latest_matching(csv_candidates)


def prepare_stage_dir(stage_dir: Path, force: bool) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in stage_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)


def run_quadcover_stage(
    candidate: Candidate,
    quadcover_exe: Path,
    quadcover_threads: int,
    cwf50_remesh_path: Path,
    stage_dir: Path,
    force: bool,
) -> tuple[str, str, Path, Path | None, Path | None]:
    prepare_stage_dir(stage_dir, force)

    quadcover_log_path = stage_dir / "quadcover_rerun.log"
    final_remesh_path = stage_dir / f"QuadCoverRerun_{candidate.model_name}_Remesh.obj"
    final_csv_path = stage_dir / f"QuadCoverRerun_{candidate.model_name}_Spheres.csv"

    if not force and final_remesh_path.exists() and final_csv_path.exists():
        return "skipped", "standardized QuadCover outputs already exist", quadcover_log_path, final_remesh_path, final_csv_path

    cmd = [
        str(quadcover_exe),
        "--surface",
        str(candidate.normalized_mesh),
        "--input",
        str(cwf50_remesh_path),
        "--name",
        candidate.model_name,
        "--output",
        str(stage_dir),
        "--threads",
        str(quadcover_threads),
        "--final-only",
    ]

    with quadcover_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[Rerun] model={candidate.model_name}\n")
        log_file.write(f"[Rerun] source_log={candidate.log_path}\n")
        log_file.write(f"[Rerun] normalized_mesh={candidate.normalized_mesh}\n")
        log_file.write(f"[Rerun] cwf50_remesh={cwf50_remesh_path}\n")
        log_file.write(f"[Rerun] quadcover_threads={quadcover_threads}\n")
        log_file.write(f"[Rerun] command={' '.join(cmd)}\n\n")
        log_file.flush()

        proc = subprocess.run(
            cmd,
            cwd=stage_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        return "failed", f"quadcover_main exited with code {proc.returncode}", quadcover_log_path, None, None

    raw_remesh, raw_csv = collect_quadcover_outputs(stage_dir)
    if raw_remesh is None or raw_csv is None:
        return "failed", "quadcover_main finished but final remesh/csv were not found", quadcover_log_path, None, None

    shutil.copy2(raw_remesh, final_remesh_path)
    shutil.copy2(raw_csv, final_csv_path)
    keep_names = {
        quadcover_log_path.name,
        final_remesh_path.name,
        final_csv_path.name,
    }
    cleanup_raw_quadcover_files(stage_dir, keep_names)
    return "success", "quadcover rerun finished", quadcover_log_path, final_remesh_path, final_csv_path


def rerun_one(
    candidate: Candidate,
    quadcover_exe: Path,
    cwf_iterations: int,
    output_subdir: str,
    quadcover_threads_default: int,
    force: bool,
) -> PipelineResult:
    pipeline_dir = candidate.result_dir / output_subdir
    quadcover_dir = pipeline_dir / "quadcover"
    cwf50_dir = candidate.result_dir / f"cwf{cwf_iterations}"
    quadcover_log_path = quadcover_dir / "quadcover_rerun.log"
    cwf_prefix = f"CWF{cwf_iterations}_{candidate.model_name}"
    cwf50_remesh_path = cwf50_dir / f"{cwf_prefix}_Remesh.obj"
    final_quadcover_remesh_path = quadcover_dir / f"QuadCoverRerun_{candidate.model_name}_Remesh.obj"
    final_quadcover_csv_path = quadcover_dir / f"QuadCoverRerun_{candidate.model_name}_Spheres.csv"

    if candidate.normalized_mesh is None:
        return PipelineResult(
            model_name=candidate.model_name,
            status="failed",
            reason="missing normalized_mesh in batch log",
            source_log_path=candidate.log_path,
            pipeline_dir=pipeline_dir,
            init_remesh_path=cwf50_remesh_path,
            quadcover_log_path=quadcover_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    if not candidate.normalized_mesh.exists():
        return PipelineResult(
            model_name=candidate.model_name,
            status="failed",
            reason=f"normalized mesh does not exist: {candidate.normalized_mesh}",
            source_log_path=candidate.log_path,
            pipeline_dir=pipeline_dir,
            init_remesh_path=cwf50_remesh_path,
            quadcover_log_path=quadcover_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    if not cwf50_remesh_path.exists():
        return PipelineResult(
            model_name=candidate.model_name,
            status="failed",
            reason=f"cwf50 remesh does not exist: {cwf50_remesh_path}",
            source_log_path=candidate.log_path,
            pipeline_dir=pipeline_dir,
            init_remesh_path=cwf50_remesh_path,
            quadcover_log_path=quadcover_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
        )

    if (
        not force
        and final_quadcover_remesh_path.exists()
        and final_quadcover_csv_path.exists()
    ):
        return PipelineResult(
            model_name=candidate.model_name,
            status="skipped",
            reason="standardized final outputs already exist",
            source_log_path=candidate.log_path,
            pipeline_dir=pipeline_dir,
            init_remesh_path=cwf50_remesh_path,
            quadcover_log_path=quadcover_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
            cwf50_remesh_path=cwf50_remesh_path,
            quadcover_remesh_path=final_quadcover_remesh_path,
            quadcover_csv_path=final_quadcover_csv_path,
        )

    quadcover_threads = (
        candidate.logged_threads
        if candidate.logged_threads is not None and candidate.logged_threads > 0
        else quadcover_threads_default
    )
    quadcover_status, quadcover_reason, quadcover_log_path, quadcover_remesh_path, quadcover_csv_path = run_quadcover_stage(
        candidate,
        quadcover_exe,
        quadcover_threads,
        cwf50_remesh_path,
        quadcover_dir,
        force,
    )

    if quadcover_status == "failed":
        return PipelineResult(
            model_name=candidate.model_name,
            status="failed",
            reason=quadcover_reason,
            source_log_path=candidate.log_path,
            pipeline_dir=pipeline_dir,
            init_remesh_path=cwf50_remesh_path,
            quadcover_log_path=quadcover_log_path,
            max_quadcover_iter=candidate.max_quadcover_iter,
            last_hinge_raw=candidate.last_hinge_raw,
            cwf50_remesh_path=cwf50_remesh_path,
        )
    return PipelineResult(
        model_name=candidate.model_name,
        status="success" if quadcover_status == "success" else "skipped",
        reason="quadcover rerun finished" if quadcover_status == "success" else "standardized outputs already exist",
        source_log_path=candidate.log_path,
        pipeline_dir=pipeline_dir,
        init_remesh_path=cwf50_remesh_path,
        quadcover_log_path=quadcover_log_path,
        max_quadcover_iter=candidate.max_quadcover_iter,
        last_hinge_raw=candidate.last_hinge_raw,
        cwf50_remesh_path=cwf50_remesh_path,
        quadcover_remesh_path=quadcover_remesh_path,
        quadcover_csv_path=quadcover_csv_path,
    )


def write_summary(result_root: Path, results: list[PipelineResult], cwf_iterations: int) -> Path:
    summary_path = result_root / f"rerun_quadcover_from_cwf{cwf_iterations}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "status",
                "reason",
                "max_quadcover_iter",
                "last_hinge_raw",
                "source_log_path",
                "pipeline_dir",
                "init_remesh_path",
                "quadcover_log_path",
                "cwf50_remesh_path",
                "quadcover_remesh_path",
                "quadcover_csv_path",
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
                    item.source_log_path,
                    item.pipeline_dir,
                    item.init_remesh_path,
                    item.quadcover_log_path,
                    item.cwf50_remesh_path or "",
                    item.quadcover_remesh_path or "",
                    item.quadcover_csv_path or "",
                ]
            )
    return summary_path


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    quadcover_exe = args.quadcover_exe.resolve()
    excluded_models = DEFAULT_SKIP_MODELS | set(args.exclude_model)

    if not result_root.is_dir():
        print(f"result root does not exist: {result_root}", file=sys.stderr)
        return 1
    if not quadcover_exe.exists():
        print(f"quadcover_main does not exist: {quadcover_exe}", file=sys.stderr)
        return 1

    auto_threads = max(1, (os.cpu_count() or 1) // max(1, args.jobs))
    quadcover_threads_default = args.threads_per_job or auto_threads

    logs = find_model_logs(result_root)
    candidates = [parse_candidate(path) for path in logs]
    filtered = [
        candidate
        for candidate in candidates
        if candidate is not None
        and is_failed_at_threshold(candidate, args.iteration_threshold, args.hinge_epsilon)
        and candidate.model_name not in excluded_models
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
        thread_text = (
            str(candidate.logged_threads)
            if candidate.logged_threads is not None
            else str(quadcover_threads_default)
        )
        print(
                f"  {candidate.model_name}: iter={candidate.max_quadcover_iter}, "
                f"hingeRaw={hinge_text}, threads={thread_text}, log={candidate.log_path}"
        )

    if args.dry_run:
        return 0

    if not filtered:
        return 0

    results: list[PipelineResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(
                rerun_one,
                candidate,
                quadcover_exe,
                args.cwf_iterations,
                args.output_subdir,
                quadcover_threads_default,
                args.force,
            ): candidate
            for candidate in filtered
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(f"[{result.status}] {result.model_name} | {result.reason}")

    results.sort(key=lambda item: item.model_name)
    summary_path = write_summary(result_root, results, args.cwf_iterations)
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
