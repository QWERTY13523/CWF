#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_SCRIPT = Path(__file__).with_name("pymeshlab_preprocess.py")
DEFAULT_INPUT_DIR = REPO_ROOT / "cleanModels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "quadResult"
DEFAULT_CWF_EXE = REPO_ROOT / "bin" / "cwf"
DEFAULT_QUADCOVER_EXE = REPO_ROOT / "bin" / "quadcover_main"
DEFAULT_SAMPLER_CANDIDATES = [
    Path("/home/yiming/research/CWFSampling/build/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/build-vcpkg/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/cmake-build-debug/vcg_poisson_sampling"),
]

ITER_PATTERN = re.compile(r"\[QuadCoverLike\]\[Adam\] iter=(\d+)")
HINGE_PREV_CURR_RE = re.compile(r"hingeRaw\(prev/curr\)=([^\s|]+)/([^\s|]+)")
HINGE_SINGLE_RE = re.compile(r"hingeRaw=([^\s|]+)")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


@dataclass
class BatchOptions:
    input_dir: Path
    output_dir: Path
    sampler_exe: Path
    preprocess_script: Path
    cwf_exe: Path
    quadcover_exe: Path
    sample_num: int
    cwf_max_iterations: int
    merge_close_threshold: float
    jobs: int
    threads_per_job: int
    seed: int
    name_contains: str
    exclude_names: tuple[str, ...]
    force: bool


@dataclass
class RunResult:
    model_name: str
    model_path: Path
    result_dir: Path
    work_dir: Path
    success: bool = False
    sample_num: int = -1
    preprocess_exit_code: int = -1
    sampler_exit_code: int = -1
    cwf_exit_code: int = -1
    quadcover_exit_code: int = -1
    quadcover_iterations: int = 0
    final_hinge_raw: float | None = None
    total_time_seconds: float = 0.0
    error: str = ""
    cwf_remesh: Path | None = None
    hinge_remesh: Path | None = None
    hinge_csv: Path | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_args() -> BatchOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Run cleanModels batch: preprocess/normalize -> 100k sampling -> CWF -> "
            "QuadCover qem+hinge on CWF remesh."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing input OBJ/OFF models. Default: ./cleanModels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Result root. Default: ./quadResult",
    )
    parser.add_argument(
        "--sampler-exe",
        type=Path,
        default=None,
        help="Path to vcg_poisson_sampling executable.",
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=PREPROCESS_SCRIPT,
        help="Path to pymeshlab_preprocess.py.",
    )
    parser.add_argument(
        "--cwf-exe",
        type=Path,
        default=DEFAULT_CWF_EXE,
        help="Path to cwf executable.",
    )
    parser.add_argument(
        "--quadcover-exe",
        type=Path,
        default=DEFAULT_QUADCOVER_EXE,
        help="Path to quadcover_main executable.",
    )
    parser.add_argument(
        "--sample-num",
        type=positive_int,
        default=100000,
        help="Number of sampled input points per model. Default: 100000",
    )
    parser.add_argument(
        "--cwf-max-iterations",
        type=positive_int,
        default=50,
        help="CWF max iteration count. Default: 50",
    )
    parser.add_argument(
        "--merge-close-threshold",
        type=float,
        default=0.001,
        help="PyMeshLab merge-close threshold. Default: 0.001",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help="Number of models processed in parallel. Default: 1",
    )
    parser.add_argument(
        "--threads-per-job",
        type=positive_int,
        default=None,
        help="Threads used by sampler/CWF/QuadCover inside one job. Default: min(64, cpu_count/jobs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed forwarded to CWFSampling. Default: 0",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only process models whose stem contains this substring.",
    )
    parser.add_argument(
        "--exclude-names",
        default="",
        help=(
            "Comma-separated model stems to skip, matched case-insensitively. "
            "Example: lucy,lucy2,raptor,vase"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing quadResult/<model> and .work/<model> outputs.",
    )
    args = parser.parse_args()

    sampler_exe = args.sampler_exe or find_first_existing(DEFAULT_SAMPLER_CANDIDATES)
    if sampler_exe is None:
      parser.error("cannot find vcg_poisson_sampling; pass --sampler-exe explicitly")

    threads_per_job = args.threads_per_job
    if threads_per_job is None:
        cpu = os.cpu_count() or 1
        threads_per_job = max(1, min(64, cpu // max(1, args.jobs)))

    exclude_names = tuple(
        sorted({
            name.strip().lower()
            for name in args.exclude_names.split(",")
            if name.strip()
        })
    )

    return BatchOptions(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        sampler_exe=sampler_exe.resolve(),
        preprocess_script=args.preprocess_script.resolve(),
        cwf_exe=args.cwf_exe.resolve(),
        quadcover_exe=args.quadcover_exe.resolve(),
        sample_num=args.sample_num,
        cwf_max_iterations=args.cwf_max_iterations,
        merge_close_threshold=args.merge_close_threshold,
        jobs=args.jobs,
        threads_per_job=threads_per_job,
        seed=args.seed,
        name_contains=args.name_contains,
        exclude_names=exclude_names,
        force=args.force,
    )


def lower_ext(path: Path) -> str:
    return path.suffix.lower()


def triangulate_face(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    triangles = []
    for i in range(1, len(indices) - 1):
        triangles.append((indices[0], indices[i], indices[i + 1]))
    return triangles


def load_obj_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                parts = raw.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif raw.startswith("f "):
                tokens = raw.strip().split()[1:]
                face_indices: list[int] = []
                for token in tokens:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    idx = int(head)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    face_indices.append(idx)
                faces.extend(triangulate_face(face_indices))
    return vertices, faces


def load_off_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        tokens = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not tokens:
        raise ValueError(f"{path} is empty")
    if tokens[0] not in {"OFF", "COFF"}:
        raise ValueError(f"{path} is not a valid OFF/COFF mesh")
    counts = tokens[1].split()
    if len(counts) < 2:
        raise ValueError(f"{path} has invalid OFF header")
    vertex_count = int(counts[0])
    face_count = int(counts[1])
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    cursor = 2
    for _ in range(vertex_count):
        parts = tokens[cursor].split()
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
        cursor += 1
    for _ in range(face_count):
        parts = tokens[cursor].split()
        count = int(parts[0])
        indices = [int(value) for value in parts[1:1 + count]]
        faces.extend(triangulate_face(indices))
        cursor += 1
    return vertices, faces


def load_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    ext = lower_ext(path)
    if ext == ".obj":
        return load_obj_mesh(path)
    if ext == ".off":
        return load_off_mesh(path)
    raise ValueError(f"unsupported mesh format: {path}")


def normalize_vertices(vertices: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    if not vertices:
        raise ValueError("mesh has no vertices")
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    extent = max(maxs[i] - mins[i] for i in range(3))
    if extent <= 0.0:
        raise ValueError("mesh bounding box is degenerate")
    normalized = []
    for v in vertices:
        normalized.append(tuple((v[i] - mins[i]) / extent for i in range(3)))
    return normalized


def write_obj_mesh(path: Path,
                   vertices: list[tuple[float, float, float]],
                   faces: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for v in vertices:
            handle.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for f in faces:
            handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def fallback_preprocess_mesh(input_path: Path, output_path: Path) -> None:
    vertices, faces = load_mesh(input_path)
    if not vertices or not faces:
        raise ValueError("fallback preprocess got an empty mesh")
    write_obj_mesh(output_path, vertices, faces)


def list_models(input_dir: Path,
                name_contains: str,
                exclude_names: tuple[str, ...]) -> list[Path]:
    models = [
        path for path in input_dir.rglob("*")
        if path.is_file() and lower_ext(path) in {".obj", ".off"}
    ]
    models = sorted(models)
    if name_contains:
        needle = name_contains.lower()
        models = [path for path in models if needle in path.stem.lower()]
    if exclude_names:
        excluded = set(exclude_names)
        models = [path for path in models if path.stem.lower() not in excluded]
    return models


def preprocess_cmd(options: BatchOptions, input_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(options.preprocess_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--merge-close-threshold",
        str(options.merge_close_threshold),
    ]


def latest_matching(paths: Iterable[Path]) -> Path | None:
    sorted_paths = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
    return sorted_paths[0] if sorted_paths else None


def collect_cwf_outputs(raw_dir: Path) -> tuple[Path | None, Path | None]:
    remesh_candidates = [
        path for path in raw_dir.iterdir()
        if path.is_file() and path.name.endswith("Remesh.obj") and "Iter" not in path.name
    ]
    points_candidates = [
        path for path in raw_dir.iterdir()
        if path.is_file() and path.name.endswith("_Points.xyz") and "_Iter" not in path.name
    ]
    return latest_matching(remesh_candidates), latest_matching(points_candidates)


def collect_quadcover_outputs(raw_dir: Path) -> tuple[Path | None, Path | None]:
    remesh_candidates = [
        path for path in raw_dir.iterdir()
        if path.is_file() and path.name.endswith("_Remesh.obj") and "_Iter" not in path.name
    ]
    csv_candidates = [
        path for path in raw_dir.iterdir()
        if path.is_file() and path.name.endswith("_Spheres.csv") and "_Iter" not in path.name
    ]
    return latest_matching(remesh_candidates), latest_matching(csv_candidates)


def parse_quadcover_log(log_path: Path) -> tuple[int, float | None]:
    max_iter = 0
    last_hinge_raw: float | None = None
    if not log_path.exists():
        return max_iter, last_hinge_raw
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            iter_match = ITER_PATTERN.search(line)
            if iter_match:
                max_iter = max(max_iter, int(iter_match.group(1)))
            prev_curr_match = HINGE_PREV_CURR_RE.search(line)
            if prev_curr_match:
                try:
                    last_hinge_raw = float(prev_curr_match.group(2))
                except ValueError:
                    pass
                continue
            single_match = HINGE_SINGLE_RE.search(line)
            if single_match:
                try:
                    last_hinge_raw = float(single_match.group(1))
                except ValueError:
                    pass
    return max_iter, last_hinge_raw


def make_job_env(threads_per_job: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(threads_per_job)
    env["OMP_NUM_THREADS"] = thread_value
    env["OPENBLAS_NUM_THREADS"] = thread_value
    env["MKL_NUM_THREADS"] = thread_value
    env["NUMEXPR_NUM_THREADS"] = thread_value
    return env


def run_logged(cmd: list[str],
               log_file,
               cwd: Path | None,
               env: dict[str, str] | None = None) -> int:
    log_file.write(" ".join(cmd) + "\n")
    log_file.flush()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_file.write(f"[exit_code] {proc.returncode}\n\n")
    log_file.flush()
    return proc.returncode


def run_one_model(options: BatchOptions, model_path: Path) -> RunResult:
    start_time = time.perf_counter()
    model_name = model_path.stem
    result_dir = options.output_dir / model_name
    work_dir = options.output_dir / ".work" / model_name
    preprocess_dir = work_dir / "preprocess"
    sampler_dir = work_dir / "sampling"
    cwf_dir = work_dir / "cwf"
    quadcover_dir = work_dir / "quadcover"

    result = RunResult(
        model_name=model_name,
        model_path=model_path,
        result_dir=result_dir,
        work_dir=work_dir,
        sample_num=options.sample_num,
    )

    if result_dir.exists():
        if not options.force:
            result.error = f"result dir exists (use --force to overwrite): {result_dir}"
            return result
        shutil.rmtree(result_dir)
    if work_dir.exists():
        if options.force:
            shutil.rmtree(work_dir)
        else:
            result.error = f"work dir exists (use --force to overwrite): {work_dir}"
            return result

    result_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    sampler_dir.mkdir(parents=True, exist_ok=True)
    cwf_dir.mkdir(parents=True, exist_ok=True)
    quadcover_dir.mkdir(parents=True, exist_ok=True)

    preprocess_log = work_dir / "preprocess.log"
    sampler_log = work_dir / "sampling.log"
    cwf_log = work_dir / "cwf.log"
    quadcover_log = work_dir / "quadcover.log"

    preprocessed_mesh = preprocess_dir / f"{model_name}_merged.obj"
    normalized_mesh = preprocess_dir / f"{model_name}_normalized.obj"
    sample_points = sampler_dir / f"n{options.sample_num}_{model_name}_inputPoints.xyz"

    try:
        with preprocess_log.open("w", encoding="utf-8") as log_file:
            cmd = preprocess_cmd(options, model_path, preprocessed_mesh)
            result.preprocess_exit_code = run_logged(cmd, log_file, cwd=REPO_ROOT)
            if result.preprocess_exit_code != 0:
                log_file.write(
                    "[fallback] PyMeshLab preprocess unavailable/failed; "
                    "fall back to direct OBJ/OFF triangulated copy.\n"
                )
                log_file.flush()
                fallback_preprocess_mesh(model_path, preprocessed_mesh)
                result.preprocess_exit_code = 0

        vertices, faces = load_mesh(preprocessed_mesh)
        normalized_vertices = normalize_vertices(vertices)
        write_obj_mesh(normalized_mesh, normalized_vertices, faces)
    except Exception as exc:  # noqa: BLE001
        result.total_time_seconds = time.perf_counter() - start_time
        result.error = f"normalize failed: {exc}"
        return result

    with sampler_log.open("w", encoding="utf-8") as log_file:
        sampler_cmd = [
            str(options.sampler_exe),
            str(normalized_mesh),
            str(sample_points),
            str(options.sample_num),
            str(options.seed),
        ]
        result.sampler_exit_code = run_logged(sampler_cmd, log_file, cwd=REPO_ROOT)
    if result.sampler_exit_code != 0:
        result.total_time_seconds = time.perf_counter() - start_time
        result.error = f"sampling failed with exit code {result.sampler_exit_code}"
        return result

    job_env = make_job_env(options.threads_per_job)
    with cwf_log.open("w", encoding="utf-8") as log_file:
        cwf_cmd = [
            str(options.cwf_exe),
            str(normalized_mesh),
            str(sample_points),
            str(options.cwf_max_iterations),
        ]
        result.cwf_exit_code = run_logged(cwf_cmd, log_file, cwd=cwf_dir, env=job_env)
    if result.cwf_exit_code != 0:
        result.total_time_seconds = time.perf_counter() - start_time
        result.error = f"cwf failed with exit code {result.cwf_exit_code}"
        return result

    cwf_remesh_raw, cwf_points_raw = collect_cwf_outputs(cwf_dir)
    if cwf_remesh_raw is None or cwf_points_raw is None:
        result.total_time_seconds = time.perf_counter() - start_time
        result.error = "cwf finished but final remesh/points were not found"
        return result

    cwf_remesh_final = result_dir / "cwf_remesh.obj"
    shutil.copy2(cwf_remesh_raw, cwf_remesh_final)
    result.cwf_remesh = cwf_remesh_final

    with quadcover_log.open("w", encoding="utf-8") as log_file:
        quadcover_cmd = [
            str(options.quadcover_exe),
            "--surface",
            str(cwf_remesh_raw),
            "--input",
            str(cwf_points_raw),
            "--output",
            str(quadcover_dir),
            "--name",
            model_name,
            "--threads",
            str(options.threads_per_job),
            "--final-only",
        ]
        result.quadcover_exit_code = run_logged(
            quadcover_cmd, log_file, cwd=REPO_ROOT, env=job_env
        )
    if result.quadcover_exit_code != 0:
        result.error = f"quadcover failed with exit code {result.quadcover_exit_code}"
        result.quadcover_iterations, result.final_hinge_raw = parse_quadcover_log(quadcover_log)
        result.total_time_seconds = time.perf_counter() - start_time
        return result

    hinge_remesh_raw, hinge_csv_raw = collect_quadcover_outputs(quadcover_dir)
    if hinge_remesh_raw is None or hinge_csv_raw is None:
        result.error = "quadcover finished but final remesh/csv were not found"
        result.quadcover_iterations, result.final_hinge_raw = parse_quadcover_log(quadcover_log)
        result.total_time_seconds = time.perf_counter() - start_time
        return result

    hinge_remesh_final = result_dir / "hinge_final_remesh.obj"
    hinge_csv_final = result_dir / "hinge_final.csv"
    shutil.copy2(hinge_remesh_raw, hinge_remesh_final)
    shutil.copy2(hinge_csv_raw, hinge_csv_final)
    result.hinge_remesh = hinge_remesh_final
    result.hinge_csv = hinge_csv_final

    result.quadcover_iterations, result.final_hinge_raw = parse_quadcover_log(quadcover_log)
    result.total_time_seconds = time.perf_counter() - start_time
    result.success = True
    return result


def write_summary(output_dir: Path, results: list[RunResult]) -> Path:
    summary_path = output_dir / "cleanmodels_cwf_hinge_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_name",
            "status",
            "sample_num",
            "preprocess_exit_code",
            "sampler_exit_code",
            "cwf_exit_code",
            "quadcover_exit_code",
            "quadcover_iterations",
            "final_hinge_raw",
            "total_time_seconds",
            "model_path",
            "result_dir",
            "cwf_remesh",
            "hinge_final_remesh",
            "hinge_final_csv",
            "work_dir",
            "error",
        ])
        for result in results:
            writer.writerow([
                result.model_name,
                "success" if result.success else "failed",
                result.sample_num,
                result.preprocess_exit_code,
                result.sampler_exit_code,
                result.cwf_exit_code,
                result.quadcover_exit_code,
                result.quadcover_iterations,
                "" if result.final_hinge_raw is None else f"{result.final_hinge_raw:.17g}",
                f"{result.total_time_seconds:.3f}",
                result.model_path,
                result.result_dir,
                result.cwf_remesh or "",
                result.hinge_remesh or "",
                result.hinge_csv or "",
                result.work_dir,
                result.error,
            ])
    return summary_path


def main() -> int:
    options = parse_args()
    if not options.input_dir.is_dir():
        print(f"input directory does not exist: {options.input_dir}", file=sys.stderr)
        return 1
    if not options.sampler_exe.exists():
        print(f"sampler executable does not exist: {options.sampler_exe}", file=sys.stderr)
        return 1
    if not options.preprocess_script.exists():
        print(f"preprocess script does not exist: {options.preprocess_script}", file=sys.stderr)
        return 1
    if not options.cwf_exe.exists():
        print(f"cwf executable does not exist: {options.cwf_exe}", file=sys.stderr)
        return 1
    if not options.quadcover_exe.exists():
        print(f"quadcover_main does not exist: {options.quadcover_exe}", file=sys.stderr)
        return 1

    models = list_models(options.input_dir, options.name_contains, options.exclude_names)
    if not models:
        print(f"no OBJ/OFF models found in {options.input_dir}", file=sys.stderr)
        return 1

    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / ".work").mkdir(parents=True, exist_ok=True)

    print(
        f"Batch start: models={len(models)} jobs={options.jobs} "
        f"threads-per-job={options.threads_per_job} sample-num={options.sample_num} "
        f"cwf-iters={options.cwf_max_iterations} "
        f"exclude={','.join(options.exclude_names) if options.exclude_names else 'none'}"
    )

    batch_start = time.perf_counter()
    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as executor:
        future_map = {executor.submit(run_one_model, options, model): model for model in models}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            status = "success" if result.success else "failed"
            hinge_text = (
                "na" if result.final_hinge_raw is None else f"{result.final_hinge_raw:.6e}"
            )
            print(
                f"[{status}] {result.model_name} | "
                f"iters={result.quadcover_iterations} | "
                f"hinge={hinge_text} | "
                f"time={result.total_time_seconds:.3f}s"
            )

    results.sort(key=lambda item: item.model_name.lower())
    summary_path = write_summary(options.output_dir, results)
    failures = sum(1 for item in results if not item.success)
    total_time = time.perf_counter() - batch_start
    print(
        f"Batch finished: success={len(results) - failures} failed={failures} "
        f"total_time={total_time:.3f}s summary={summary_path}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
