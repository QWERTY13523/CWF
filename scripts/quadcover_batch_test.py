#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import multiprocessing as mp
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_SCRIPT = Path(__file__).with_name("pymeshlab_preprocess.py")
DEFAULT_INPUT_CANDIDATES = [
    REPO_ROOT / "ABC",
    Path("/home/yiming/research/CWFSampling/ABC"),
]
DEFAULT_SAMPLER_CANDIDATES = [
    Path("/home/yiming/research/CWFSampling/build/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/build-vcpkg/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/cmake-build-debug/vcg_poisson_sampling"),
]
ITER_PATTERN = re.compile(r"\[QuadCoverLike\]\[Adam\] iter=(\d+)")


@dataclass
class BatchOptions:
    input_dir: Path
    output_dir: Path
    sampler_exe: Path
    cwf_exe: Path
    quadcover_exe: Path
    preprocess_script: Path
    sample_num: int | None
    merge_close_threshold: float
    jobs: int
    threads_per_job: int
    seed: int
    cwf_iterations: int
    name_contains: str
    limit: int | None


@dataclass
class RunResult:
    model_key: str
    model_name: str
    model_path: Path
    result_dir: Path
    model_log_path: Path
    preprocessed_mesh: Path
    normalized_mesh: Path
    sample_points: Path
    cwf_work_dir: Path
    quadcover_work_dir: Path
    cwf_remesh_path: Path | None = None
    quadcover_remesh_path: Path | None = None
    quadcover_spheres_csv_path: Path | None = None
    sample_num: int = -1
    quadcover_iterations: int = 0
    start_time_utc: str = ""
    end_time_utc: str = ""
    total_time_seconds: float = 0.0
    preprocess_seconds: float = 0.0
    normalize_seconds: float = 0.0
    sampling_seconds: float = 0.0
    cwf_seconds: float = 0.0
    quadcover_seconds: float = 0.0
    export_seconds: float = 0.0
    preprocess_exit_code: int = -1
    sampler_exit_code: int = -1
    cwf_exit_code: int = -1
    quadcover_exit_code: int = -1
    success: bool = False
    error: str = ""


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args() -> BatchOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Batch test all OBJ/OFF models in ABC: preprocess -> normalize -> "
            "CWFSampling -> CWF(50) -> QuadCover(final)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing OBJ/OFF models. Default: ABC or ~/research/CWFSampling/ABC",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "result" / "ABC",
        help="Output root. Default: ./result/ABC",
    )
    parser.add_argument(
        "--sampler-exe",
        type=Path,
        default=None,
        help="Path to vcg_poisson_sampling executable.",
    )
    parser.add_argument(
        "--cwf-exe",
        type=Path,
        default=REPO_ROOT / "bin" / "cwf",
        help="Path to the cwf executable. Default: ./bin/cwf",
    )
    parser.add_argument(
        "--quadcover-exe",
        type=Path,
        default=REPO_ROOT / "bin" / "quadcover_main",
        help="Path to the quadcover_main executable. Default: ./bin/quadcover_main",
    )
    parser.add_argument(
        "--sample-num",
        type=positive_int,
        default=None,
        help="Fixed number of sampled points per model. Default: max(8000, vertex_count / 3)",
    )
    parser.add_argument(
        "--merge-close-threshold",
        type=float,
        default=0.001,
        help=(
            "PyMeshLab merge-close threshold in PercentageValue units. "
            "Default: 0.001"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of models processed in parallel.",
    )
    parser.add_argument(
        "--threads-per-job",
        type=positive_int,
        default=None,
        help="Threads used inside one quadcover_main process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to CWFSampling. Default: 0",
    )
    parser.add_argument(
        "--cwf-iterations",
        type=non_negative_int,
        default=50,
        help="CWF iterations before QuadCover. Default: 50",
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
        help="Only process the first N models after filtering.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or find_first_existing(DEFAULT_INPUT_CANDIDATES)
    if input_dir is None:
        parser.error("cannot find ABC folder; pass --input-dir explicitly")

    sampler_exe = args.sampler_exe or find_first_existing(DEFAULT_SAMPLER_CANDIDATES)
    if sampler_exe is None:
        parser.error("cannot find vcg_poisson_sampling; pass --sampler-exe explicitly")

    threads_per_job = args.threads_per_job
    if threads_per_job is None:
        threads_per_job = max(1, (os.cpu_count() or 1) // args.jobs)

    return BatchOptions(
        input_dir=input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        sampler_exe=sampler_exe.resolve(),
        cwf_exe=args.cwf_exe.resolve(),
        quadcover_exe=args.quadcover_exe.resolve(),
        preprocess_script=PREPROCESS_SCRIPT.resolve(),
        sample_num=args.sample_num,
        merge_close_threshold=args.merge_close_threshold,
        jobs=args.jobs,
        threads_per_job=threads_per_job,
        seed=args.seed,
        cwf_iterations=args.cwf_iterations,
        name_contains=args.name_contains,
        limit=args.limit,
    )


def lower_ext(path: Path) -> str:
    return path.suffix.lower()


def triangulate_face(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    triangles = []
    for idx in range(1, len(indices) - 1):
        triangles.append((indices[0], indices[idx], indices[idx + 1]))
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

    mins = [min(v[idx] for v in vertices) for idx in range(3)]
    maxs = [max(v[idx] for v in vertices) for idx in range(3)]
    extent = max(maxs[idx] - mins[idx] for idx in range(3))
    if extent <= 0.0:
        raise ValueError("mesh bounding box is degenerate")

    normalized = []
    for vertex in vertices:
        normalized.append(tuple((vertex[idx] - mins[idx]) / extent for idx in range(3)))
    return normalized


def write_obj_mesh(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def list_models(input_dir: Path, name_contains: str, limit: int | None) -> list[Path]:
    models = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and lower_ext(path) in {".obj", ".off"}
    ]
    models.sort()
    if name_contains:
        models = [path for path in models if name_contains in path.stem]
    if limit is not None:
        models = models[:limit]
    return models


def make_model_key(input_dir: Path, model_path: Path) -> str:
    relative = model_path.relative_to(input_dir).with_suffix("")
    return "__".join(relative.parts)


def resolve_sample_num(options: BatchOptions, vertex_count: int) -> int:
    if options.sample_num is not None:
        return options.sample_num
    return max(8000, vertex_count // 3)


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


def parse_quadcover_iterations(log_path: Path) -> int:
    max_iter = 0
    if not log_path.exists():
        return max_iter
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = ITER_PATTERN.search(line)
            if match:
                max_iter = max(max_iter, int(match.group(1)))
    return max_iter


def format_cmd(command: list[str]) -> str:
    return shlex.join(command)


def sanitized_artifact_name(path: Path) -> str:
    return path.name.lstrip("\\/")


def find_exported_artifact(output_dir: Path, prefix: str, suffix: str) -> Path | None:
    candidates: list[Path] = []
    if not output_dir.exists():
        return None

    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        name = sanitized_artifact_name(path)
        if not name.startswith(prefix):
            continue
        if not name.endswith(suffix):
            continue
        candidates.append(path)

    if not candidates:
        return None

    final_candidates = [
        path for path in candidates if "Iter" not in sanitized_artifact_name(path)
    ]
    selected_pool = final_candidates or candidates
    selected_pool.sort(key=lambda path: (path.stat().st_mtime, sanitized_artifact_name(path)))
    return selected_pool[-1]


def find_exported_remesh(output_dir: Path, prefix: str) -> Path | None:
    return find_exported_artifact(output_dir, prefix, "_Remesh.obj")


def find_exported_spheres_csv(output_dir: Path, prefix: str) -> Path | None:
    return find_exported_artifact(output_dir, prefix, "_Spheres.csv")


def run_logged_command(
    command: list[str],
    log_file,
    cwd: Path | None = None,
) -> int:
    if cwd is not None:
        cwd.mkdir(parents=True, exist_ok=True)
        log_file.write(f"[Batch] cwd={cwd}\n")
    log_file.write(format_cmd(command) + "\n\n")
    log_file.flush()
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode


def copy_artifact(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def acquire_batch_lock(output_dir: Path) -> tuple[int, Path]:
    lock_path = output_dir / ".batch.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"output directory is already locked by another batch run: {lock_path}"
        ) from exc

    payload = (
        f"pid={os.getpid()}\n"
        f"start_time_utc={utc_now_iso()}\n"
        f"output_dir={output_dir}\n"
    )
    os.write(fd, payload.encode("utf-8"))
    os.fsync(fd)
    return fd, lock_path


def release_batch_lock(fd: int, lock_path: Path) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def run_one_model(options: BatchOptions, model_path: Path) -> RunResult:
    start_time = time.perf_counter()
    model_key = make_model_key(options.input_dir, model_path)
    model_name = model_path.stem
    start_time_utc = utc_now_iso()

    result_dir = options.output_dir / model_name
    work_dir = options.output_dir / ".work" / model_key
    log_path = result_dir / f"{model_name}.log"
    preprocessed_mesh = work_dir / f"{model_name}_merged.obj"
    normalized_mesh = work_dir / f"{model_name}_normalized.obj"
    sample_points = work_dir / f"{model_name}_inputPoints.xyz"
    cwf_work_dir = work_dir / f"cwf{options.cwf_iterations}"
    quadcover_work_dir = work_dir / "quadcover_final"

    result = RunResult(
        model_key=model_key,
        model_name=model_name,
        model_path=model_path,
        result_dir=result_dir,
        model_log_path=log_path,
        preprocessed_mesh=preprocessed_mesh,
        normalized_mesh=normalized_mesh,
        sample_points=sample_points,
        cwf_work_dir=cwf_work_dir,
        quadcover_work_dir=quadcover_work_dir,
        start_time_utc=start_time_utc,
    )

    if result_dir.exists():
        shutil.rmtree(result_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[Batch] model={model_path}\n")
        log_file.write(f"[Batch] model_key={model_key}\n")
        log_file.write(f"[Batch] start_time_utc={result.start_time_utc}\n")
        log_file.write(f"[Batch] result_dir={result_dir}\n")
        log_file.write(f"[Batch] work_dir={work_dir}\n")
        log_file.write(f"[Batch] merge_close_threshold={options.merge_close_threshold}\n")
        log_file.write(f"[Batch] seed={options.seed}\n")
        log_file.write(f"[Batch] cwf_iterations={options.cwf_iterations}\n")
        log_file.write(f"[Batch] threads_per_job={options.threads_per_job}\n\n")
        log_file.flush()

        preprocess_command = preprocess_cmd(options, model_path, preprocessed_mesh)
        log_file.write("[Batch] Preprocess command:\n")
        preprocess_start = time.perf_counter()
        result.preprocess_exit_code = run_logged_command(preprocess_command, log_file)
        result.preprocess_seconds = time.perf_counter() - preprocess_start
        log_file.write(f"[Batch] preprocess_seconds={result.preprocess_seconds:.6f}\n\n")
        log_file.flush()
        if result.preprocess_exit_code != 0:
            result.error = f"preprocess failed with exit code {result.preprocess_exit_code}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        try:
            normalize_start = time.perf_counter()
            vertices, faces = load_mesh(preprocessed_mesh)
            normalized_vertices = normalize_vertices(vertices)
            sample_num = resolve_sample_num(options, len(vertices))
            sample_points = work_dir / f"n{sample_num}_{model_name}_inputPoints.xyz"
            result.sample_num = sample_num
            result.sample_points = sample_points
            write_obj_mesh(normalized_mesh, normalized_vertices, faces)
            result.normalize_seconds = time.perf_counter() - normalize_start
        except Exception as exc:  # noqa: BLE001
            result.error = f"normalize failed: {exc}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        log_file.write(f"[Batch] sample_num={result.sample_num}\n")
        log_file.write(f"[Batch] normalized_mesh={normalized_mesh}\n")
        log_file.write(f"[Batch] sample_points={sample_points}\n\n")
        log_file.write(f"[Batch] normalize_seconds={result.normalize_seconds:.6f}\n\n")
        log_file.flush()

        sampler_cmd = [
            str(options.sampler_exe),
            str(normalized_mesh),
            str(sample_points),
            str(result.sample_num),
            str(options.seed),
        ]
        log_file.write("[Batch] CWFSampling command:\n")
        sampling_start = time.perf_counter()
        result.sampler_exit_code = run_logged_command(sampler_cmd, log_file)
        result.sampling_seconds = time.perf_counter() - sampling_start
        log_file.write(f"[Batch] sampling_seconds={result.sampling_seconds:.6f}\n\n")
        log_file.flush()
        if result.sampler_exit_code != 0:
            result.error = f"CWFSampling failed with exit code {result.sampler_exit_code}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        cwf_cmd = [
            str(options.cwf_exe),
            str(normalized_mesh),
            str(sample_points),
            str(options.cwf_iterations),
        ]
        log_file.write("[Batch] CWF command:\n")
        cwf_start = time.perf_counter()
        result.cwf_exit_code = run_logged_command(cwf_cmd, log_file, cwd=cwf_work_dir)
        result.cwf_seconds = time.perf_counter() - cwf_start
        log_file.write(f"[Batch] cwf_seconds={result.cwf_seconds:.6f}\n\n")
        log_file.flush()
        if result.cwf_exit_code != 0:
            result.error = f"cwf failed with exit code {result.cwf_exit_code}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        export_start = time.perf_counter()
        cwf_source_remesh = find_exported_remesh(cwf_work_dir, "Ours_")
        if cwf_source_remesh is None:
            result.error = f"cannot find CWF final remesh in {cwf_work_dir}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        cwf_target_remesh = result_dir / f"CWF{options.cwf_iterations}_{model_name}_Remesh.obj"
        result.cwf_remesh_path = copy_artifact(cwf_source_remesh, cwf_target_remesh)
        log_file.write(f"[Batch] selected_cwf_remesh={cwf_source_remesh}\n")
        log_file.write(f"[Batch] exported_cwf_remesh={result.cwf_remesh_path}\n\n")
        log_file.flush()
        result.export_seconds += time.perf_counter() - export_start

        quadcover_cmd = [
            str(options.quadcover_exe),
            "--surface",
            str(normalized_mesh),
            "--input",
            str(cwf_source_remesh),
            "--name",
            model_name,
            "--output",
            str(quadcover_work_dir),
            "--threads",
            str(options.threads_per_job),
            "--cwf-iters",
            "0",
            "--final-only",
        ]
        log_file.write("[Batch] QuadCover command:\n")
        quadcover_start = time.perf_counter()
        result.quadcover_exit_code = run_logged_command(quadcover_cmd, log_file)
        result.quadcover_seconds = time.perf_counter() - quadcover_start
        log_file.write(f"[Batch] quadcover_seconds={result.quadcover_seconds:.6f}\n\n")
        log_file.flush()
        if result.quadcover_exit_code != 0:
            result.error = f"quadcover_main failed with exit code {result.quadcover_exit_code}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        export_start = time.perf_counter()
        quadcover_source_remesh = find_exported_remesh(quadcover_work_dir, "QuadCover_")
        if quadcover_source_remesh is None:
            result.error = f"cannot find QuadCover final remesh in {quadcover_work_dir}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        quadcover_source_spheres_csv = find_exported_spheres_csv(quadcover_work_dir, "QuadCover_")
        if quadcover_source_spheres_csv is None:
            result.error = f"cannot find QuadCover final spheres csv in {quadcover_work_dir}"
            result.end_time_utc = utc_now_iso()
            result.total_time_seconds = time.perf_counter() - start_time
            return result

        result.quadcover_remesh_path = copy_artifact(
            quadcover_source_remesh,
            result_dir / sanitized_artifact_name(quadcover_source_remesh),
        )
        result.quadcover_spheres_csv_path = copy_artifact(
            quadcover_source_spheres_csv,
            result_dir / sanitized_artifact_name(quadcover_source_spheres_csv),
        )
        log_file.write(f"[Batch] selected_quadcover_remesh={quadcover_source_remesh}\n")
        log_file.write(f"[Batch] exported_quadcover_remesh={result.quadcover_remesh_path}\n")
        log_file.write(f"[Batch] selected_quadcover_spheres_csv={quadcover_source_spheres_csv}\n")
        log_file.write(f"[Batch] exported_quadcover_spheres_csv={result.quadcover_spheres_csv_path}\n")
        result.export_seconds += time.perf_counter() - export_start

    result.quadcover_iterations = parse_quadcover_iterations(log_path)
    result.end_time_utc = utc_now_iso()
    result.total_time_seconds = time.perf_counter() - start_time
    result.success = True
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[Batch] end_time_utc={result.end_time_utc}\n"
            f"[Batch] total_time_seconds={result.total_time_seconds:.6f}\n"
            f"[Batch] export_seconds={result.export_seconds:.6f}\n"
        )
    return result


def write_summary(output_dir: Path, results: list[RunResult]) -> Path:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_key",
            "model_name",
            "status",
            "sample_num",
            "quadcover_iterations",
            "start_time_utc",
            "end_time_utc",
            "total_time_seconds",
            "preprocess_seconds",
            "normalize_seconds",
            "sampling_seconds",
            "cwf_seconds",
            "quadcover_seconds",
            "export_seconds",
            "preprocess_exit_code",
            "sampler_exit_code",
            "cwf_exit_code",
            "quadcover_exit_code",
            "model_path",
            "model_log_path",
            "result_dir",
            "cwf_remesh_path",
            "quadcover_remesh_path",
            "quadcover_spheres_csv_path",
            "error",
        ])
        for result in results:
            writer.writerow([
                result.model_key,
                result.model_name,
                "success" if result.success else "failed",
                result.sample_num,
                result.quadcover_iterations,
                result.start_time_utc,
                result.end_time_utc,
                f"{result.total_time_seconds:.3f}",
                f"{result.preprocess_seconds:.3f}",
                f"{result.normalize_seconds:.3f}",
                f"{result.sampling_seconds:.3f}",
                f"{result.cwf_seconds:.3f}",
                f"{result.quadcover_seconds:.3f}",
                f"{result.export_seconds:.3f}",
                result.preprocess_exit_code,
                result.sampler_exit_code,
                result.cwf_exit_code,
                result.quadcover_exit_code,
                result.model_path,
                result.model_log_path,
                result.result_dir,
                result.cwf_remesh_path or "",
                result.quadcover_remesh_path or "",
                result.quadcover_spheres_csv_path or "",
                result.error,
            ])
    return summary_path


def write_total_log(output_dir: Path, results: list[RunResult], batch_total_time: float) -> Path:
    total_log_path = output_dir / "total.log"
    with total_log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "model_name,status,start_time_utc,end_time_utc,total_time_seconds,"
            "preprocess_seconds,normalize_seconds,sampling_seconds,cwf_seconds,"
            "quadcover_seconds,export_seconds,quadcover_iterations,error\n"
        )
        for result in results:
            handle.write(
                f"{result.model_name},"
                f"{'success' if result.success else 'failed'},"
                f"{result.start_time_utc},"
                f"{result.end_time_utc},"
                f"{result.total_time_seconds:.3f},"
                f"{result.preprocess_seconds:.3f},"
                f"{result.normalize_seconds:.3f},"
                f"{result.sampling_seconds:.3f},"
                f"{result.cwf_seconds:.3f},"
                f"{result.quadcover_seconds:.3f},"
                f"{result.export_seconds:.3f},"
                f"{result.quadcover_iterations},"
                f"{result.error}\n"
            )
        handle.write(f"\nbatch_total_time_seconds,{batch_total_time:.3f}\n")
    return total_log_path


def main() -> int:
    options = parse_args()
    if not options.input_dir.is_dir():
        print(f"input directory does not exist: {options.input_dir}", file=sys.stderr)
        return 1
    if not options.sampler_exe.exists():
        print(f"sampler executable does not exist: {options.sampler_exe}", file=sys.stderr)
        return 1
    if not options.cwf_exe.exists():
        print(f"cwf executable does not exist: {options.cwf_exe}", file=sys.stderr)
        return 1
    if not options.quadcover_exe.exists():
        print(f"quadcover_main does not exist: {options.quadcover_exe}", file=sys.stderr)
        return 1
    if not options.preprocess_script.exists():
        print(f"preprocess script does not exist: {options.preprocess_script}", file=sys.stderr)
        return 1

    models = list_models(options.input_dir, options.name_contains, options.limit)
    if not models:
        print(f"no OBJ/OFF models found in {options.input_dir}", file=sys.stderr)
        return 1

    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / ".work").mkdir(parents=True, exist_ok=True)
    try:
        lock_fd, lock_path = acquire_batch_lock(options.output_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sample_desc = (
        str(options.sample_num)
        if options.sample_num is not None
        else "auto(max(8000, vertex_count/3))"
    )
    print(
        f"Batch start: models={len(models)} jobs={options.jobs} "
        f"threads-per-job={options.threads_per_job} sample-num={sample_desc} "
        f"cwf-iters={options.cwf_iterations} output={options.output_dir}"
    )

    try:
        batch_start_time = time.perf_counter()
        results: list[RunResult] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=options.jobs,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_map = {executor.submit(run_one_model, options, model): model for model in models}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                results.append(result)
                status = "success" if result.success else "failed"
                print(
                    f"[{status}] {result.model_name} | "
                    f"sample_num={result.sample_num} | "
                    f"iters={result.quadcover_iterations} | "
                    f"total_time={result.total_time_seconds:.3f}s"
                )

        results.sort(key=lambda item: item.model_key)
        summary_path = write_summary(options.output_dir, results)
        batch_total_time = time.perf_counter() - batch_start_time
        total_log_path = write_total_log(options.output_dir, results, batch_total_time)
        failures = sum(1 for item in results if not item.success)
        print(
            f"Batch finished: success={len(results) - failures} failed={failures} "
            f"total_time={batch_total_time:.3f}s summary={summary_path} total_log={total_log_path}"
        )
        return 0 if failures == 0 else 2
    finally:
        release_batch_lock(lock_fd, lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
