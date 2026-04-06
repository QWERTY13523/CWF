#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import multiprocessing as mp
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
DEFAULT_INPUT_CANDIDATES = [
    REPO_ROOT / "ABC",
    Path("/home/yiming/research/CWFSampling/ABC"),
]
DEFAULT_SAMPLER_CANDIDATES = [
    Path("/home/yiming/research/CWFSampling/build/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/build-vcpkg/vcg_poisson_sampling"),
    Path("/home/yiming/research/CWFSampling/cmake-build-debug/vcg_poisson_sampling"),
]


@dataclass
class BatchOptions:
    input_dir: Path
    output_dir: Path
    sampler_exe: Path
    quadcover_exe: Path
    preprocess_script: Path
    sample_num: int | None
    merge_close_threshold: float
    jobs: int
    threads_per_job: int
    seed: int


@dataclass
class RunResult:
    model_key: str
    model_name: str
    model_path: Path
    preprocessed_mesh: Path
    normalized_mesh: Path
    sample_points: Path
    log_path: Path
    model_log_path: Path
    result_dir: Path
    sample_num: int = -1
    quadcover_iterations: int = 0
    total_time_seconds: float = 0.0
    preprocess_exit_code: int = -1
    sampler_exit_code: int = -1
    quadcover_exit_code: int = -1
    success: bool = False
    error: str = ""


ITER_PATTERN = re.compile(r"\[QuadCoverLike\]\[Adam\] iter=(\d+)")


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> BatchOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test QuadCover on all OBJ/OFF models in ABC: "
            "normalize -> CWFSampling -> CWF warm start -> QuadCover"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Folder containing OBJ/OFF models. Default: ABC or ~/research/CWFSampling/ABC")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "quadResult",
                        help="Output root. Default: ./quadResult")
    parser.add_argument("--sampler-exe", type=Path, default=None,
                        help="Path to vcg_poisson_sampling executable.")
    parser.add_argument("--quadcover-exe", type=Path, default=REPO_ROOT / "bin" / "quadcover_main",
                        help="Path to quadcover_main executable.")
    parser.add_argument("--sample-num", type=positive_int, default=None,
                        help="Fixed number of CWFSampling points per model. Default: max(8000, vertex_count / 3)")
    parser.add_argument("--merge-close-threshold", type=float, default=0.001,
                        help=(
                            "PyMeshLab merge-close threshold in PercentageValue units. "
                            "Default: 0.001 (very conservative). "
                            "Note: 1.0 means 1%% of bbox scale and is usually far too large."
                        ))
    parser.add_argument("--jobs", type=positive_int, default=max(1, min(4, os.cpu_count() or 1)),
                        help="Number of models processed in parallel.")
    parser.add_argument("--threads-per-job", type=positive_int, default=None,
                        help="Threads used inside one quadcover_main process.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed forwarded to CWFSampling. Default: 0")
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
        quadcover_exe=args.quadcover_exe.resolve(),
        preprocess_script=PREPROCESS_SCRIPT.resolve(),
        sample_num=args.sample_num,
        merge_close_threshold=args.merge_close_threshold,
        jobs=args.jobs,
        threads_per_job=threads_per_job,
        seed=args.seed,
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


def list_models(input_dir: Path) -> list[Path]:
    models = [
        path for path in input_dir.rglob("*")
        if path.is_file() and lower_ext(path) in {".obj", ".off"}
    ]
    return sorted(models)


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


def create_model_log_link(source_log_path: Path, model_log_path: Path) -> None:
    source_log_path.parent.mkdir(parents=True, exist_ok=True)
    source_log_path.touch(exist_ok=True)
    if model_log_path.exists() or model_log_path.is_symlink():
        model_log_path.unlink()
    try:
        model_log_path.symlink_to(source_log_path)
    except OSError:
        if model_log_path.exists():
            model_log_path.unlink()
        model_log_path.hardlink_to(source_log_path)


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


def run_one_model(options: BatchOptions, model_path: Path) -> RunResult:
    start_time = time.perf_counter()
    model_key = make_model_key(options.input_dir, model_path)
    model_name = model_path.stem
    sample_num = -1

    log_dir = options.output_dir / "quadCover"
    work_dir = options.output_dir / ".work" / model_key
    result_dir = options.output_dir / model_key
    log_path = log_dir / f"{model_key}.log"
    model_log_path = result_dir / f"{model_name}.log"
    preprocessed_mesh = work_dir / f"{model_name}_merged.obj"
    normalized_mesh = work_dir / f"{model_name}_normalized.obj"
    sample_points = work_dir / f"{model_name}_inputPoints.xyz"

    result = RunResult(
        model_key=model_key,
        model_name=model_name,
        model_path=model_path,
        preprocessed_mesh=preprocessed_mesh,
        normalized_mesh=normalized_mesh,
        sample_points=sample_points,
        log_path=log_path,
        model_log_path=model_log_path,
        result_dir=result_dir,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    create_model_log_link(log_path, model_log_path)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[Batch] model={model_path}\n")
        log_file.write(f"[Batch] model_log={model_log_path}\n")
        log_file.write(f"[Batch] preprocessed_mesh={preprocessed_mesh}\n")
        log_file.write(f"[Batch] normalized_mesh={normalized_mesh}\n")
        log_file.write(f"[Batch] sample_points={sample_points}\n")
        log_file.write(f"[Batch] result_dir={result_dir}\n")
        log_file.write(f"[Batch] merge_close_threshold={options.merge_close_threshold}\n")
        log_file.write(f"[Batch] seed={options.seed}\n\n")

        preprocess_command = preprocess_cmd(options, model_path, preprocessed_mesh)
        log_file.write("[Batch] Preprocess command:\n")
        log_file.write(" ".join(preprocess_command) + "\n\n")
        log_file.flush()
        preprocess_proc = subprocess.run(
            preprocess_command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result.preprocess_exit_code = preprocess_proc.returncode
        if preprocess_proc.returncode != 0:
            result.total_time_seconds = time.perf_counter() - start_time
            result.error = (
                f"preprocess failed with exit code {preprocess_proc.returncode}"
            )
            return result

        try:
            vertices, faces = load_mesh(preprocessed_mesh)
            normalized_vertices = normalize_vertices(vertices)
            sample_num = resolve_sample_num(options, len(vertices))
            sample_points = work_dir / f"n{sample_num}_{model_name}_inputPoints.xyz"
            result.sample_num = sample_num
            result.sample_points = sample_points
            write_obj_mesh(normalized_mesh, normalized_vertices, faces)
        except Exception as exc:  # noqa: BLE001
            result.total_time_seconds = time.perf_counter() - start_time
            result.error = f"normalize failed: {exc}"
            return result

        log_file.write(f"[Batch] sample_num={sample_num}\n\n")

        sampler_cmd = [
            str(options.sampler_exe),
            str(normalized_mesh),
            str(sample_points),
            str(sample_num),
            str(options.seed),
        ]
        log_file.write("[Batch] CWFSampling command:\n")
        log_file.write(" ".join(sampler_cmd) + "\n\n")
        log_file.flush()
        sampler_proc = subprocess.run(
            sampler_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result.sampler_exit_code = sampler_proc.returncode
        if sampler_proc.returncode != 0:
            result.total_time_seconds = time.perf_counter() - start_time
            result.error = f"CWFSampling failed with exit code {sampler_proc.returncode}"
            return result

        log_file.write("\n[Batch] QuadCover command:\n")
        quadcover_cmd = [
            str(options.quadcover_exe),
            "--surface",
            str(normalized_mesh),
            "--input",
            str(sample_points),
            "--name",
            model_name,
            "--output",
            str(result_dir),
            "--threads",
            str(options.threads_per_job),
            "--final-only",
        ]
        log_file.write(" ".join(quadcover_cmd) + "\n\n")
        log_file.flush()
        quadcover_proc = subprocess.run(
            quadcover_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result.quadcover_exit_code = quadcover_proc.returncode
        if quadcover_proc.returncode != 0:
            result.total_time_seconds = time.perf_counter() - start_time
            result.error = f"quadcover_main failed with exit code {quadcover_proc.returncode}"
            return result

    result.quadcover_iterations = parse_quadcover_iterations(log_path)
    result.total_time_seconds = time.perf_counter() - start_time
    result.success = True
    return result


def write_summary(output_dir: Path, results: list[RunResult]) -> Path:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_key",
            "model_name",
            "status",
            "preprocess_exit_code",
            "sampler_exit_code",
            "quadcover_exit_code",
            "sample_num",
            "quadcover_iterations",
            "total_time_seconds",
            "model_path",
            "preprocessed_mesh",
            "normalized_mesh",
            "sample_points",
            "log_path",
            "model_log_path",
            "result_dir",
            "error",
        ])
        for result in results:
            writer.writerow([
                result.model_key,
                result.model_name,
                "success" if result.success else "failed",
                result.preprocess_exit_code,
                result.sampler_exit_code,
                result.quadcover_exit_code,
                result.sample_num,
                result.quadcover_iterations,
                f"{result.total_time_seconds:.3f}",
                result.model_path,
                result.preprocessed_mesh,
                result.normalized_mesh,
                result.sample_points,
                result.log_path,
                result.model_log_path,
                result.result_dir,
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
    if not options.quadcover_exe.exists():
        print(f"quadcover_main does not exist: {options.quadcover_exe}", file=sys.stderr)
        return 1
    if not options.preprocess_script.exists():
        print(f"preprocess script does not exist: {options.preprocess_script}", file=sys.stderr)
        return 1

    models = list_models(options.input_dir)
    if not models:
        print(f"no OBJ/OFF models found in {options.input_dir}", file=sys.stderr)
        return 1

    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "quadCover").mkdir(parents=True, exist_ok=True)
    (options.output_dir / ".work").mkdir(parents=True, exist_ok=True)

    sample_desc = (
        str(options.sample_num)
        if options.sample_num is not None
        else "auto(max(8000, vertex_count/3))"
    )
    print(
        f"Batch start: models={len(models)} jobs={options.jobs} "
        f"threads-per-job={options.threads_per_job} sample-num={sample_desc} "
        f"merge-close-threshold={options.merge_close_threshold}"
    )

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
                f"[{status}] {result.model_key} | "
                f"iters={result.quadcover_iterations} | "
                f"total_time={result.total_time_seconds:.3f}s"
            )

    results.sort(key=lambda item: item.model_key)
    summary_path = write_summary(options.output_dir, results)
    failures = sum(1 for item in results if not item.success)
    batch_total_time = time.perf_counter() - batch_start_time
    print(
        f"Batch finished: success={len(results) - failures} "
        f"failed={failures} total_time={batch_total_time:.3f}s summary={summary_path}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
