#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "pymeshlab_preprocess.py"
DEFAULT_SAMPLER_EXE = Path("/home/yiming/research/CWFSampling/build/vcg_poisson_sampling")
DEFAULT_CWF_EXE = REPO_ROOT / "bin" / "cwf"
DEFAULT_QUADCOVER_EXE = REPO_ROOT / "bin" / "quadcover_main"


@dataclass
class ModelConfig:
    path: Path
    name: str
    bbox_size: float
    sample_num: int
    base_learning_rate: float


@dataclass
class RunResult:
    model: ModelConfig
    success: bool = False
    error: str = ""
    preprocess_exit_code: int = -1
    sampler_exit_code: int = -1
    cwf_exit_code: int = -1
    quadcover_exit_code: int = -1
    total_time_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gallery models with per-model sample counts and bbox-scaled QuadCover lr."
    )
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "gallery")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "result" / "gallery_variable_lr")
    parser.add_argument("--sampler-exe", type=Path, default=DEFAULT_SAMPLER_EXE)
    parser.add_argument("--cwf-exe", type=Path, default=DEFAULT_CWF_EXE)
    parser.add_argument("--quadcover-exe", type=Path, default=DEFAULT_QUADCOVER_EXE)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)
    parser.add_argument("--default-sample-num", type=int, default=30000)
    parser.add_argument("--car-sample-num", type=int, default=300000)
    parser.add_argument("--cwf-max-iterations", type=int, default=50)
    parser.add_argument("--max-outer-iters", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--threads-per-job", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_logged(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[exit_code] {proc.returncode}\n\n")
        return proc.returncode


def load_obj_bbox(path: Path) -> float:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if not raw.startswith("v "):
                continue
            parts = raw.split()
            if len(parts) < 4:
                continue
            values = [float(parts[1]), float(parts[2]), float(parts[3])]
            for i, value in enumerate(values):
                mins[i] = min(mins[i], value)
                maxs[i] = max(maxs[i], value)
            count += 1
    if count == 0:
        raise ValueError(f"{path} has no vertices")
    bbox_size = max(maxs[i] - mins[i] for i in range(3))
    if not math.isfinite(bbox_size) or bbox_size <= 0.0:
        raise ValueError(f"{path} has invalid bbox size {bbox_size}")
    return bbox_size


def collect_models(args: argparse.Namespace) -> list[ModelConfig]:
    models: list[ModelConfig] = []
    if args.input_dir.is_file():
        paths = [args.input_dir]
    else:
        paths = sorted(args.input_dir.rglob("*.obj"))
    for path in paths:
        if path.suffix.lower() != ".obj":
            continue
        name = path.stem
        if name.lower() == "woman" or "woman" in {part.lower() for part in path.parts}:
            continue
        bbox_size = load_obj_bbox(path)
        sample_num = args.car_sample_num if name.lower() == "car" else args.default_sample_num
        models.append(
            ModelConfig(
                path=path.resolve(),
                name=name,
                bbox_size=bbox_size,
                sample_num=sample_num,
                base_learning_rate=args.base_learning_rate,
            )
        )
    return models


def make_job_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(threads)
    env["OMP_NUM_THREADS"] = value
    env["OPENBLAS_NUM_THREADS"] = value
    env["MKL_NUM_THREADS"] = value
    env["NUMEXPR_NUM_THREADS"] = value
    return env


def collect_cwf_output(cwf_dir: Path, suffix: str) -> Path | None:
    candidates = [
        path
        for path in cwf_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix) and "Iter" not in path.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_one(args: argparse.Namespace, model: ModelConfig) -> RunResult:
    started = time.perf_counter()
    result = RunResult(model=model)
    model_dir = args.output_dir / model.name
    work_dir = args.output_dir / ".work" / model.name

    if model_dir.exists() or work_dir.exists():
        if not args.force:
            result.error = "output exists; pass --force to overwrite"
            return result
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)

    preprocess_dir = work_dir / "preprocess"
    sample_dir = work_dir / "sampling"
    cwf_dir = work_dir / "cwf"
    quadcover_dir = work_dir / "quadcover"
    for path in (model_dir, preprocess_dir, sample_dir, cwf_dir, quadcover_dir):
        path.mkdir(parents=True, exist_ok=True)

    preprocessed_mesh = preprocess_dir / f"{model.name}_merged.obj"
    sample_points = sample_dir / f"n{model.sample_num}_{model.name}_inputPoints.xyz"
    env = make_job_env(args.threads_per_job)

    result.preprocess_exit_code = run_logged(
        [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--input",
            str(model.path),
            "--output",
            str(preprocessed_mesh),
            "--merge-close-threshold",
            "0.001",
        ],
        work_dir / "preprocess.log",
        REPO_ROOT,
    )
    if result.preprocess_exit_code != 0:
        result.error = f"preprocess failed with exit code {result.preprocess_exit_code}"
        result.total_time_seconds = time.perf_counter() - started
        return result

    result.sampler_exit_code = run_logged(
        [
            str(args.sampler_exe),
            str(preprocessed_mesh),
            str(sample_points),
            str(model.sample_num),
            str(args.seed),
        ],
        work_dir / "sampling.log",
        REPO_ROOT,
        env,
    )
    if result.sampler_exit_code != 0:
        result.error = f"sampling failed with exit code {result.sampler_exit_code}"
        result.total_time_seconds = time.perf_counter() - started
        return result

    result.cwf_exit_code = run_logged(
        [
            str(args.cwf_exe),
            str(preprocessed_mesh),
            str(sample_points),
            str(args.cwf_max_iterations),
        ],
        work_dir / "cwf.log",
        cwf_dir,
        env,
    )
    if result.cwf_exit_code != 0:
        result.error = f"cwf failed with exit code {result.cwf_exit_code}"
        result.total_time_seconds = time.perf_counter() - started
        return result

    cwf_remesh = collect_cwf_output(cwf_dir, "Remesh.obj")
    cwf_points = collect_cwf_output(cwf_dir, "_Points.xyz")
    if cwf_remesh is None or cwf_points is None:
        result.error = "cwf outputs not found"
        result.total_time_seconds = time.perf_counter() - started
        return result

    shutil.copy2(cwf_remesh, model_dir / "cwf_remesh.obj")
    result.quadcover_exit_code = run_logged(
        [
            str(args.quadcover_exe),
            "--surface",
            str(cwf_remesh),
            "--input",
            str(cwf_points),
            "--output",
            str(quadcover_dir),
            "--name",
            model.name,
            "--threads",
            str(args.threads_per_job),
            "--cwf-iters",
            "0",
            "--max-outer-iters",
            str(args.max_outer_iters),
            "--learning-rate",
            f"{model.base_learning_rate:.17g}",
            "--final-only",
        ],
        work_dir / "quadcover.log",
        REPO_ROOT,
        env,
    )
    if result.quadcover_exit_code != 0:
        result.error = f"quadcover failed with exit code {result.quadcover_exit_code}"
        result.total_time_seconds = time.perf_counter() - started
        return result

    for path in quadcover_dir.iterdir():
        if path.is_file() and ("_Remesh.obj" in path.name or "_Spheres.csv" in path.name):
            shutil.copy2(path, model_dir / path.name)

    result.success = True
    result.total_time_seconds = time.perf_counter() - started
    return result


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    for executable in (args.sampler_exe, args.cwf_exe, args.quadcover_exe):
        if not executable.exists():
            print(f"missing executable: {executable}", file=sys.stderr)
            return 1

    models = collect_models(args)
    if not models:
        print(f"no gallery OBJ models found under {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Gallery start: models={len(models)} jobs={args.jobs} "
        f"threads-per-job={args.threads_per_job}"
    )
    for model in models:
        print(
            f"[config] {model.name} sample={model.sample_num} "
            f"bbox={model.bbox_size:.12g} base_lr={model.base_learning_rate:.12g}"
        )

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {executor.submit(run_one, args, model): model for model in models}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            status = "success" if result.success else "failed"
            print(
                f"[{status}] {result.model.name} "
                f"time={result.total_time_seconds:.3f}s {result.error}"
            )

    results.sort(key=lambda item: item.model.name.lower())
    summary_path = args.output_dir / "gallery_variable_lr_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model",
            "status",
            "bbox_size",
            "sample_num",
            "base_learning_rate",
            "preprocess_exit_code",
            "sampler_exit_code",
            "cwf_exit_code",
            "quadcover_exit_code",
            "total_time_seconds",
            "input_path",
            "error",
        ])
        for result in results:
            writer.writerow([
                result.model.name,
                "success" if result.success else "failed",
                f"{result.model.bbox_size:.17g}",
                result.model.sample_num,
                f"{result.model.base_learning_rate:.17g}",
                result.preprocess_exit_code,
                result.sampler_exit_code,
                result.cwf_exit_code,
                result.quadcover_exit_code,
                f"{result.total_time_seconds:.3f}",
                result.model.path,
                result.error,
            ])
    failures = sum(not result.success for result in results)
    print(f"Gallery finished: success={len(results) - failures} failed={failures} summary={summary_path}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
