#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import re
import shlex
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

ITER_RE = re.compile(r"\[QuadCoverLike\]\[Adam\] iter=(\d+)")
ACT_RE = re.compile(r"Act\(prev/curr\)=([0-9+-]+)/([0-9+-]+)")
QEM_RE = re.compile(r"QEM\(prev/curr\)=([^\s|]+)/([^\s|]+)")
HINGE_RE = re.compile(r"hingeRaw\(prev/curr\)=([^\s|]+)/([^\s|]+)")
STOP_HINGE_RE = re.compile(r"hingeRaw=([^\s|]+)")
STOP_ACT_RE = re.compile(r"active_quads=([0-9+-]+)")

Vec3 = tuple[float, float, float]
Tri = tuple[int, int, int]
Edge = tuple[int, int]


@dataclass(frozen=True)
class Config:
    name: str
    group: str
    energy_mode: str = "full"
    weight_schedule: bool = True
    tangential_perturb: bool = True
    init: str = "cwf50"


@dataclass
class Options:
    input_dir: Path
    output_dir: Path
    sampler_exe: Path
    cwf_exe: Path
    quadcover_exe: Path
    sample_num: int | None
    merge_close_threshold: float
    cwf_iterations: int
    max_outer_iters: int
    jobs: int
    threads_per_job: int
    seed: int
    limit: int | None
    name_contains: str
    exclude_names: tuple[str, ...]
    eq_epsilon: float
    feature_angle: float
    no_abs_dot: bool
    force: bool


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


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_args() -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABC ablations for energy terms, weight scheduling, tangential "
            "perturbation, sharing one CWF50 initialization per model."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "ablation")
    parser.add_argument("--sampler-exe", type=Path, default=None)
    parser.add_argument("--cwf-exe", type=Path, default=REPO_ROOT / "bin" / "cwf")
    parser.add_argument("--quadcover-exe", type=Path, default=REPO_ROOT / "bin" / "quadcover_main")
    parser.add_argument("--sample-num", type=positive_int, default=None)
    parser.add_argument("--merge-close-threshold", type=float, default=0.001)
    parser.add_argument("--cwf-iterations", type=non_negative_int, default=50)
    parser.add_argument("--max-outer-iters", type=positive_int, default=750)
    parser.add_argument("--jobs", type=positive_int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--threads-per-job", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--name-contains", default="")
    parser.add_argument("--exclude-names", default="")
    parser.add_argument("--eq-epsilon", type=float, default=1e-8)
    parser.add_argument("--feature-angle", type=float, default=30.0)
    parser.add_argument("--no-abs-dot", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir or find_first_existing(DEFAULT_INPUT_CANDIDATES)
    if input_dir is None:
        parser.error("cannot find ABC folder; pass --input-dir explicitly")
    sampler_exe = args.sampler_exe or find_first_existing(DEFAULT_SAMPLER_CANDIDATES)
    if sampler_exe is None:
        parser.error("cannot find vcg_poisson_sampling; pass --sampler-exe explicitly")
    threads_per_job = args.threads_per_job or max(1, (os.cpu_count() or 1) // args.jobs)
    exclude_names = tuple(
        sorted({name.strip().lower() for name in args.exclude_names.split(",") if name.strip()})
    )
    return Options(
        input_dir=input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        sampler_exe=sampler_exe.resolve(),
        cwf_exe=args.cwf_exe.resolve(),
        quadcover_exe=args.quadcover_exe.resolve(),
        sample_num=args.sample_num,
        merge_close_threshold=args.merge_close_threshold,
        cwf_iterations=args.cwf_iterations,
        max_outer_iters=args.max_outer_iters,
        jobs=args.jobs,
        threads_per_job=threads_per_job,
        seed=args.seed,
        limit=args.limit,
        name_contains=args.name_contains,
        exclude_names=exclude_names,
        eq_epsilon=args.eq_epsilon,
        feature_angle=args.feature_angle,
        no_abs_dot=args.no_abs_dot,
        force=args.force,
    )


def lower_ext(path: Path) -> str:
    return path.suffix.lower()


def list_models(options: Options) -> list[Path]:
    models = [
        path for path in options.input_dir.rglob("*")
        if path.is_file() and lower_ext(path) in {".obj", ".off"}
    ]
    models.sort()
    if options.name_contains:
        models = [path for path in models if options.name_contains in path.stem]
    if options.exclude_names:
        excluded = set(options.exclude_names)
        models = [path for path in models if path.stem.lower() not in excluded]
    if options.limit is not None:
        models = models[: options.limit]
    return models


def model_key(input_dir: Path, model_path: Path) -> str:
    return "__".join(model_path.relative_to(input_dir).with_suffix("").parts)


def triangulate(indices: list[int]) -> list[Tri]:
    if len(indices) < 3:
        return []
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def read_obj(path: Path) -> tuple[list[Vec3], list[Tri]]:
    vertices: list[Vec3] = []
    faces: list[Tri] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("v "):
                p = raw.strip().split()
                if len(p) >= 4:
                    vertices.append((float(p[1]), float(p[2]), float(p[3])))
            elif raw.startswith("f "):
                face: list[int] = []
                for token in raw.strip().split()[1:]:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    idx = int(head)
                    face.append(len(vertices) + idx if idx < 0 else idx - 1)
                faces.extend(triangulate(face))
    return vertices, faces


def read_off(path: Path) -> tuple[list[Vec3], list[Tri]]:
    lines = [
        line.strip() for line in path.open("r", encoding="utf-8", errors="ignore")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines or lines[0] not in {"OFF", "COFF"}:
        raise ValueError(f"unsupported OFF header in {path}")
    nv, nf = [int(x) for x in lines[1].split()[:2]]
    vertices = [tuple(float(x) for x in lines[2 + i].split()[:3]) for i in range(nv)]
    faces: list[Tri] = []
    for i in range(nf):
        parts = lines[2 + nv + i].split()
        count = int(parts[0])
        faces.extend(triangulate([int(x) for x in parts[1:1 + count]]))
    return vertices, faces


def read_mesh(path: Path) -> tuple[list[Vec3], list[Tri]]:
    return read_off(path) if lower_ext(path) == ".off" else read_obj(path)


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vdot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(a: Vec3) -> float:
    return math.sqrt(vdot(a, a))


def edge_key(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def face_normal(vertices: list[Vec3], tri: Tri) -> Vec3 | None:
    n = vcross(vsub(vertices[tri[1]], vertices[tri[0]]), vsub(vertices[tri[2]], vertices[tri[0]]))
    nn = vnorm(n)
    if nn <= 1e-18:
        return None
    return (n[0] / nn, n[1] / nn, n[2] / nn)


def feature_edges(vertices: list[Vec3], faces: list[Tri], angle_deg: float, use_abs_dot: bool) -> set[Edge]:
    edge_faces: dict[Edge, list[int]] = {}
    for fi, (i, j, k) in enumerate(faces):
        for a, b in ((i, j), (j, k), (k, i)):
            edge_faces.setdefault(edge_key(a, b), []).append(fi)
    normals = [face_normal(vertices, tri) for tri in faces]
    edges: set[Edge] = set()
    for edge, adj in edge_faces.items():
        if len(adj) != 2:
            edges.add(edge)
            continue
        n0, n1 = normals[adj[0]], normals[adj[1]]
        if n0 is None or n1 is None:
            continue
        dot = vdot(n0, n1)
        if use_abs_dot:
            dot = abs(dot)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        if angle > angle_deg:
            edges.add(edge)
    return edges


def sample_edges(vertices: list[Vec3], edges: set[Edge], max_points: int = 20000) -> list[Vec3]:
    if not edges:
        return []
    per_edge = max(2, min(8, max_points // max(1, len(edges))))
    points: list[Vec3] = []
    for i, j in edges:
        a, b = vertices[i], vertices[j]
        for s in range(per_edge):
            t = s / float(per_edge - 1)
            points.append((a[0] * (1 - t) + b[0] * t,
                           a[1] * (1 - t) + b[1] * t,
                           a[2] * (1 - t) + b[2] * t))
    return points


def directed_hausdorff(a: list[Vec3], b: list[Vec3]) -> float:
    if not a or not b:
        return float("nan")
    try:
        from scipy.spatial import cKDTree  # type: ignore
        dists, _ = cKDTree(b).query(a, k=1, workers=-1)
        return float(max(dists))
    except Exception:
        best_max = 0.0
        for p in a:
            best = min(vnorm(vsub(p, q)) for q in b)
            best_max = max(best_max, best)
        return best_max


def feature_hausdorff(source_mesh: Path, result_mesh: Path, angle_deg: float, use_abs_dot: bool) -> float:
    v0, f0 = read_mesh(source_mesh)
    v1, f1 = read_mesh(result_mesh)
    p0 = sample_edges(v0, feature_edges(v0, f0, angle_deg, use_abs_dot))
    p1 = sample_edges(v1, feature_edges(v1, f1, angle_deg, use_abs_dot))
    if not p0 or not p1:
        return float("nan")
    return max(directed_hausdorff(p0, p1), directed_hausdorff(p1, p0))


def mesh_quality(path: Path) -> dict[str, float]:
    vertices, faces = read_mesh(path)
    angles: list[float] = []
    aspects: list[float] = []
    q_values: list[float] = []
    for i, j, k in faces:
        a, b, c = vertices[i], vertices[j], vertices[k]
        lij, ljk, lki = vnorm(vsub(a, b)), vnorm(vsub(b, c)), vnorm(vsub(c, a))
        if min(lij, ljk, lki) <= 1e-18:
            continue
        area = 0.5 * vnorm(vcross(vsub(b, a), vsub(c, a)))
        half_perimeter = 0.5 * (lij + ljk + lki)
        longest_edge = max(lij, ljk, lki)
        denom = half_perimeter * longest_edge
        if area > 1e-18 and denom > 1e-18:
            q_values.append((6.0 / math.sqrt(3.0)) * area / denom)
        def angle(opposite: float, side1: float, side2: float) -> float:
            cosv = (side1 * side1 + side2 * side2 - opposite * opposite) / (2.0 * side1 * side2)
            return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
        angles.extend([angle(ljk, lij, lki), angle(lki, lij, ljk), angle(lij, lki, ljk)])
        aspects.append(max(lij, ljk, lki) / min(lij, ljk, lki))
    if not angles:
        return {
            "min_angle": float("nan"),
            "avg_aspect": float("nan"),
            "p95_aspect": float("nan"),
            "q_min": float("nan"),
        }
    aspects_sorted = sorted(aspects)
    p95 = aspects_sorted[int(0.95 * (len(aspects_sorted) - 1))] if aspects_sorted else float("nan")
    return {
        "min_angle": min(angles),
        "avg_aspect": sum(aspects) / len(aspects) if aspects else float("nan"),
        "p95_aspect": p95,
        "q_min": min(q_values) if q_values else float("nan"),
    }


def parse_log(log_path: Path, eq_epsilon: float) -> tuple[int, int, float, float, int]:
    max_iter = 0
    residual_active = -1
    final_qem = float("nan")
    final_hinge = float("nan")
    eq_converged_iter = -1
    if not log_path.exists():
        return max_iter, residual_active, final_qem, final_hinge, eq_converged_iter
    for line in log_path.open("r", encoding="utf-8", errors="ignore"):
        if m := ITER_RE.search(line):
            max_iter = max(max_iter, int(m.group(1)))
        if m := ACT_RE.search(line):
            residual_active = int(m.group(2))
        if m := QEM_RE.search(line):
            try:
                final_qem = float(m.group(2))
                if eq_converged_iter < 0 and final_qem < eq_epsilon and max_iter > 0:
                    eq_converged_iter = max_iter
            except ValueError:
                pass
        if m := HINGE_RE.search(line):
            try:
                final_hinge = float(m.group(2))
            except ValueError:
                pass
        if m := STOP_HINGE_RE.search(line):
            try:
                final_hinge = float(m.group(1))
            except ValueError:
                pass
        if m := STOP_ACT_RE.search(line):
            residual_active = int(m.group(1))
    return max_iter, residual_active, final_qem, final_hinge, eq_converged_iter


def write_curve_csv(log_path: Path, curve_path: Path) -> None:
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["iteration", "eq", "ena", "active_quads"])
        current_iter = 0
        for line in log_path.open("r", encoding="utf-8", errors="ignore"):
            if m := ITER_RE.search(line):
                current_iter = int(m.group(1))
            qem = QEM_RE.search(line)
            hinge = HINGE_RE.search(line)
            active = ACT_RE.search(line)
            if qem or hinge or active:
                writer.writerow([
                    current_iter,
                    qem.group(2) if qem else "",
                    hinge.group(2) if hinge else "",
                    active.group(2) if active else "",
                ])


def run_logged(command: list[str], log_file, cwd: Path | None = None) -> int:
    if cwd is not None:
        cwd.mkdir(parents=True, exist_ok=True)
        log_file.write(f"[Ablation] cwd={cwd}\n")
    log_file.write(shlex.join(command) + "\n\n")
    log_file.flush()
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode


def sanitized_name(path: Path) -> str:
    return path.name.lstrip("\\/")


def find_artifact(output_dir: Path, suffix: str, prefix: str = "QuadCover_") -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_file()
        and sanitized_name(p).startswith(prefix)
        and sanitized_name(p).endswith(suffix)
    ]
    final = [p for p in candidates if "Iter" not in sanitized_name(p)]
    pool = final or candidates
    if not pool:
        return None
    pool.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return pool[-1]


def config_dir(options: Options, cfg: Config, model_name: str) -> Path:
    return options.output_dir / cfg.name / model_name


def config_log_path(options: Options, cfg: Config, model_name: str) -> Path:
    return config_dir(options, cfg, model_name) / f"{model_name}_{cfg.name}.log"


def has_existing_run(options: Options, cfg: Config, model_name: str) -> bool:
    return config_log_path(options, cfg, model_name).exists()


def row_from_existing_run(options: Options, cfg: Config, model_name: str) -> dict[str, object]:
    cfg_dir = config_dir(options, cfg, model_name)
    log_path = config_log_path(options, cfg, model_name)
    max_iter, active, final_qem, final_hinge, eq_conv_iter = parse_log(log_path, options.eq_epsilon)
    curve_path = cfg_dir / "curve.csv"
    if not curve_path.exists():
        write_curve_csv(log_path, curve_path)
    remesh = find_artifact(cfg_dir, "_Remesh.obj")
    csv_path = find_artifact(cfg_dir, "_Spheres.csv")
    return {
        "model": model_name,
        "group": cfg.group,
        "config": cfg.name,
        "success": remesh is not None,
        "exit_code": "",
        "seconds": "0.000",
        "iterations": max_iter,
        "eq_convergence_iter": eq_conv_iter,
        "final_eq": final_qem,
        "final_ena": final_hinge,
        "residual_violated_quadruples": active,
        "init": cfg.init,
        "energy_mode": cfg.energy_mode,
        "weight_schedule": cfg.weight_schedule,
        "tangential_perturb": cfg.tangential_perturb,
        "remesh": remesh or "",
        "spheres_csv": csv_path or "",
        "log": log_path,
    }


def ordered_rows_with_total(
    cfgs: list[Config],
    rows_by_config: dict[str, dict[str, object]],
    model_total_seconds: float,
) -> list[dict[str, object]]:
    rows = [rows_by_config[cfg.name] for cfg in cfgs]
    for row in rows:
        row["model_total_seconds"] = f"{model_total_seconds:.3f}"
    return rows


def configs() -> list[Config]:
    return [
        Config("no_cwf50", "cwf_initialization", init="sample_points"),
        Config("eq_only", "energy_terms", energy_mode="eq-only"),
        Config("no_schedule", "weight_scheduling", weight_schedule=False),
        Config("no_perturb", "perturbation", tangential_perturb=False),
    ]


def resolve_sample_num(options: Options, mesh_path: Path) -> int:
    if options.sample_num is not None:
        return options.sample_num
    vertices, _ = read_mesh(mesh_path)
    return max(8000, len(vertices) // 3)


def run_one_model(options: Options, model_path: Path) -> list[dict[str, object]]:
    model_start = time.perf_counter()
    key = model_key(options.input_dir, model_path)
    name = model_path.stem
    cfgs = configs()
    prepare_dir = options.output_dir / ".prepare" / name
    result_dirs = [config_dir(options, cfg, name) for cfg in cfgs]
    work_dir = options.output_dir / ".work" / key
    rows_by_config: dict[str, dict[str, object]] = {}
    pending_cfgs = cfgs

    if options.force:
        for path in result_dirs:
            if path.exists():
                shutil.rmtree(path)
    else:
        pending_cfgs = []
        for cfg in cfgs:
            if has_existing_run(options, cfg, name):
                rows_by_config[cfg.name] = row_from_existing_run(options, cfg, name)
            else:
                pending_cfgs.append(cfg)
        if not pending_cfgs:
            return ordered_rows_with_total(cfgs, rows_by_config, time.perf_counter() - model_start)

    if prepare_dir.exists() and options.force:
        shutil.rmtree(prepare_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    prepare_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_log = prepare_dir / f"{name}_prepare.log"
    preprocessed = work_dir / f"{name}_merged.obj"
    sample_points = work_dir / f"{name}_input.xyz"
    cwf_work = work_dir / f"cwf{options.cwf_iterations}"
    cwf_remesh: Path | None = None

    with model_log.open("w", encoding="utf-8") as log:
        if run_logged([
            sys.executable, str(PREPROCESS_SCRIPT), "--input", str(model_path),
            "--output", str(preprocessed), "--merge-close-threshold",
            str(options.merge_close_threshold),
        ], log) != 0:
            raise RuntimeError(f"preprocess failed for {name}")
        sample_num = resolve_sample_num(options, preprocessed)
        sample_points = work_dir / f"n{sample_num}_{name}_input.xyz"
        if run_logged([str(options.sampler_exe), str(preprocessed), str(sample_points), str(sample_num), str(options.seed)], log) != 0:
            raise RuntimeError(f"sampling failed for {name}")
        if any(cfg.init == "cwf50" for cfg in pending_cfgs):
            if run_logged(
                [
                    str(options.cwf_exe),
                    str(preprocessed),
                    str(sample_points),
                    str(options.cwf_iterations),
                ],
                log,
                cwd=cwf_work,
            ) != 0:
                raise RuntimeError(f"CWF{options.cwf_iterations} failed for {name}")
            cwf_remesh = find_artifact(cwf_work, "_Remesh.obj", prefix="Ours_")
            if cwf_remesh is None:
                raise RuntimeError(f"cannot find shared CWF remesh for {name} in {cwf_work}")
            shared_cwf_target = prepare_dir / f"CWF{options.cwf_iterations}_{name}_Remesh.obj"
            shutil.copy2(cwf_remesh, shared_cwf_target)
    for cfg in pending_cfgs:
        cfg_dir = config_dir(options, cfg, name)
        cfg_work = work_dir / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_work.mkdir(parents=True, exist_ok=True)
        log_path = config_log_path(options, cfg, name)
        if cfg.init == "cwf50":
            if cwf_remesh is None:
                raise RuntimeError(f"{cfg.name} requested CWF init but no CWF remesh exists")
            init_path = cwf_remesh
        else:
            init_path = sample_points
        command = [
            str(options.quadcover_exe),
            "--surface", str(preprocessed),
            "--input", str(init_path),
            "--name", f"{name}_{cfg.name}",
            "--output", str(cfg_work),
            "--threads", str(options.threads_per_job),
            "--cwf-iters", "0",
            "--max-outer-iters", str(options.max_outer_iters),
            "--energy-mode", cfg.energy_mode,
            "--final-only",
        ]
        if not cfg.weight_schedule:
            command.append("--no-weight-schedule")
        if not cfg.tangential_perturb:
            command.append("--no-tangential-perturb")

        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            exit_code = run_logged(command, log)
        seconds = time.perf_counter() - start
        max_iter, active, final_qem, final_hinge, eq_conv_iter = parse_log(log_path, options.eq_epsilon)
        write_curve_csv(log_path, cfg_dir / "curve.csv")

        remesh = find_artifact(cfg_work, "_Remesh.obj")
        csv_path = find_artifact(cfg_work, "_Spheres.csv")
        exported_csv: Path | str = ""
        if remesh is not None:
            target_remesh = cfg_dir / remesh.name
            shutil.copy2(remesh, target_remesh)
            remesh = target_remesh
        if csv_path is not None:
            exported_csv = cfg_dir / csv_path.name
            shutil.copy2(csv_path, exported_csv)

        rows_by_config[cfg.name] = {
            "model": name,
            "group": cfg.group,
            "config": cfg.name,
            "success": exit_code == 0,
            "exit_code": exit_code,
            "seconds": f"{seconds:.3f}",
            "iterations": max_iter,
            "eq_convergence_iter": eq_conv_iter,
            "final_eq": final_qem,
            "final_ena": final_hinge,
            "residual_violated_quadruples": active,
            "init": cfg.init,
            "energy_mode": cfg.energy_mode,
            "weight_schedule": cfg.weight_schedule,
            "tangential_perturb": cfg.tangential_perturb,
            "remesh": remesh or "",
            "spheres_csv": exported_csv,
            "log": log_path,
        }
    return ordered_rows_with_total(cfgs, rows_by_config, time.perf_counter() - model_start)


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    summary = output_dir / "ablation_summary.csv"
    fields = [
        "model", "group", "config", "success", "exit_code", "seconds",
        "model_total_seconds",
        "iterations", "eq_convergence_iter", "final_eq", "final_ena",
        "residual_violated_quadruples", "init", "energy_mode",
        "weight_schedule", "tangential_perturb", "remesh", "spheres_csv", "log",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    options = parse_args()
    if not options.cwf_exe.exists():
        print(f"cwf executable not found: {options.cwf_exe}", file=sys.stderr)
        return 2
    if not options.quadcover_exe.exists():
        print(f"quadcover_main executable not found: {options.quadcover_exe}", file=sys.stderr)
        return 2
    models = list_models(options)
    if not models:
        print("no ABC models matched", file=sys.stderr)
        return 1
    options.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Ablation] models={len(models)} output={options.output_dir}")

    all_rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
        future_to_model = {pool.submit(run_one_model, options, path): path for path in models}
        for future in concurrent.futures.as_completed(future_to_model):
            path = future_to_model[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(f"[Ablation] done {path.stem}")
            except Exception as exc:
                print(f"[Ablation] failed {path}: {exc}", file=sys.stderr)
    summary = write_summary(options.output_dir, all_rows)
    print(f"[Ablation] summary={summary}")
    return 0 if all(row.get("success") for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
