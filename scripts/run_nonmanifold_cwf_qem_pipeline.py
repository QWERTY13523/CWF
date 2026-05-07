#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从原始 OBJ 开始，自动执行：
1) 使用 PyMeshLab Poisson disk sampling 采样 k 个点，并启用 Precise num of points
2) 调用已编译好的 bin/quadcover_main 执行 CWF warm start + QEM + hinge loss

说明：
- 这是一个 Python 驱动脚本，不再自动生成、配置或编译 C++ helper。
- CWF/QuadCover 直接复用仓库里已经编译好的可执行程序。
- 采样部分使用 PyMeshLab 的 generate_sampling_poisson_disk(..., exactnumflag=True)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def pymeshlab_poisson_sample_xyz(input_mesh: Path, output_xyz: Path, sample_num: int) -> None:
    print(
        "[RUN] pymeshlab generate_sampling_poisson_disk "
        f"input={input_mesh} output={output_xyz} samplenum={sample_num} exactnumflag=True"
    )
    try:
        import pymeshlab  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pymeshlab is required for Poisson disk sampling; install it first, "
            "e.g. 'pip install pymeshlab'"
        ) from exc

    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_mesh))
    ms.generate_sampling_poisson_disk(
        samplenum=sample_num,
        exactnumflag=True,
    )
    written_samples = ms.current_mesh().vertex_number()
    ms.save_current_mesh(str(output_xyz), save_vertex_normal=True)
    if written_samples != sample_num:
        print(
            "[WARN] PyMeshLab precise sampling returned "
            f"{written_samples} points for requested samplenum={sample_num}"
        )
    print(f"Sample number  : {written_samples}")
    print(f"Saved to       : {output_xyz}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="处理模型 -> 采样 k 点 -> CWF -> QEM+hinge 的完整测试流程")
    p.add_argument("--project-root", default=".", help="BGAL 工程根目录。Default: .")
    p.add_argument("--vcglib-dir", default="", help="保留兼容旧命令；PyMeshLab 采样不再使用 vcglib")
    p.add_argument("--input-obj", required=True, help="原始输入 OBJ")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--k", type=int, required=True, help="采样点数")
    p.add_argument("--model-name", default="model", help="输出命名前缀")
    p.add_argument("--seed", type=int, default=0, help="保留兼容旧命令；PyMeshLab 采样当前不使用")
    p.add_argument("--cwf-iters", type=int, default=50, help="CWF 迭代次数")
    p.add_argument("--quad-iters", type=int, default=0, help="保留兼容旧命令；当前不限制 QuadCover 外层迭代次数")
    p.add_argument("--export-each-iter", action="store_true", help="QuadCover 每若干步导出一次")
    p.add_argument("--export-interval", type=int, default=50, help="保留兼容旧命令；quadcover_main 当前不使用")
    p.add_argument("--cwf-export-process", action="store_true", help="保留兼容旧命令；quadcover_main 当前不使用")
    p.add_argument("--processed-obj", default="", help="若已知处理后的模型路径，可作为 surface 输入")
    p.add_argument("--workspace", default="", help="保留兼容旧命令；当前不再使用")
    p.add_argument("--build-type", default="Release", choices=["Release", "RelWithDebInfo", "Debug", "MinSizeRel"], help="保留兼容旧命令；当前不再使用")
    p.add_argument("--cmake-prefix-path", action="append", default=[], help="保留兼容旧命令；当前不再使用")
    p.add_argument("--cmake-arg", action="append", default=[], help="保留兼容旧命令；当前不再使用")
    p.add_argument("--force-configure", action="store_true", help="保留兼容旧命令；当前不再 configure")
    p.add_argument("--quadcover-exe", default="", help="已编译 quadcover_main 路径；默认 project-root/bin/quadcover_main")
    p.add_argument("--threads", type=int, default=0, help="传给 quadcover_main 的线程数；0 表示使用程序默认值")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    input_obj = Path(args.input_obj).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_obj = Path(args.processed_obj).resolve() if args.processed_obj else input_obj
    sampled_xyz = output_dir / f"{args.model_name}_sampled_{args.k}.xyz"
    quadcover_exe = (
        Path(args.quadcover_exe).resolve()
        if args.quadcover_exe
        else (project_root / "bin" / "quadcover_main")
    )

    if not quadcover_exe.exists():
        raise FileNotFoundError(f"quadcover_main executable not found: {quadcover_exe}")

    pymeshlab_poisson_sample_xyz(surface_obj, sampled_xyz, args.k)

    quadcover_cmd = [
        str(quadcover_exe),
        "--surface", str(surface_obj),
        "--input", str(sampled_xyz),
        "--output", str(output_dir),
        "--name", str(args.model_name),
        "--cwf-iters", str(args.cwf_iters),
    ]
    if args.threads > 0:
        quadcover_cmd.extend(["--threads", str(args.threads)])
    if not args.export_each_iter:
        quadcover_cmd.append("--final-only")
    run(quadcover_cmd)

    manifest = {
        "input_obj": str(input_obj),
        "sampling_method": "pymeshlab.generate_sampling_poisson_disk(exactnumflag=True)",
        "surface_obj": str(surface_obj),
        "sampled_xyz": str(sampled_xyz),
        "quadcover_exe": str(quadcover_exe),
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / f"{args.model_name}_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[DONE] pipeline finished.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] subprocess failed with return code {e.returncode}", file=sys.stderr)
        raise
