#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyMeshLab preprocessing in an isolated subprocess."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input mesh path.")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path.")
    parser.add_argument(
        "--merge-close-threshold",
        type=float,
        default=0.001,
        help=(
            "PyMeshLab merge-close threshold in PercentageValue units. "
            "Default: 0.001."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import pymeshlab  # type: ignore
    except ModuleNotFoundError:
        print(
            "pymeshlab is required for preprocessing; install it first, "
            "e.g. 'pip install pymeshlab'",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(args.input))
        if args.merge_close_threshold > 0.0:
            ms.meshing_merge_close_vertices(
                threshold=pymeshlab.PercentageValue(args.merge_close_threshold)
            )
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_poly_to_tri()
        ms.save_current_mesh(str(args.output))
    except Exception as exc:  # noqa: BLE001
        print(f"PyMeshLab preprocessing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Preprocessed mesh saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
