#!/usr/bin/env python3
import argparse
import math
import statistics
from typing import Dict, List, Optional, Set, Tuple


Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Edge = Tuple[int, int]


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


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def edge_key(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def read_obj(path: str) -> Tuple[List[Vec3], List[Tri]]:
    vertices: List[Vec3] = []
    faces: List[Tri] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                idx: List[int] = []
                valid = True
                for token in parts:
                    head = token.split("/")[0]
                    if not head:
                        valid = False
                        break
                    k = int(head)
                    # OBJ index is 1-based; negative indices are relative to current list end.
                    if k > 0:
                        idx.append(k - 1)
                    else:
                        idx.append(len(vertices) + k)
                if not valid:
                    continue
                # Fan triangulation for polygons.
                v0 = idx[0]
                for i in range(1, len(idx) - 1):
                    faces.append((v0, idx[i], idx[i + 1]))

    return vertices, faces


def face_unit_normal(vertices: List[Vec3], tri: Tri) -> Optional[Vec3]:
    a = vertices[tri[0]]
    b = vertices[tri[1]]
    c = vertices[tri[2]]
    n = vcross(vsub(b, a), vsub(c, a))
    nn = vnorm(n)
    if nn <= 1e-18:
        return None
    return (n[0] / nn, n[1] / nn, n[2] / nn)


def build_edge_to_faces(faces: List[Tri]) -> Dict[Edge, List[int]]:
    edge_faces: Dict[Edge, List[int]] = {}
    for fi, tri in enumerate(faces):
        i, j, k = tri
        for a, b in ((i, j), (j, k), (k, i)):
            e = edge_key(a, b)
            edge_faces.setdefault(e, []).append(fi)
    return edge_faces


def collect_feature_edges(
    vertices: List[Vec3], faces: List[Tri], feature_angle_deg: float, use_abs_dot: bool
) -> Set[Edge]:
    edge_faces = build_edge_to_faces(faces)
    face_normals: List[Optional[Vec3]] = [face_unit_normal(vertices, tri) for tri in faces]
    feature_edges: Set[Edge] = set()

    for e, adj_faces in edge_faces.items():
        if len(adj_faces) != 2:
            # Boundary edge or non-manifold edge: treat as feature.
            feature_edges.add(e)
            continue

        f0, f1 = adj_faces
        n0 = face_normals[f0]
        n1 = face_normals[f1]
        if n0 is None or n1 is None:
            continue
        dot = vdot(n0, n1)
        if use_abs_dot:
            dot = abs(dot)
        c = clamp(dot, -1.0, 1.0)
        ang = math.degrees(math.acos(c))
        if ang > feature_angle_deg:
            feature_edges.add(e)

    return feature_edges


def edge_length(vertices: List[Vec3], i: int, j: int) -> float:
    return vnorm(vsub(vertices[i], vertices[j]))


def triangle_area(vertices: List[Vec3], tri: Tri) -> float:
    a = vertices[tri[0]]
    b = vertices[tri[1]]
    c = vertices[tri[2]]
    return 0.5 * vnorm(vcross(vsub(b, a), vsub(c, a)))


def triangle_max_altitude(vertices: List[Vec3], tri: Tri) -> float:
    i, j, k = tri
    l0 = edge_length(vertices, j, k)
    l1 = edge_length(vertices, i, k)
    l2 = edge_length(vertices, i, j)
    lmin = min(l0, l1, l2)
    if lmin <= 1e-18:
        return 0.0
    area = triangle_area(vertices, tri)
    if area <= 1e-18:
        return 0.0
    # h = 2A / base; max altitude corresponds to the shortest base.
    return (2.0 * area) / lmin


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0.0:
        return sorted_vals[0]
    if p >= 100.0:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    t = pos - lo
    return sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    s = sorted(values)
    return {
        "count": float(len(values)),
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(s),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p05": percentile(s, 5.0),
        "p25": percentile(s, 25.0),
        "p75": percentile(s, 75.0),
        "p95": percentile(s, 95.0),
    }


def print_summary(title: str, stats: Dict[str, float]) -> None:
    print(f"  {title}:")
    print(f"    count: {int(stats['count'])}")
    print(f"    min: {stats['min']:.10f}")
    print(f"    p05: {stats['p05']:.10f}")
    print(f"    p25: {stats['p25']:.10f}")
    print(f"    median: {stats['median']:.10f}")
    print(f"    mean: {stats['mean']:.10f}")
    print(f"    p75: {stats['p75']:.10f}")
    print(f"    p95: {stats['p95']:.10f}")
    print(f"    max: {stats['max']:.10f}")
    print(f"    std: {stats['std']:.10f}")


def save_histogram(
    all_values: List[float], feature_values: List[float], output_png: str, bins: int
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    plt.figure(figsize=(10, 6))
    plt.hist(
        all_values,
        bins=max(5, bins),
        alpha=0.55,
        label="All triangles",
        color="#1f77b4",
    )
    if feature_values:
        plt.hist(
            feature_values,
            bins=max(5, bins),
            alpha=0.55,
            label="Triangles with >=1 feature-edge vertex",
            color="#d62728",
        )
    plt.xlabel("Triangle max altitude")
    plt.ylabel("Count")
    plt.title("Triangle Max Altitude Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "统计两类三角形的最大高: "
            "1) 至少一个顶点在特征边上的三角形; 2) 所有三角形。"
        )
    )
    parser.add_argument("mesh", help="输入 OBJ 网格路径")
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=30.0,
        help="特征边阈值（二面角，单位度），默认 30",
    )
    parser.add_argument(
        "--no-abs-dot",
        action="store_true",
        help="不对法线点积取绝对值（当 OBJ 面朝向一致时可关闭）",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="可选：输出分布直方图 PNG 路径（需要 matplotlib）",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="直方图柱数，默认 60",
    )
    args = parser.parse_args()

    vertices, faces = read_obj(args.mesh)
    if not vertices or not faces:
        print("网格为空或读取失败。")
        return 1

    feature_edges = collect_feature_edges(
        vertices, faces, args.feature_angle, use_abs_dot=not args.no_abs_dot
    )
    feature_vertices: Set[int] = set()
    for i, j in feature_edges:
        feature_vertices.add(i)
        feature_vertices.add(j)

    all_heights: List[float] = []
    feature_heights: List[float] = []

    for tri in faces:
        h = triangle_max_altitude(vertices, tri)
        all_heights.append(h)
        if (
            tri[0] in feature_vertices
            or tri[1] in feature_vertices
            or tri[2] in feature_vertices
        ):
            feature_heights.append(h)

    all_stats = summarize(all_heights)
    feature_stats = summarize(feature_heights)

    print("Feature Triangle Height Report")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(faces)}")
    print(f"  Feature edges: {len(feature_edges)} (angle > {args.feature_angle} deg or boundary)")
    print(f"  Feature-edge-vertex triangles: {int(feature_stats['count'])}")
    print_summary("All triangles max-altitude stats", all_stats)
    if feature_heights:
        print_summary(
            "Feature-edge-vertex triangles max-altitude stats", feature_stats
        )
    else:
        print("  Feature-edge-vertex triangles max-altitude stats: N/A (none)")

    if args.plot:
        ok = save_histogram(all_heights, feature_heights, args.plot, args.bins)
        if ok:
            print(f"  Histogram saved: {args.plot}")
        else:
            print("  Histogram not generated: matplotlib is not available.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
