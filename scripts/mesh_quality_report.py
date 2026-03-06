#!/usr/bin/env python3
import math
import sys


def read_obj(path):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
                    except ValueError:
                        continue
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                try:
                    idx = [int(p.split("/")[0]) - 1 for p in parts]
                except ValueError:
                    continue
                v0 = idx[0]
                for i in range(1, len(idx) - 1):
                    faces.append((v0, idx[i], idx[i + 1]))
    return verts, faces


def edge_len(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def triangle_angles(a, b, c):
    # edges opposite to vertices a,b,c
    la = edge_len(b, c)
    lb = edge_len(a, c)
    lc = edge_len(a, b)
    if la <= 0.0 or lb <= 0.0 or lc <= 0.0:
        return None, None
    cos_a = clamp((lb * lb + lc * lc - la * la) / (2.0 * lb * lc), -1.0, 1.0)
    cos_b = clamp((la * la + lc * lc - lb * lb) / (2.0 * la * lc), -1.0, 1.0)
    cos_c = clamp((la * la + lb * lb - lc * lc) / (2.0 * la * lb), -1.0, 1.0)
    ang_a = math.degrees(math.acos(cos_a))
    ang_b = math.degrees(math.acos(cos_b))
    ang_c = math.degrees(math.acos(cos_c))
    return (ang_a, ang_b, ang_c), (la, lb, lc)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/mesh_quality_report.py <mesh.obj>")
        return 2
    path = sys.argv[1]
    verts, faces = read_obj(path)
    if not verts or not faces:
        print("No vertices or faces found.")
        return 1

    angles = []
    aspects = []
    skipped = 0

    for i0, i1, i2 in faces:
        if i0 < 0 or i1 < 0 or i2 < 0:
            skipped += 1
            continue
        if i0 >= len(verts) or i1 >= len(verts) or i2 >= len(verts):
            skipped += 1
            continue
        angs, lens = triangle_angles(verts[i0], verts[i1], verts[i2])
        if angs is None:
            skipped += 1
            continue
        angles.extend(angs)
        min_len = min(lens)
        max_len = max(lens)
        if min_len > 0.0:
            aspects.append(max_len / min_len)
        else:
            skipped += 1

    if not angles:
        print("No valid triangles.")
        return 1

    min_angle = min(angles)
    max_angle = max(angles)
    avg_angle = sum(angles) / len(angles)
    min_aspect = min(aspects) if aspects else float("inf")
    max_aspect = max(aspects) if aspects else float("inf")
    avg_aspect = sum(aspects) / len(aspects) if aspects else float("inf")
    p95_aspect = float("inf")
    if aspects:
        aspects_sorted = sorted(aspects)
        idx = int(0.95 * (len(aspects_sorted) - 1))
        p95_aspect = aspects_sorted[idx]

    print("Mesh quality report")
    print(f"  Triangles: {len(faces)}")
    print(f"  Skipped:   {skipped}")
    print(f"  Min angle: {min_angle:.3f} deg")
    print(f"  Max angle: {max_angle:.3f} deg")
    print(f"  Avg angle: {avg_angle:.3f} deg")
    print(
        "  Aspect ratio (max edge / min edge): "
        f"min {min_aspect:.3f}, avg {avg_aspect:.3f}, max {max_aspect:.3f}"
    )
    print(f"  Aspect ratio p95: {p95_aspect:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
