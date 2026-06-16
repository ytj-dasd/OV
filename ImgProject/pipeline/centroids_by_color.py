"""Read a LAS file, group points by unique RGB color, compute XY centroid per color, save centroids as LAS."""

import sys
from pathlib import Path

import laspy
import numpy as np


def main() -> None:
    if len(sys.argv) > 1:
        las_path = Path(sys.argv[1])
    else:
        las_path = Path("/home/guitu/文档/OV/benchmark/road2-1/test.las")

    if not las_path.exists():
        sys.exit(f"File not found: {las_path}")

    out_path = las_path.parent / (las_path.stem + "_centroids.las")

    las = laspy.read(str(las_path))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    # Build per-point RGB as uint32 for unique grouping
    r = np.asarray(las.red, dtype=np.uint16).reshape(-1)
    g = np.asarray(las.green, dtype=np.uint16).reshape(-1)
    b = np.asarray(las.blue, dtype=np.uint16).reshape(-1)
    color_key = (r.astype(np.uint32) << 32) | (g.astype(np.uint32) << 16) | b.astype(np.uint32)

    unique_colors, inverse, counts = np.unique(color_key, return_inverse=True, return_counts=True)

    centroids_xyz = np.zeros((unique_colors.size, 3), dtype=np.float64)
    centroids_z = np.zeros(unique_colors.size, dtype=np.float64)
    for i in range(unique_colors.size):
        mask = inverse == i
        centroids_xyz[i] = xyz[mask].mean(axis=0)

    # Decode colors back
    centroid_r = ((unique_colors >> 32) & 0xFFFF).astype(np.uint16)
    centroid_g = ((unique_colors >> 16) & 0xFFFF).astype(np.uint16)
    centroid_b = (unique_colors & 0xFFFF).astype(np.uint16)

    # Write output LAS
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(las.header.scales, copy=True)
    header.offsets = np.array(las.header.offsets, copy=True)
    las_out = laspy.LasData(header)
    las_out.x = centroids_xyz[:, 0]
    las_out.y = centroids_xyz[:, 1]
    las_out.z = centroids_xyz[:, 2]
    las_out.red = centroid_r
    las_out.green = centroid_g
    las_out.blue = centroid_b

    las_out.write(str(out_path))

    print(f"Input:  {las_path}")
    print(f"Points: {xyz.shape[0]}, unique colors: {unique_colors.size}")
    for i in range(unique_colors.size):
        r8 = int(centroid_r[i] // 256)
        g8 = int(centroid_g[i] // 256)
        b8 = int(centroid_b[i] // 256)
        print(f"  color ({r8},{g8},{b8}): {counts[i]} pts, centroid=({centroids_xyz[i,0]:.3f}, {centroids_xyz[i,1]:.3f}, {centroids_xyz[i,2]:.3f})")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
