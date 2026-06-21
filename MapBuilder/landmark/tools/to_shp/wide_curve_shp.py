"""Wide-curve label-map vectorization tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from landmark.tools.to_shp.long_laneline_shp import long_laneline_to_shp


def _coerce_bev_meta(meta_or_path: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(meta_or_path, dict):
        meta = dict(meta_or_path)
    else:
        path = Path(meta_or_path).expanduser()
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    if "min_xy" in meta and "max_xy" in meta:
        return meta
    if "canvas_size" in meta and "global_origin_xy" in meta:
        canvas_w, canvas_h = meta["canvas_size"]
        mpp = float(meta["meters_per_pixel"])
        g_min_x, g_max_y = meta["global_origin_xy"]
        min_y = g_max_y - (canvas_h - 1) * mpp
        return {
            "width": int(canvas_w),
            "height": int(canvas_h),
            "meters_per_pixel": mpp,
            "min_xy": [float(g_min_x), float(min_y)],
            "max_xy": [float(g_min_x + (canvas_w - 1) * mpp), float(g_max_y)],
        }
    raise KeyError("Expected bev_meta with min_xy/max_xy or summary with canvas_size/global_origin_xy")


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 3:
        return arr.astype(bool, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.int32, copy=False)
    raise ValueError(f"masks must have shape (K,H,W) or (H,W), got {arr.shape}")


def wide_curve_masks_to_shp(
    label_map: np.ndarray,
    geo_meta: dict[str, Any] | Path | str,
    ply_path: Path | str | None,
    output_dir: Path | str,
    *,
    sample_interval_m: float = 0.10,
    cross_half_width_m: float = 0.30,
    min_length_m: float = 10.0,
    centerline_smooth_m: float = 0.10,
) -> Path:
    del ply_path  # kept for a uniform tool signature

    bev_meta = _coerce_bev_meta(geo_meta)
    masks_bool = _normalize_masks(label_map)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"

    return long_laneline_to_shp(
        masks_bool,
        bev_meta,
        output_dir,
        sample_interval_m=sample_interval_m,
        cross_half_width_m=cross_half_width_m,
        min_length_m=min_length_m,
        centerline_smooth_m=centerline_smooth_m,
        debug_dir=debug_dir,
        shp_stem="wide_curve",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize long-curve instance label_map → wide_curve.shp.")
    parser.add_argument("label_map_npy", help="Path to label_map .npy file with shape (H,W).")
    parser.add_argument("geo_meta_json", help="Path to geo_meta.json or summary.json.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/shp/wide_curve", help="Output directory.")
    parser.add_argument("--ply-path", default=None, help="Optional PLY path. Accepted for interface consistency.")
    parser.add_argument("--interval", type=float, default=0.10, help="Control point interval in metres.")
    parser.add_argument("--half-width", type=float, default=0.30, help="Normal cross-section half-width in metres.")
    parser.add_argument("--min-length", type=float, default=10.0, help="Minimum curve length in metres.")
    parser.add_argument("--smooth", type=float, default=0.10, help="Centerline smoothing sigma in metres.")
    args = parser.parse_args()

    masks = np.load(args.label_map_npy)
    wide_curve_masks_to_shp(
        masks,
        args.geo_meta_json,
        args.ply_path,
        args.output_dir,
        sample_interval_m=args.interval,
        cross_half_width_m=args.half_width,
        min_length_m=args.min_length,
        centerline_smooth_m=args.smooth,
    )


if __name__ == "__main__":
    main()
