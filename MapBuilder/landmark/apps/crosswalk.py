"""Crosswalk vectorization app."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from landmark.tools.to_shp.crosswalk_shp import crosswalk_masks_to_shp


def run_crosswalk(
    label_map_path: Path | str,
    geo_meta_path: Path | str,
    laneline_shp_path: Path | str,
    output_dir: Path | str,
    *,
    short_length_threshold_m: float = 10.0,
    linearity_threshold: float = 3.0,
) -> Path:
    label_map = np.load(Path(label_map_path).expanduser())
    return crosswalk_masks_to_shp(
        label_map,
        geo_meta_path,
        laneline_shp_path,
        output_dir,
        short_length_threshold_m=short_length_threshold_m,
        linearity_threshold=linearity_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize crosswalk label_map and zebra-line lanelines.")
    parser.add_argument("label_map_npy", help="Crosswalk instance label_map .npy with shape (H,W).")
    parser.add_argument("geo_meta_json", help="geo_meta.json or summary.json path.")
    parser.add_argument("laneline_shp", help="laneline.shp path.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/crosswalk", help="Output directory.")
    parser.add_argument("--short-threshold", type=float, default=10.0)
    parser.add_argument("--linearity-threshold", type=float, default=3.0)
    args = parser.parse_args()
    run_crosswalk(
        args.label_map_npy,
        args.geo_meta_json,
        args.laneline_shp,
        args.output_dir,
        short_length_threshold_m=args.short_threshold,
        linearity_threshold=args.linearity_threshold,
    )


if __name__ == "__main__":
    main()
