"""Arrow vectorization app."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from landmark.tools.to_shp.arrow_shp import DEFAULT_MAX_MATCH_SCORE, DEFAULT_MAX_OVERFLOW, arrow_masks_to_shp


def run_road_arrow(
    label_map_path: Path | str,
    geo_meta_path: Path | str,
    output_dir: Path | str,
    *,
    ply_path: Path | str | None = None,
    template_dir: Path | str | None = None,
    vertices_path: Path | str | None = None,
    angle_steps: int = 72,
    min_mask_area: int = 2000,
    max_match_score: float = DEFAULT_MAX_MATCH_SCORE,
    max_overflow: float = DEFAULT_MAX_OVERFLOW,
) -> Path:
    label_map = np.load(Path(label_map_path).expanduser())
    return arrow_masks_to_shp(
        label_map,
        geo_meta_path,
        ply_path,
        output_dir,
        template_dir=template_dir,
        vertices_path=vertices_path,
        angle_steps=angle_steps,
        min_mask_area=min_mask_area,
        max_match_score=max_match_score,
        max_overflow=max_overflow,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize arrow label_map to road_arrow SHP output.")
    parser.add_argument("label_map_npy", help="Arrow instance label_map .npy with shape (H,W).")
    parser.add_argument("geo_meta_json", help="geo_meta.json or summary.json path.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/road_arrow", help="Output directory.")
    parser.add_argument("--ply-path", default=None, help="Optional PLY path.")
    parser.add_argument("--template-dir", default=None, help="Optional arrow template dir.")
    parser.add_argument("--vertices-path", default=None, help="Optional arrow vertices JSON path.")
    parser.add_argument("--angle-steps", type=int, default=72)
    parser.add_argument("--min-mask-area", type=int, default=2000)
    parser.add_argument("--max-match-score", type=float, default=DEFAULT_MAX_MATCH_SCORE)
    parser.add_argument("--max-overflow", type=float, default=DEFAULT_MAX_OVERFLOW)
    args = parser.parse_args()
    run_road_arrow(
        args.label_map_npy,
        args.geo_meta_json,
        args.output_dir,
        ply_path=args.ply_path,
        template_dir=args.template_dir,
        vertices_path=args.vertices_path,
        angle_steps=args.angle_steps,
        min_mask_area=args.min_mask_area,
        max_match_score=args.max_match_score,
        max_overflow=args.max_overflow,
    )


if __name__ == "__main__":
    main()
