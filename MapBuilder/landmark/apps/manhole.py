"""Manhole visual retrieval and minimum-enclosing-circle vectorization."""

from __future__ import annotations

import argparse
from pathlib import Path

from landmark.tools.sam3.sam3_about import DEFAULT_CONDA_ENV, DEFAULT_SAM3_DIR
from landmark.tools.sam3.visual_instance_seg import run_visual_instance_seg
from landmark.tools.to_shp.manhole_shp import label_map_to_manhole_shp


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "asserts" / "manhole_visual_samples" / "manifest.json"


def run_manhole(
    parts_json_path: Path | str,
    manifest_path: Path | str,
    rgb_filled_path: Path | str,
    geo_meta_path: Path | str,
    output_dir: Path | str,
    *,
    sam3_dir: Path | str = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.1,
    min_radius_m: float = 0.15,
    max_radius_m: float = 1.20,
    circle_points: int = 64,
    force: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser()
    seg_outputs = run_visual_instance_seg(
        parts_json_path,
        manifest_path,
        rgb_filled_path,
        output_dir,
        sam3_dir=sam3_dir,
        conda_env=conda_env,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        force=force,
    )
    shp_path = label_map_to_manhole_shp(
        seg_outputs["label_map"],
        geo_meta_path,
        output_dir.parent / "shp" / "manhole",
        circle_points=circle_points,
        min_radius_m=min_radius_m,
        max_radius_m=max_radius_m,
    )
    return {**seg_outputs, "shp": shp_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and vectorize manholes from existing road RGB parts.")
    parser.add_argument("parts_json")
    parser.add_argument("rgb_filled")
    parser.add_argument("geo_meta")
    parser.add_argument("--samples", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sam3-dir", default=str(DEFAULT_SAM3_DIR))
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--score-th", type=float, default=0.5)
    parser.add_argument("--iou-th", type=float, default=0.1)
    parser.add_argument("--min-radius-m", type=float, default=0.15)
    parser.add_argument("--max-radius-m", type=float, default=1.20)
    parser.add_argument("--circle-points", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = run_manhole(
        args.parts_json,
        args.samples,
        args.rgb_filled,
        args.geo_meta,
        args.out,
        sam3_dir=args.sam3_dir,
        conda_env=args.conda_env,
        score_threshold=args.score_th,
        iou_threshold=args.iou_th,
        min_radius_m=args.min_radius_m,
        max_radius_m=args.max_radius_m,
        circle_points=args.circle_points,
        force=args.force,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

