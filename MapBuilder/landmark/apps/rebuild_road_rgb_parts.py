"""Rebuild the filled CSF RGB BEV and existing road RGB parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from landmark.tools.pc_process.pre_part import fill_image_holes_file
from landmark.tools.sam3.bev_part import write_bev_parts


def _load_existing_parts_parameters(parts_json_path: Path) -> dict[str, float]:
    with parts_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    required = ("tile_size_m", "fill_ratio_threshold", "tile_overlap_ratio")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Existing road parts.json is missing parameters: {missing}")
    return {
        "tile_size_m": float(payload["tile_size_m"]),
        "fill_ratio_threshold": float(payload["fill_ratio_threshold"]),
        "tile_overlap_ratio": float(payload["tile_overlap_ratio"]),
    }


def rebuild_road_rgb_parts(
    output_root: Path | str,
    *,
    radius_px: int = 3,
) -> dict[str, Any]:
    """Refill the existing raw RGB BEV and overwrite road RGB parts."""
    output_root = Path(output_root).expanduser()
    bev_dir = output_root / "pre-part" / "bev_pc_csf"
    road_dir = output_root / "objs" / "road"
    parts_dir = road_dir / "parts"
    rgb_path = bev_dir / "bev_pc_csf_rgb.png"
    filled_path = bev_dir / "bev_pc_csf_rgb_filled.png"
    geo_meta_path = bev_dir / "pc_csf_geo_meta.json"
    parts_json_path = parts_dir / "parts.json"

    for path in (rgb_path, geo_meta_path, parts_json_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing MapBuilder output is missing: {path}")

    parameters = _load_existing_parts_parameters(parts_json_path)
    fill_image_holes_file(rgb_path, filled_path, radius_px=radius_px)

    parts_bev_dir = parts_dir / "bev"
    if parts_bev_dir.is_dir():
        stale_parts = sorted(parts_bev_dir.glob("part_*.png"))
        for stale_part in tqdm(stale_parts, desc="Remove old road RGB parts", unit="part"):
            stale_part.unlink()
        print(f"[rebuild_road_rgb_parts] removed {len(stale_parts)} old road RGB parts", flush=True)

    payload = write_bev_parts(
        filled_path,
        geo_meta_path,
        road_dir,
        **parameters,
    )
    result = {
        "rgb_filled": filled_path,
        "parts_json": parts_json_path,
        "num_parts": int(payload["num_parts"]),
    }
    print(f"[rebuild_road_rgb_parts] rebuilt {result['num_parts']} road RGB parts", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refill an existing raw CSF RGB BEV and overwrite objs/road/parts.",
    )
    parser.add_argument("output_root", help="Existing landmark-full output root.")
    parser.add_argument("--radius-px", type=int, default=3, help="RGB hole-fill radius in pixels.")
    args = parser.parse_args()
    outputs = rebuild_road_rgb_parts(args.output_root, radius_px=args.radius_px)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
