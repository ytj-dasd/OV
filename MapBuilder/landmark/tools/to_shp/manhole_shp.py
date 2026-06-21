"""Convert manhole instance labels to minimum-enclosing-circle polygons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile

from landmark.tools.to_shp.geometry import pixel_to_xy


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_map_to_manhole_shp(
    label_map_path: Path | str,
    geo_meta_path: Path | str,
    output_dir: Path | str,
    *,
    circle_points: int = 64,
    min_radius_m: float = 0.15,
    max_radius_m: float = 1.20,
) -> Path:
    """Write one sampled circle polygon per accepted manhole instance."""
    if circle_points < 8:
        raise ValueError("circle_points must be >= 8")
    label_map_path = Path(label_map_path).expanduser()
    geo_meta_path = Path(geo_meta_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = np.load(label_map_path, mmap_mode="r")
    meta = _load_json(geo_meta_path)
    mpp = float(meta["meters_per_pixel"])

    base = output_dir / "manhole"
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("id", "N", decimal=0)
    writer.field("center_x", "F", size=20, decimal=6)
    writer.field("center_y", "F", size=20, decimal=6)
    writer.field("radius_m", "F", size=16, decimal=6)
    writer.field("area_px", "N", decimal=0)
    writer.field("fill_rate", "F", size=12, decimal=6)

    accepted = 0
    rejected_radius = 0
    for label_id in [int(value) for value in np.unique(label_map) if int(value) >= 0]:
        rows, cols = np.where(label_map == label_id)
        if rows.size < 3:
            continue
        points_px = np.column_stack([cols, rows]).astype(np.float32)
        (center_col, center_row), radius_px = cv2.minEnclosingCircle(points_px)
        radius_m = float(radius_px) * mpp
        if radius_m < float(min_radius_m) or radius_m > float(max_radius_m):
            rejected_radius += 1
            continue
        center_xy = pixel_to_xy(np.asarray([[center_col, center_row]], dtype=np.float32), meta)[0]
        angles = np.linspace(0.0, -2.0 * math.pi, num=int(circle_points), endpoint=False)
        ring = [
            [
                float(center_xy[0] + radius_m * math.cos(angle)),
                float(center_xy[1] + radius_m * math.sin(angle)),
            ]
            for angle in angles
        ]
        ring.append(ring[0])
        area_px = int(rows.size)
        circle_area_px = math.pi * float(radius_px) ** 2
        fill_rate = float(area_px / circle_area_px) if circle_area_px > 0 else 0.0
        writer.poly([ring])
        writer.record(
            id=label_id,
            center_x=float(center_xy[0]),
            center_y=float(center_xy[1]),
            radius_m=radius_m,
            area_px=area_px,
            fill_rate=fill_rate,
        )
        accepted += 1
    writer.close()
    summary = {
        "label_map": str(label_map_path),
        "geo_meta": str(geo_meta_path),
        "shp": str(base.with_suffix(".shp")),
        "feature_count": accepted,
        "rejected_radius": rejected_radius,
        "circle_points": int(circle_points),
        "min_radius_m": float(min_radius_m),
        "max_radius_m": float(max_radius_m),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return base.with_suffix(".shp")

