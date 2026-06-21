from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

Image.MAX_IMAGE_PIXELS = None
_FIXED_PART_BEV_SIZE = 1008


def load_parts(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"No parts found in {path}")
    return payload


def _coerce_vertices_xy(part: dict[str, Any]) -> np.ndarray:
    candidates = (
        part.get("vertices_xy"),
        part.get("vertices"),
        part.get("corners_xy"),
    )
    for raw in candidates:
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        verts: list[list[float]] = []
        ok = True
        for item in raw:
            if isinstance(item, dict):
                if "xy" in item and isinstance(item["xy"], list) and len(item["xy"]) >= 2:
                    verts.append([float(item["xy"][0]), float(item["xy"][1])])
                elif {"x", "y"}.issubset(item):
                    verts.append([float(item["x"]), float(item["y"])])
                else:
                    ok = False
                    break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                verts.append([float(item[0]), float(item[1])])
            else:
                ok = False
                break
        if ok:
            arr = np.asarray(verts, dtype=np.float64)
            if arr.shape == (4, 2):
                return arr
    raise KeyError("Each part must provide 4 vertices via vertices_xy / vertices / corners_xy")


def _points_in_convex_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64)
    if polygon.shape != (4, 2):
        raise ValueError(f"polygon must have shape (4,2), got {polygon.shape}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N,2), got {points.shape}")

    signs: list[np.ndarray] = []
    for i in range(4):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % 4]
        edge = p1 - p0
        rel = points - p0
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        signs.append(cross)
    stacked = np.stack(signs, axis=1)
    eps = 1e-9
    return np.all(stacked >= -eps, axis=1) | np.all(stacked <= eps, axis=1)


def part_point_mask(points_xyz: np.ndarray, part: dict[str, Any]) -> np.ndarray:
    vertices_xy = _coerce_vertices_xy(part)
    return _points_in_convex_polygon(np.asarray(points_xyz)[:, :2], vertices_xy)


def _load_geo_meta(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _world_bounds_to_pixel_window(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    geo_meta: dict[str, Any],
) -> tuple[int, int, int, int]:
    min_x, min_y = [float(v) for v in geo_meta["min_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    width = int(geo_meta["width"])
    height = int(geo_meta["height"])
    world_top_exclusive = float(min_y + height * mpp)

    pixel_min_x = int(math.floor((left - min_x) / mpp))
    pixel_max_x = int(math.ceil((right - min_x) / mpp))
    pixel_min_y = int(math.floor((world_top_exclusive - top) / mpp))
    pixel_max_y = int(math.ceil((world_top_exclusive - bottom) / mpp))

    pixel_min_x = max(0, min(width, pixel_min_x))
    pixel_max_x = max(0, min(width, pixel_max_x))
    pixel_min_y = max(0, min(height, pixel_min_y))
    pixel_max_y = max(0, min(height, pixel_max_y))
    return pixel_min_x, pixel_max_x, pixel_min_y, pixel_max_y


def _pixel_to_world_xy(col: np.ndarray, row: np.ndarray, geo_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y = [float(v) for v in geo_meta["min_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    height = int(geo_meta["height"])
    x = min_x + (col.astype(np.float64) + 0.5) * mpp
    y = min_y + (height - row.astype(np.float64) - 0.5) * mpp
    return x, y


def _world_xy_to_pixel_float(x: np.ndarray, y: np.ndarray, geo_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    min_x, _min_y = [float(v) for v in geo_meta["min_xy"]]
    _max_x, max_y = [float(v) for v in geo_meta["max_xy"]]
    mpp = float(geo_meta["meters_per_pixel"])
    col = (x.astype(np.float64) - min_x) / mpp
    row = (max_y - y.astype(np.float64)) / mpp
    return col, row


def _warp_bev_to_part(
    image: np.ndarray,
    polygon_xy: np.ndarray,
    geo_meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
    if polygon_xy.shape != (4, 2):
        raise ValueError(f"polygon_xy must have shape (4,2), got {polygon_xy.shape}")

    src_cols, src_rows = _world_xy_to_pixel_float(
        polygon_xy[:, 0],
        polygon_xy[:, 1],
        geo_meta,
    )
    src = np.column_stack([src_cols, src_rows]).astype(np.float32)
    dst = np.asarray(
        [
            [0.0, _FIXED_PART_BEV_SIZE - 1.0],
            [_FIXED_PART_BEV_SIZE - 1.0, _FIXED_PART_BEV_SIZE - 1.0],
            [_FIXED_PART_BEV_SIZE - 1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        np.asarray(image),
        matrix,
        (_FIXED_PART_BEV_SIZE, _FIXED_PART_BEV_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    edge_lengths = [
        float(np.linalg.norm(polygon_xy[(i + 1) % 4] - polygon_xy[i]))
        for i in range(4)
    ]
    horizontal_length = max(edge_lengths[0], edge_lengths[2])
    vertical_length = max(edge_lengths[1], edge_lengths[3])
    meters_per_pixel_x = horizontal_length / max(_FIXED_PART_BEV_SIZE - 1, 1)
    meters_per_pixel_y = vertical_length / max(_FIXED_PART_BEV_SIZE - 1, 1)
    warp_meta = {
        "min_xy": polygon_xy[0].tolist(),
        "max_xy": polygon_xy[2].tolist(),
        "meters_per_pixel": float(max(meters_per_pixel_x, meters_per_pixel_y)),
        "width": _FIXED_PART_BEV_SIZE,
        "height": _FIXED_PART_BEV_SIZE,
        "corners_xy": polygon_xy.tolist(),
        "pixel_corners_xy": dst.tolist(),
    }
    return warped, warp_meta


def _resolve_pre_part_assets(parts_json_path: Path) -> dict[str, Path] | None:
    pre_part_dir = parts_json_path.parent
    pc_csf_bev_dir = pre_part_dir / "bev_pc_csf"
    asset_dir = pc_csf_bev_dir if pc_csf_bev_dir.is_dir() else pre_part_dir
    rgb_png_path = asset_dir / "bev_pc_csf_rgb_filled.png"
    if not rgb_png_path.is_file():
        rgb_png_path = asset_dir / "bev_pc_csf_rgb.png"
    candidates = {
        "pc_csf_ply": pre_part_dir / "pc_csf.ply",
        "pc_csf_rgb_png": rgb_png_path,
        "pc_csf_intensity_png": asset_dir / "bev_pc_csf_intensity.png",
        "pc_csf_geo_meta": asset_dir / "pc_csf_geo_meta.json",
    }
    if all(path.is_file() for path in candidates.values()):
        return candidates
    return None


def split_ply_by_parts(
    ply_path: Path | str,
    parts_json_path: Path | str,
    output_dir: Path | str,
    *,
    render_bev: bool = True,
    mpp: float = 0.02,
    split_point_cloud: bool = False,
) -> list[Path]:
    ply_path = Path(ply_path).expanduser()
    parts_json_path = Path(parts_json_path).expanduser()
    output_dir = Path(output_dir).expanduser()

    parts_payload = load_parts(parts_json_path)
    parts = parts_payload["parts"]

    pre_part_assets = _resolve_pre_part_assets(parts_json_path)
    if not split_point_cloud and pre_part_assets is None:
        raise FileNotFoundError(
            "Pure BEV slicing mode requires pre-part assets next to parts.json: "
            "pc_csf.ply, bev_pc_csf_rgb.png, bev_pc_csf_intensity.png, pc_csf_geo_meta.json"
        )

    source_ply_path = pre_part_assets["pc_csf_ply"] if pre_part_assets is not None else ply_path
    ply: PlyData | None = None
    vertex = None
    points_xyz: np.ndarray | None = None
    if split_point_cloud or pre_part_assets is None:
        ply = PlyData.read(str(source_ply_path))
        if "vertex" not in ply:
            raise KeyError(f"'vertex' element not found in {source_ply_path}")
        vertex = ply["vertex"]
        names = vertex.data.dtype.names or ()
        required = {"x", "y", "z"}
        if not required.issubset(names):
            raise KeyError(f"PLY vertex fields must include {sorted(required)}")
        points_xyz = np.stack([vertex.data["x"], vertex.data["y"], vertex.data["z"]], axis=-1)

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = output_dir / "ply"
    if split_point_cloud:
        ply_dir.mkdir(parents=True, exist_ok=True)
    bev_dir = output_dir / "bev"
    if render_bev:
        bev_dir.mkdir(parents=True, exist_ok=True)
    tmp_ply_dir = output_dir / "_tmp_ply"
    if render_bev and pre_part_assets is None:
        tmp_ply_dir.mkdir(parents=True, exist_ok=True)
    full_bev_images: dict[str, np.ndarray] = {}
    full_bev_meta: dict[str, Any] | None = None
    if render_bev and pre_part_assets is not None:
        full_bev_images = {
            "rgb": np.asarray(Image.open(pre_part_assets["pc_csf_rgb_png"]).convert("RGB")),
            "intensity": np.asarray(Image.open(pre_part_assets["pc_csf_intensity_png"]).convert("RGB")),
        }
        full_bev_meta = _load_geo_meta(pre_part_assets["pc_csf_geo_meta"])

    written: list[Path] = []
    manifest_parts: list[dict[str, Any]] = []
    for idx, part in enumerate(parts, start=1):
        part_id = part.get("part_id", idx)
        part_path = ply_dir / f"part_{int(part_id):03d}.ply"
        temp_part_path = tmp_ply_dir / f"part_{int(part_id):03d}.ply"
        render_part_ply_path = part_path if split_point_cloud else temp_part_path
        num_points: int | None = None

        if split_point_cloud or (render_bev and pre_part_assets is None):
            assert points_xyz is not None
            assert vertex is not None
            assert ply is not None
            keep = part_point_mask(points_xyz, part)
            if not np.any(keep):
                continue
            num_points = int(np.count_nonzero(keep))
            part_vertex = vertex.data[keep]
            part_ply = PlyData(
                [PlyElement.describe(part_vertex, "vertex")],
                text=ply.text,
                byte_order=ply.byte_order,
            )
            part_ply.write(str(render_part_ply_path))
            if split_point_cloud:
                written.append(part_path)

        part_record = dict(part)
        part_record["part_index"] = idx
        part_record["part_id"] = part_id
        part_record["output"] = str(part_path) if split_point_cloud else None
        part_record["num_points"] = num_points

        bev_meta_record: dict[str, Any] | None = None
        if render_bev:
            out_map = {
                mode: bev_dir / f"{part_path.stem}_{mode}.png"
                for mode in ("rgb", "intensity")
            }
            if pre_part_assets is not None and full_bev_meta is not None:
                polygon_xy = _coerce_vertices_xy(part)
                warped_results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
                for mode_name, full_image in full_bev_images.items():
                    img, crop_meta = _warp_bev_to_part(full_image, polygon_xy, full_bev_meta)
                    Image.fromarray(img).save(str(out_map[mode_name]))
                    warped_results[mode_name] = (img, crop_meta)
                sample_meta = next(iter(warped_results.values()))[1] if warped_results else None
            else:
                from landmark.tools.pc_process.bev import render_bev as _render_bev

                bev_results = _render_bev(
                    render_part_ply_path,
                    mode=["rgb", "intensity"],
                    mpp=mpp,
                    skip_missing_fields=True,
                )
                assert isinstance(bev_results, dict)
                for mode_name, (img, _meta) in bev_results.items():
                    Image.fromarray(img).save(str(out_map[mode_name]))
                sample_meta = next(iter(bev_results.values()))[1] if bev_results else None
            if sample_meta is not None:
                bev_meta_record = {
                    "min_xy": sample_meta["min_xy"],
                    "max_xy": sample_meta["max_xy"],
                    "meters_per_pixel": sample_meta["meters_per_pixel"],
                    "width": sample_meta["width"],
                    "height": sample_meta["height"],
                    "corners_xy": sample_meta.get("corners_xy"),
                    "pixel_corners_xy": sample_meta.get("pixel_corners_xy"),
                }
                part_record["bev_meta"] = bev_meta_record
        manifest_parts.append(part_record)

    geo_meta_parts: list[dict[str, Any]] = []
    for rec in manifest_parts:
        if "bev_meta" not in rec:
            continue
        bm = rec["bev_meta"]
        geo_meta_parts.append({
            "tile_name": f"part_{int(rec['part_id']):03d}",
            "part_id": rec.get("part_id"),
            "bev_origin_xy": bm["min_xy"],
            "bev_size": [bm["width"], bm["height"]],
            "meters_per_pixel": bm["meters_per_pixel"],
            "corners_xy": bm.get("corners_xy"),
            "pixel_corners_xy": bm.get("pixel_corners_xy"),
        })
    if geo_meta_parts:
        geo_meta = {
            "description": (
                "Part BEV pixel to world coordinate conversion. "
                "For perspective-warped tiles, corners_xy and pixel_corners_xy define the "
                "world-to-image corner correspondence and should be preferred over the "
                "axis-aligned bev_origin_xy/meters_per_pixel approximation. "
                "bev_origin_xy is the world XY of the BEV image bottom-left corner. "
                "world_x = bev_origin_xy[0] + col * meters_per_pixel; "
                "world_y = bev_origin_xy[1] + (bev_size[1] - 1 - row) * meters_per_pixel."
            ),
            "tiles": geo_meta_parts,
        }
        geo_meta_path = output_dir / "geo_meta.json"
        with geo_meta_path.open("w", encoding="utf-8") as f:
            json.dump(geo_meta, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_ply": str(source_ply_path),
        "parts_json": str(parts_json_path),
        "num_parts_written": len(written),
        "parts": manifest_parts,
    }
    manifest_path = output_dir / "parts_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a PLY file using parts.json polygons.")
    parser.add_argument("ply_path", help="Input PLY path.")
    parser.add_argument("parts_json_path", help="parts.json path.")
    parser.add_argument("--out", default="outputs/parts", help="Output directory.")
    parser.add_argument("--no-bev", action="store_true", help="Skip rgb/intensity BEV rendering.")
    parser.add_argument("--mpp", type=float, default=0.02, help="Meters per pixel for part BEV rendering.")
    parser.add_argument(
        "--split-point-cloud",
        action="store_true",
        help="Debug option: also split and save per-part point clouds.",
    )
    args = parser.parse_args()

    written = split_ply_by_parts(
        args.ply_path,
        args.parts_json_path,
        args.out,
        render_bev=not args.no_bev,
        mpp=args.mpp,
        split_point_cloud=args.split_point_cloud,
    )
    print(f"Wrote {len(written)} part PLY file(s) to {Path(args.out).expanduser()}")


if __name__ == "__main__":
    main()
