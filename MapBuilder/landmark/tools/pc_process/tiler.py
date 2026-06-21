from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def load_tiles_geo(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"No tiles found in {path}")
    return payload


def tile_point_mask(points_xyz: np.ndarray, tile: dict[str, Any]) -> np.ndarray:
    left = float(tile["geo_left"])
    right = float(tile["geo_right"])
    top = float(tile["geo_top"])
    bottom = float(tile["geo_bottom"])
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    return (x >= left) & (x < right) & (y >= bottom) & (y < top)


def split_ply_by_tiles(
    ply_path: Path | str,
    tiles_geo_path: Path | str,
    output_dir: Path | str,
    *,
    render_bev: bool = True,
    mpp: float = 0.02,
) -> list[Path]:
    ply_path = Path(ply_path).expanduser()
    tiles_geo_path = Path(tiles_geo_path).expanduser()
    output_dir = Path(output_dir).expanduser()

    tiles_payload = load_tiles_geo(tiles_geo_path)
    tiles = tiles_payload["tiles"]

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise KeyError(f"'vertex' element not found in {ply_path}")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise KeyError(f"PLY vertex fields must include {sorted(required)}")

    points_xyz = np.stack([vertex.data["x"], vertex.data["y"], vertex.data["z"]], axis=-1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    manifest_tiles: list[dict[str, Any]] = []
    for idx, tile in enumerate(tiles, start=1):
        keep = tile_point_mask(points_xyz, tile)
        if not np.any(keep):
            continue

        tile_vertex = vertex.data[keep]
        tile_name = (
            f"tile_{idx:03d}"
            f"_x{int(tile['pixel_min_x'])}_y{int(tile['pixel_min_y'])}.ply"
        )
        tile_path = ply_dir / tile_name
        tile_ply = PlyData(
            [PlyElement.describe(tile_vertex, "vertex")],
            text=ply.text,
            byte_order=ply.byte_order,
        )
        tile_ply.write(str(tile_path))
        written.append(tile_path)

        bev_meta_record: dict[str, Any] | None = None
        if render_bev:
            bev_dir = output_dir / "bev"
            bev_dir.mkdir(parents=True, exist_ok=True)
            stem = tile_path.stem
            out_map = {
                m: bev_dir / f"{stem}_{m}.png" for m in ("rgb", "intensity")
            }
            try:
                from landmark.tools.pc_process.bev import render_bev as _render_bev

                bev_results = _render_bev(
                    tile_path, mode=["rgb", "intensity"], mpp=mpp,
                )
                assert isinstance(bev_results, dict)
                for m, (img, _meta) in bev_results.items():
                    out_p = Path(out_map[m])
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(str(out_p))
                _sample_meta = next(iter(bev_results.values()))[1]
                bev_meta_record = {
                    "min_xy": _sample_meta["min_xy"],
                    "max_xy": _sample_meta["max_xy"],
                    "meters_per_pixel": _sample_meta["meters_per_pixel"],
                    "width": _sample_meta["width"],
                    "height": _sample_meta["height"],
                }
            except (KeyError, ValueError) as exc:
                print(f"[tiler] skip BEV for {tile_name}: {exc}")

        tile_record = dict(tile)
        tile_record["tile_index"] = idx
        tile_record["output"] = str(tile_path)
        tile_record["num_points"] = int(np.count_nonzero(keep))
        if bev_meta_record is not None:
            tile_record["bev_meta"] = bev_meta_record
        manifest_tiles.append(tile_record)

    geo_meta_tiles: list[dict[str, Any]] = []
    for rec in manifest_tiles:
        if "bev_meta" not in rec:
            continue
        bm = rec["bev_meta"]
        geo_meta_tiles.append({
            "tile_name": Path(rec["output"]).stem,
            "pixel_min_x": rec.get("pixel_min_x"),
            "pixel_min_y": rec.get("pixel_min_y"),
            "bev_origin_xy": bm["min_xy"],
            "bev_size": [bm["width"], bm["height"]],
            "meters_per_pixel": bm["meters_per_pixel"],
        })
    if geo_meta_tiles:
        geo_meta = {
            "description": (
                "BEV pixel to world coordinate conversion. "
                "pixel_min_x / pixel_min_y are this tile's offset in the global BEV grid. "
                "bev_origin_xy is the world XY of the BEV image bottom-left corner. "
                "world_x = bev_origin_xy[0] + col * meters_per_pixel; "
                "world_y = bev_origin_xy[1] + (bev_size[1] - 1 - row) * meters_per_pixel."
            ),
            "tiles": geo_meta_tiles,
        }
        geo_meta_path = output_dir / "geo_meta.json"
        with geo_meta_path.open("w", encoding="utf-8") as f:
            json.dump(geo_meta, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_ply": str(ply_path),
        "tiles_geo": str(tiles_geo_path),
        "num_tiles_written": len(written),
        "tiles": manifest_tiles,
    }
    manifest_path = output_dir / "tiles_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a PLY file using tiles_geo.json bounds.")
    parser.add_argument(
        "ply_path",
        nargs="?",
        default="input/RS_with_labels.ply",
        help="Input PLY path.",
    )
    parser.add_argument(
        "tiles_geo_path",
        nargs="?",
        default="input/tiles_geo.json",
        help="tiles_geo.json path.",
    )
    parser.add_argument(
        "--out",
        default="input/tiles",
        help="Output directory for tiled PLY files.",
    )
    parser.add_argument(
        "--no-bev",
        action="store_true",
        help="Skip BEV image rendering.",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=0.02,
        help="Meters per pixel for BEV rendering (default: 0.02).",
    )
    args = parser.parse_args()

    written = split_ply_by_tiles(
        args.ply_path,
        args.tiles_geo_path,
        args.out,
        render_bev=not args.no_bev,
        mpp=args.mpp,
    )
    print(f"Wrote {len(written)} tile PLY file(s) to {Path(args.out).expanduser()}")


if __name__ == "__main__":
    main()
