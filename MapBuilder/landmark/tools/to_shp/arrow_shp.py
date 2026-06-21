"""Arrow label-map vectorization tool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_ARROW_VERTICES_PATH = Path("asserts/arrow_line/arrow_vertices.json")
DEFAULT_MAX_MATCH_SCORE = 8.0
DEFAULT_MAX_OVERFLOW = 0.70


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


def _masks_to_label_map(masks: np.ndarray) -> np.ndarray:
    label_map = np.full(masks.shape[1:], -1, dtype=np.int32)
    for idx, mask in enumerate(masks):
        label_map[mask.astype(bool)] = idx
    return label_map


def _render_masks_overview(masks: np.ndarray, output_path: Path) -> Path:
    arr = np.asarray(masks)
    if arr.ndim == 3:
        label_map = _masks_to_label_map(_normalize_masks(arr))
    elif arr.ndim == 2:
        label_map = arr.astype(np.int32, copy=False)
    else:
        raise ValueError(f"masks must have shape (K,H,W) or (H,W), got {arr.shape}")
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    ids = ids[ids >= 0]
    rng = np.random.default_rng(42)
    for oid in ids:
        rgb[label_map == oid] = rng.integers(60, 256, size=3, dtype=np.uint8)
    Image.fromarray(rgb).save(output_path)
    return output_path


def _iter_object_crops(
    final_masks: np.ndarray,
) -> list[tuple[int, np.ndarray, int, int, int, int]]:
    arr = np.asarray(final_masks)
    if arr.ndim == 3:
        arr = _masks_to_label_map(_normalize_masks(arr))
    if arr.ndim != 2:
        raise ValueError(f"final_masks must have shape (H,W) or (K,H,W), got {arr.shape}")

    ids = np.unique(arr)
    ids = ids[ids >= 0]
    results: list[tuple[int, np.ndarray, int, int, int, int]] = []
    for oid in ids:
        rows, cols = np.where(arr == oid)
        if rows.size == 0:
            continue
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = (arr[r0:r1, c0:c1] == oid)
        results.append((int(oid), crop, r0, r1, c0, c1))
    return results


def extract_arrow_features(
    final_masks_path: Path,
    bev_meta: dict[str, Any],
    output_dir: Path,
    *,
    template_dir: Path | str | None = None,
    angle_steps: int = 72,
    min_mask_area: int = 2000,
    max_match_score: float = DEFAULT_MAX_MATCH_SCORE,
    max_overflow: float = DEFAULT_MAX_OVERFLOW,
    bev_img_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Chamfer-match each arrow mask against templates."""
    from landmark.tools.to_shp.arrow_ops import chamfer_match

    final_masks = np.load(final_masks_path)
    if final_masks.ndim == 3:
        final_masks = _masks_to_label_map(_normalize_masks(final_masks))
    if final_masks.ndim != 2:
        raise ValueError(f"final_masks must have shape (H,W) or (K,H,W), got {final_masks.shape}")
    crops = _iter_object_crops(final_masks)
    canvas_h, canvas_w = final_masks.shape
    del final_masks

    if not crops:
        print(f"[vector] arrow: no objects in {final_masks_path.name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "arrow_results.json").open("w", encoding="utf-8") as f:
            json.dump({"bev_meta": bev_meta, "results": []}, f, ensure_ascii=False, indent=2)
        return []

    using_default_templates = template_dir is None
    tmpl_root = Path(template_dir) if template_dir else Path("asserts/arrow_templates")
    tmpl_root = tmpl_root.expanduser()
    edge_root = tmpl_root.parent / "arrow_edges"
    if using_default_templates and not any(tmpl_root.glob("*.png")):
        from landmark.tools.to_shp.arrow_ops import generate_default_arrow_templates, generate_edge_templates

        generate_default_arrow_templates(tmpl_root)
        generate_edge_templates(tmpl_root, edge_root)
    templates: dict[str, np.ndarray] = {}
    for p in sorted(tmpl_root.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[p.stem] = (img > 0).astype(np.uint8)
    if not templates:
        raise FileNotFoundError(f"No template PNGs found in {tmpl_root}")
    print(f"[vector] arrow: loaded {len(templates)} templates: {list(templates.keys())}")

    max_tmpl_dim = max(
        max(t.shape[0], t.shape[1]) for t in templates.values()
    )
    pad = max(100, max_tmpl_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    sam3_dir = output_dir / "sam3_res"
    sam3_dir.mkdir(parents=True, exist_ok=True)

    bev_h = bev_meta["height"]
    mpp = bev_meta["meters_per_pixel"]
    min_x, min_y = bev_meta["min_xy"]

    bev_img: np.ndarray | None = None
    if bev_img_path is not None and bev_img_path.is_file():
        bev_img = np.array(Image.open(bev_img_path).convert("RGB"))

    overlay = bev_img.copy() if bev_img is not None else None
    combined_arrow_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    results: list[dict[str, Any]] = []
    for oid, crop, r0, r1, c0, c1 in crops:
        area = int(np.sum(crop))
        if area < min_mask_area:
            continue

        pr0 = max(0, r0 - pad)
        pr1 = min(canvas_h, r1 + pad)
        pc0 = max(0, c0 - pad)
        pc1 = min(canvas_w, c1 + pad)
        padded_mask = np.zeros((pr1 - pr0, pc1 - pc0), dtype=bool)
        local_r0 = r0 - pr0
        local_c0 = c0 - pc0
        padded_mask[local_r0:local_r0 + (r1 - r0), local_c0:local_c0 + (c1 - c0)] = crop

        mask_u8 = crop.astype(np.uint8) * 255
        Image.fromarray(mask_u8).save(sam3_dir / f"mask_{oid:03d}.png")

        if bev_img is not None:
            bev_crop = bev_img[r0:r1, c0:c1].copy()
            bev_crop[~crop] = 0
            Image.fromarray(bev_crop).save(sam3_dir / f"mask_{oid:03d}_crop.png")

        try:
            match = chamfer_match(
                padded_mask, templates,
                edge_dir=edge_root if edge_root.is_dir() else None,
                angle_steps=angle_steps,
            )
            fitted_local = match.pop("fitted_mask")

            match["tx"] = float(match["tx"]) + pc0
            match["ty"] = float(match["ty"]) + pr0

            match["id"] = oid
            match["area"] = area
            match["bbox"] = [c0, r0, c1 - c0, r1 - r0]

            wx = match["tx"] * mpp + min_x
            wy = (bev_h - 1 - match["ty"]) * mpp + min_y
            world_yaw = -match["theta"]
            match["world_xy"] = [float(wx), float(wy)]
            match["world_yaw"] = float(world_yaw)

            score = float(match.get("score", float("inf")))
            overflow = float(match.get("overflow", 1.0))
            if score > max_match_score or overflow > max_overflow:
                print(
                    f"  mask {oid}: rejected type={match['type']}, score={score:.2f}, "
                    f"overflow={overflow:.2f}, max_score={max_match_score:.2f}, "
                    f"max_overflow={max_overflow:.2f}",
                    flush=True,
                )
                del fitted_local
                continue

            results.append(match)
            combined_arrow_mask[pr0:pr1, pc0:pc1][fitted_local] = 255

            if bev_img is not None:
                global_fitted = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                global_fitted[pr0:pr1, pc0:pc1][fitted_local] = 255

                fit_vis = bev_img.copy()
                gf_bool = global_fitted > 0
                fit_vis[gf_bool] = np.clip(
                    fit_vis[gf_bool].astype(np.int16) + np.array([0, 80, 0], dtype=np.int16),
                    0, 255,
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    global_fitted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(fit_vis, contours, -1, (0, 255, 0), 2)
                vis_pad = 30
                vy0 = max(0, r0 - vis_pad)
                vy1 = min(canvas_h, r1 + vis_pad)
                vx0 = max(0, c0 - vis_pad)
                vx1 = min(canvas_w, c1 + vis_pad)
                Image.fromarray(fit_vis[vy0:vy1, vx0:vx1]).save(
                    sam3_dir / f"mask_{oid:03d}_fit.png"
                )
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                del global_fitted

            del fitted_local

            print(
                f"  mask {oid}: type={match['type']}, score={match['score']:.2f}, "
                f"tail={match.get('tail_frac', 1.0):.2f}, "
                f"overflow={match.get('overflow', 0):.2f}, "
                f"fill={match.get('fill_ratio', 0):.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"  mask {oid}: chamfer match failed – {e}", flush=True)
            if overlay is not None:
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                offset_contours = [c + np.array([[[c0, r0]]]) for c in contours]
                cv2.drawContours(overlay, offset_contours, -1, (128, 128, 128), 1)

    if overlay is not None:
        Image.fromarray(overlay).save(output_dir / "arrow_matched_overlay.png")
        print(f"[vector] vis saved → {output_dir / 'arrow_matched_overlay.png'}", flush=True)
    Image.fromarray(combined_arrow_mask).save(output_dir / "arrow_fitted_mask.png")

    with (output_dir / "arrow_results.json").open("w", encoding="utf-8") as f:
        json.dump({"bev_meta": bev_meta, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[vector] arrow: {len(results)} matched out of {len(crops)} masks")
    return results


def _rotate_and_translate(
    vertices_px: list[list[float]],
    template_size: list[int],
    theta: float,
    tx_global: float,
    ty_global: float,
    bev_meta: dict[str, Any],
) -> list[list[float]]:
    h_tmpl, w_tmpl = template_size
    cx_tmpl = w_tmpl / 2.0
    cy_tmpl = h_tmpl / 2.0

    mpp = float(bev_meta["meters_per_pixel"])
    bev_h = int(bev_meta["height"])
    min_x, min_y = bev_meta["min_xy"]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    world_pts: list[list[float]] = []
    for vx, vy in vertices_px:
        dx = vx - cx_tmpl
        dy = vy - cy_tmpl
        rx = cos_t * dx - sin_t * dy
        ry = sin_t * dx + cos_t * dy
        px = rx + tx_global
        py = ry + ty_global
        wx = px * mpp + min_x
        wy = (bev_h - 1 - py) * mpp + min_y
        world_pts.append([wx, wy])
    return world_pts


def arrow_results_to_shp(
    arrow_results: list[dict[str, Any]],
    bev_meta: dict[str, Any],
    output_dir: Path | str,
    *,
    vertices_path: Path | str | None = None,
    shp_stem: str = "arrow",
) -> Path:
    vp = Path(vertices_path).expanduser() if vertices_path is not None else _ARROW_VERTICES_PATH
    with vp.open("r", encoding="utf-8") as f:
        vtx_data = json.load(f)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    shp_path = output_dir / shp_stem
    w = shapefile.Writer(str(shp_path))
    w.shapeType = shapefile.POLYGON
    w.field("id", "N", decimal=0)
    w.field("type", "C", size=20)
    w.field("score", "F", decimal=4)
    w.field("yaw_deg", "F", decimal=2)
    w.field("wx", "F", decimal=3)
    w.field("wy", "F", decimal=3)

    count = 0
    for ar in arrow_results:
        atype = ar["type"]
        if atype not in vtx_data:
            print(f"[shp] arrow: no vertices template for type={atype!r}, skipping id={ar['id']}")
            continue

        tmpl_info = vtx_data[atype]
        world_pts = _rotate_and_translate(
            tmpl_info["vertices_px"],
            tmpl_info["size"],
            theta=ar["theta"],
            tx_global=ar["tx"],
            ty_global=ar["ty"],
            bev_meta=bev_meta,
        )
        ring = [list(p) for p in world_pts]
        ring.append(ring[0])
        w.poly([ring])
        w.record(
            id=ar["id"],
            type=atype,
            score=ar["score"],
            yaw_deg=math.degrees(ar.get("world_yaw", 0.0)),
            wx=ar["world_xy"][0],
            wy=ar["world_xy"][1],
        )
        count += 1

    w.close()
    print(f"[shp] wrote {count} arrow features → {shp_path}.shp", flush=True)
    return Path(f"{shp_path}.shp")


def arrow_masks_to_shp(
    label_map: np.ndarray,
    geo_meta: dict[str, Any] | Path | str,
    ply_path: Path | str | None,
    output_dir: Path | str,
    *,
    template_dir: Path | str | None = None,
    vertices_path: Path | str | None = None,
    angle_steps: int = 72,
    min_mask_area: int = 2000,
    max_match_score: float = DEFAULT_MAX_MATCH_SCORE,
    max_overflow: float = DEFAULT_MAX_OVERFLOW,
) -> Path:
    del ply_path  # kept for a uniform tool signature

    bev_meta = _coerce_bev_meta(geo_meta)
    masks_arr = _normalize_masks(label_map)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = _render_masks_overview(masks_arr, output_dir / "arrow_input_masks.png")
    masks_path = output_dir / "_arrow_input_masks.npy"
    if np.asarray(masks_arr).ndim == 3:
        np.save(masks_path, _masks_to_label_map(masks_arr))
    else:
        np.save(masks_path, np.asarray(masks_arr, dtype=np.int32))

    results = extract_arrow_features(
        masks_path,
        bev_meta,
        output_dir,
        template_dir=template_dir,
        angle_steps=angle_steps,
        min_mask_area=min_mask_area,
        max_match_score=max_match_score,
        max_overflow=max_overflow,
        bev_img_path=overview_path,
    )
    return arrow_results_to_shp(
        results,
        bev_meta,
        output_dir,
        vertices_path=vertices_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize arrow instance label_map → arrow.shp.")
    parser.add_argument("label_map_npy", help="Path to label_map .npy file with shape (H,W).")
    parser.add_argument("geo_meta_json", help="Path to geo_meta.json or summary.json.")
    parser.add_argument("-o", "--output-dir", default="outputs/apps/shp/arrow", help="Output directory.")
    parser.add_argument("--ply-path", default=None, help="Optional PLY path. Accepted for interface consistency.")
    parser.add_argument("--template-dir", default=None, help="Optional arrow template directory.")
    parser.add_argument("--vertices-path", default=None, help="Optional arrow vertices JSON.")
    parser.add_argument("--angle-steps", type=int, default=72, help="Number of template rotation steps.")
    parser.add_argument("--min-mask-area", type=int, default=2000, help="Minimum arrow mask area in pixels.")
    parser.add_argument(
        "--max-match-score",
        type=float,
        default=DEFAULT_MAX_MATCH_SCORE,
        help="Maximum accepted chamfer score.",
    )
    parser.add_argument(
        "--max-overflow",
        type=float,
        default=DEFAULT_MAX_OVERFLOW,
        help="Maximum accepted template overflow ratio.",
    )
    args = parser.parse_args()

    masks = np.load(args.label_map_npy)
    arrow_masks_to_shp(
        masks,
        args.geo_meta_json,
        args.ply_path,
        args.output_dir,
        template_dir=args.template_dir,
        vertices_path=args.vertices_path,
        angle_steps=args.angle_steps,
        min_mask_area=args.min_mask_area,
        max_match_score=args.max_match_score,
        max_overflow=args.max_overflow,
    )


if __name__ == "__main__":
    main()
