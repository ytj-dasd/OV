from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from landmark.tools.sam3.bev_part import FIXED_TILE_SIZE, write_bev_parts
from landmark.tools.sam3.sam3_about import (
    DEFAULT_CONDA_ENV,
    DEFAULT_SAM3_DIR,
    cross_tile_merge,
    get_connected_components_filtered_masks,
    get_overlap_reduced_masks,
    load_masks,
    render_objs_image,
    run_sam3_tile_inference,
)


Image.MAX_IMAGE_PIXELS = None


def _load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_overlay(base_image: np.ndarray, label_map: np.ndarray, out_path: Path) -> None:
    if base_image.ndim == 2:
        base_rgb = np.stack([base_image] * 3, axis=2)
    else:
        base_rgb = np.asarray(base_image)[..., :3].copy()

    h, w = label_map.shape
    vis_rgba = render_objs_image(label_map, h, w)
    alpha = (vis_rgba[:, :, 3].astype(np.float32) / 255.0) * 0.55
    overlay = base_rgb.astype(np.float32)
    overlay = overlay * (1.0 - alpha[..., None]) + vis_rgba[:, :, :3].astype(np.float32) * alpha[..., None]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def _append_warped_object_to_rotated(
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]],
    obj_mask: np.ndarray,
    tile_info: dict[str, Any],
    *,
    rotated_shape: tuple[int, int],
    tile_index: int,
) -> bool:
    rot_h, rot_w = rotated_shape
    src = np.asarray(tile_info["pixel_corners_xy"], dtype=np.float32)
    dst = np.asarray(tile_info["rotated_pixel_corners_xy"], dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("pixel_corners_xy and rotated_pixel_corners_xy must both have shape (4,2)")

    matrix = cv2.getPerspectiveTransform(src, dst)
    obj_rows, obj_cols = np.nonzero(obj_mask)
    if len(obj_rows) == 0:
        return False
    src_r0 = int(obj_rows.min())
    src_r1 = int(obj_rows.max()) + 1
    src_c0 = int(obj_cols.min())
    src_c1 = int(obj_cols.max()) + 1
    src_crop = obj_mask[src_r0:src_r1, src_c0:src_c1].astype(np.uint8)

    crop_corners = np.asarray(
        [
            [src_c0, src_r0],
            [src_c1 - 1, src_r0],
            [src_c1 - 1, src_r1 - 1],
            [src_c0, src_r1 - 1],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    dst_corners = cv2.perspectiveTransform(crop_corners, matrix).reshape(4, 2)
    c0 = max(0, int(np.floor(float(dst_corners[:, 0].min()))) - 2)
    c1 = min(rot_w, int(np.ceil(float(dst_corners[:, 0].max()))) + 3)
    r0 = max(0, int(np.floor(float(dst_corners[:, 1].min()))) - 2)
    r1 = min(rot_h, int(np.ceil(float(dst_corners[:, 1].max()))) + 3)
    if r1 <= r0 or c1 <= c0:
        return False

    src_translate = np.asarray(
        [[1.0, 0.0, float(src_c0)], [0.0, 1.0, float(src_r0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dst_translate = np.asarray(
        [[1.0, 0.0, float(-c0)], [0.0, 1.0, float(-r0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    roi_matrix = dst_translate @ matrix @ src_translate
    warped = cv2.warpPerspective(
        src_crop,
        roi_matrix,
        (c1 - c0, r1 - r0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not np.any(warped):
        return False

    rows, cols = np.nonzero(warped)
    crop_r0 = int(rows.min())
    crop_r1 = int(rows.max()) + 1
    crop_c0 = int(cols.min())
    crop_c1 = int(cols.max()) + 1
    out_r0 = r0 + crop_r0
    out_r1 = r0 + crop_r1
    out_c0 = c0 + crop_c0
    out_c1 = c0 + crop_c1
    all_objs.append(
        (
            warped[crop_r0:crop_r1, crop_c0:crop_c1].astype(bool),
            (slice(out_r0, out_r1), slice(out_c0, out_c1)),
            tile_index,
        )
    )
    return True


def _rotate_label_map_back(
    rotated_label_map: np.ndarray,
    inverse_affine: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = output_shape
    final = np.full((out_h, out_w), -1, dtype=np.int32)
    max_id = int(rotated_label_map.max())
    if max_id < 0:
        return final

    for obj_id in range(max_id + 1):
        mask = rotated_label_map == obj_id
        if not np.any(mask):
            continue
        warped = cv2.warpAffine(
            mask.astype(np.uint8),
            np.asarray(inverse_affine, dtype=np.float32),
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        final[warped] = obj_id
    return final


def run_instance_seg_v2(
    bev_path: Path | str,
    prompt: str,
    geo_meta_path: Path | str,
    tile_size_m: float,
    output_dir: Path | str,
    *,
    spt_road_path: Path | str | None = None,
    sam3_dir: Path | str = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    th: float | None = 0.5,
    score_th: float | None = 0.2,
    fill_ratio_threshold: float = 0.10,
    tile_overlap_ratio: float = 0.10,
) -> dict[str, Path]:
    bev_path = Path(bev_path).expanduser()
    geo_meta_path = Path(geo_meta_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    sam3_dir = Path(sam3_dir).expanduser()

    if not bev_path.is_file():
        raise FileNotFoundError(f"BEV image not found: {bev_path}")
    if not geo_meta_path.is_file():
        raise FileNotFoundError(f"geo_meta.json not found: {geo_meta_path}")

    parts_payload = write_bev_parts(
        bev_path,
        geo_meta_path,
        output_dir,
        spt_road_path=spt_road_path,
        tile_size_m=tile_size_m,
        fill_ratio_threshold=fill_ratio_threshold,
        tile_overlap_ratio=tile_overlap_ratio,
    )

    parts_dir = output_dir / "parts"
    sam3_out_root = output_dir / "sam3"
    result_dir = output_dir / "result"
    sam3_out_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = [parts_dir / "bev" / f"{part['tile_name']}.png" for part in parts_payload["parts"]]
    tile_out_dirs = [sam3_out_root / part["tile_name"] for part in parts_payload["parts"]]
    missing_tile_outputs = [out_dir for out_dir in tile_out_dirs if not (out_dir / "masks.npz").is_file()]
    if tile_paths and missing_tile_outputs:
        run_sam3_tile_inference(
            tile_paths,
            prompt,
            tile_out_dirs,
            sam3_dir=sam3_dir,
            conda_env=conda_env,
            th=th,
            score_th=score_th,
        )

    rotated_shape = tuple(int(v) for v in parts_payload["rotated_shape"])
    tile_grid_positions = [
        (int(part["grid_row"]), int(part["grid_col"]))
        for part in parts_payload["parts"]
    ]
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]] = []
    tile_summaries: list[dict[str, object]] = []
    for idx, (part, tile_path, out_dir) in enumerate(zip(parts_payload["parts"], tile_paths, tile_out_dirs)):
        mask_file = out_dir / "masks.npz"
        if not mask_file.is_file():
            tile_summaries.append({"tile": part["tile_name"], "status": "missing_masks"})
            continue

        masks = load_masks(mask_file)
        masks = get_connected_components_filtered_masks(masks, angle_threshold=None)
        masks = get_overlap_reduced_masks(masks, overlap_regions=[])

        accepted = 0
        for obj_mask in masks:
            if not np.any(obj_mask):
                continue
            accepted += int(
                _append_warped_object_to_rotated(
                    all_objs,
                    obj_mask,
                    part,
                    rotated_shape=rotated_shape,
                    tile_index=idx,
                )
            )

        tile_summaries.append(
            {
                "tile": part["tile_name"],
                "tile_path": str(tile_path),
                "num_input_masks": int(masks.shape[0]),
                "num_accepted_masks": int(accepted),
                "status": "ok",
            }
        )

    rotated_label_map = cross_tile_merge(
        all_objs,
        image_hw=rotated_shape,
        tile_grid_positions=tile_grid_positions,
        min_overlap_px=100,
        min_overlap_ratio=0.03,
    )
    inverse_affine = np.asarray(parts_payload["rotated_to_original_affine"], dtype=np.float32)
    original_shape = tuple(int(v) for v in parts_payload["original_shape"])
    final_label_map = _rotate_label_map_back(rotated_label_map, inverse_affine, original_shape)

    bev_image = np.asarray(Image.open(bev_path))
    rotated_bev = np.asarray(Image.open(parts_dir / "rotated_bev.png"))
    np.save(result_dir / "rotated_label_map.npy", np.asarray(rotated_label_map, dtype=np.int32))
    np.save(result_dir / "label_map.npy", np.asarray(final_label_map, dtype=np.int32))
    _save_overlay(rotated_bev, rotated_label_map, result_dir / "rotated_objs.png")
    _save_overlay(bev_image, final_label_map, result_dir / "objs.png")

    final_obj_num = int(final_label_map.max()) + 1 if int(final_label_map.max()) >= 0 else 0
    summary = {
        "input_bev": str(bev_path),
        "input_geo_meta": str(geo_meta_path),
        "input_spt_road": str(Path(spt_road_path).expanduser()) if spt_road_path is not None else None,
        "prompt": prompt,
        "original_shape": list(original_shape),
        "rotated_shape": list(rotated_shape),
        "rotation_deg": float(parts_payload["rotation_deg"]),
        "tile_size_m": float(tile_size_m),
        "tile_size_px_rotated": int(parts_payload["tile_size_px_rotated"]),
        "tile_overlap_ratio": float(parts_payload["tile_overlap_ratio"]),
        "tile_stride_px_rotated": int(parts_payload["tile_stride_px_rotated"]),
        "fill_ratio_threshold": float(fill_ratio_threshold),
        "num_parts": int(parts_payload["num_parts"]),
        "final_obj_num": final_obj_num,
        "outputs": {
            "parts_json": str(parts_dir / "parts.json"),
            "rotated_label_map": str(result_dir / "rotated_label_map.npy"),
            "rotated_objs": str(result_dir / "rotated_objs.png"),
            "label_map": str(result_dir / "label_map.npy"),
            "objs": str(result_dir / "objs.png"),
        },
        "tile_summaries": tile_summaries,
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "parts_json": parts_dir / "parts.json",
        "rotated_bev": parts_dir / "rotated_bev.png",
        "rotated_label_map": result_dir / "rotated_label_map.npy",
        "rotated_objs": result_dir / "rotated_objs.png",
        "label_map": result_dir / "label_map.npy",
        "objs": result_dir / "objs.png",
        "summary": result_dir / "summary.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-BEV SAM3 instance segmentation with rotated tiling.")
    parser.add_argument("bev_path", help="Input BEV image path.")
    parser.add_argument("prompt", help="Single SAM3 text prompt.")
    parser.add_argument("geo_meta_path", help="Input geo_meta.json path.")
    parser.add_argument("--spt-road", dest="spt_road_path", help="Optional spt-road.png used only for rotation estimation.")
    parser.add_argument("--tile-size", type=float, required=True, dest="tile_size_m", help="Tile size in meters.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--sam3-dir", default=str(DEFAULT_SAM3_DIR), help="SAM3 repository directory.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="SAM3 conda env or python path.")
    parser.add_argument("--th", type=float, default=0.5, help="Optional SAM3 threshold.")
    parser.add_argument("--score-th", type=float, default=0.2, help="Optional SAM3 score threshold.")
    parser.add_argument(
        "--fill-threshold",
        type=float,
        default=0.10,
        dest="fill_ratio_threshold",
        help="Minimum valid-pixel ratio for keeping a tile.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.10,
        dest="tile_overlap_ratio",
        help="Neighboring-tile overlap ratio in rotated-image pixels.",
    )
    args = parser.parse_args()
    run_instance_seg_v2(
        args.bev_path,
        args.prompt,
        args.geo_meta_path,
        args.tile_size_m,
        args.out,
        spt_road_path=args.spt_road_path,
        sam3_dir=args.sam3_dir,
        conda_env=args.conda_env,
        th=args.th,
        score_th=args.score_th,
        fill_ratio_threshold=args.fill_ratio_threshold,
        tile_overlap_ratio=args.tile_overlap_ratio,
    )


if __name__ == "__main__":
    main()
