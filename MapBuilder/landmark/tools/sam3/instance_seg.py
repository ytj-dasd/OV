"""Instance segmentation: SAM3 inference on tiler BEV outputs with cross-tile merging.

Usage:
    python -m landmark.tools.sam3.instance_seg input_dir output_dir [--modes rgb intensity]

Runs five fixed prompts (arrow, crosswalk, yellow box, lane line, road marking)
on each BEV mode.  Results are written to::

    output_dir/objs/
    ├── arrow/       {rgb/, intensity/}
    ├── crosswalk/   {rgb/, intensity/}
    ├── nsb/         {rgb/, intensity/}   ("yellow box" prompt)
    ├── laneline/    {rgb/, intensity/}
    └── roadmarking/ {rgb/, intensity/}   ("road marking" prompt)

"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from landmark.tools.sam3.sam3_about import (
    DEFAULT_CONDA_ENV,
    DEFAULT_SAM3_DIR,
    cross_tile_merge,
    generate_distinct_colors,
    get_connected_components_filtered_masks,
    get_overlap_reduced_masks,
    load_masks,
    render_objs_image,
    run_sam3_tile_inference,
)

# Fixed prompts and their output directory names
PROMPTS: list[str] = ["arrow", "crosswalk", "yellow box", "lane line", "road marking"]
PROMPT_DIR: dict[str, str] = {
    "arrow": "arrow",
    "crosswalk": "crosswalk",
    "yellow box": "nsb",
    "lane line": "laneline",
    "road marking": "roadmarking",
}


def _default_prompt_dir(prompt: str) -> str:
    normalized = str(prompt).strip().lower()
    if normalized in PROMPT_DIR:
        return PROMPT_DIR[normalized]
    return normalized.replace(" ", "_").replace("-", "_")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_geo_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tile_name_from_bev(bev_path: Path, mode: str) -> str:
    """Strip the ``_{mode}`` suffix to recover the tile name.

    ``tile_001_x0_y0_rgb.png`` → ``tile_001_x0_y0``
    """
    stem = bev_path.stem
    suffix = f"_{mode}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _compute_global_canvas(
    tile_infos: list[dict[str, Any]],
) -> tuple[int, int, float, float, float]:
    """Return (canvas_h, canvas_w, global_min_x, global_max_y, mpp).

    Each tile in *tile_infos* must have ``bev_origin_xy``, ``bev_size``,
    ``meters_per_pixel``.  BEV convention: ``bev_origin_xy`` is the world
    coord of the *bottom-left* pixel; row 0 = top of image = max world Y.
    """
    mpps = {t["meters_per_pixel"] for t in tile_infos}
    if len(mpps) != 1:
        raise ValueError(f"All tiles must share the same mpp; got {mpps}")
    mpp = mpps.pop()

    global_min_x = min(t["bev_origin_xy"][0] for t in tile_infos)
    global_min_y = min(t["bev_origin_xy"][1] for t in tile_infos)
    global_max_x = max(
        t["bev_origin_xy"][0] + (t["bev_size"][0] - 1) * mpp for t in tile_infos
    )
    global_max_y = max(
        t["bev_origin_xy"][1] + (t["bev_size"][1] - 1) * mpp for t in tile_infos
    )

    canvas_w = math.ceil((global_max_x - global_min_x) / mpp) + 1
    canvas_h = math.ceil((global_max_y - global_min_y) / mpp) + 1
    return canvas_h, canvas_w, global_min_x, global_max_y, mpp


def _tile_has_corner_mapping(tile_info: dict[str, Any]) -> bool:
    corners = tile_info.get("corners_xy")
    pixel_corners = tile_info.get("pixel_corners_xy")
    return (
        isinstance(corners, list)
        and len(corners) == 4
        and isinstance(pixel_corners, list)
        and len(pixel_corners) == 4
    )


def _compute_global_canvas_from_corners(
    tile_infos: list[dict[str, Any]],
) -> tuple[int, int, float, float, float]:
    all_corners: list[np.ndarray] = []
    mpps: list[float] = []
    for tile_info in tile_infos:
        corners = np.asarray(tile_info["corners_xy"], dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"Invalid corners_xy shape: {corners.shape}")
        all_corners.append(corners)
        mpps.append(float(tile_info["meters_per_pixel"]))

    stacked = np.vstack(all_corners)
    global_min_x = float(np.min(stacked[:, 0]))
    global_max_x = float(np.max(stacked[:, 0]))
    global_min_y = float(np.min(stacked[:, 1]))
    global_max_y = float(np.max(stacked[:, 1]))
    mpp = float(np.median(np.asarray(mpps, dtype=np.float64)))

    canvas_w = math.ceil((global_max_x - global_min_x) / mpp) + 1
    canvas_h = math.ceil((global_max_y - global_min_y) / mpp) + 1
    return canvas_h, canvas_w, global_min_x, global_max_y, mpp


def _tile_offset_in_canvas(
    tile_info: dict[str, Any],
    global_min_x: float,
    global_max_y: float,
    mpp: float,
) -> tuple[int, int]:
    """Return (col_offset, row_offset) of this tile's (0,0) in the global canvas."""
    ox, oy = tile_info["bev_origin_xy"]
    tile_h = tile_info["bev_size"][1]
    col_offset = round((ox - global_min_x) / mpp)
    # Tile (0,0) has world_y = oy + (tile_h - 1) * mpp  (top-left = max Y)
    row_offset = round((global_max_y - (oy + (tile_h - 1) * mpp)) / mpp)
    return col_offset, row_offset


def _world_xy_to_canvas_pixel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    global_min_x: float,
    global_max_y: float,
    mpp: float,
) -> tuple[np.ndarray, np.ndarray]:
    col = (x.astype(np.float64) - global_min_x) / mpp
    row = (global_max_y - y.astype(np.float64)) / mpp
    return col, row


def _append_warped_object(
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]],
    obj_mask: np.ndarray,
    tile_info: dict[str, Any],
    *,
    canvas_h: int,
    canvas_w: int,
    global_min_x: float,
    global_max_y: float,
    mpp: float,
    tile_index: int,
) -> bool:
    corners_xy = np.asarray(tile_info["corners_xy"], dtype=np.float32)
    pixel_corners_xy = np.asarray(tile_info["pixel_corners_xy"], dtype=np.float32)
    if corners_xy.shape != (4, 2) or pixel_corners_xy.shape != (4, 2):
        raise ValueError(
            "corners_xy and pixel_corners_xy must both have shape (4,2) "
            f"for tile {tile_info.get('tile_name')}"
        )

    dst_cols, dst_rows = _world_xy_to_canvas_pixel(
        corners_xy[:, 0],
        corners_xy[:, 1],
        global_min_x=global_min_x,
        global_max_y=global_max_y,
        mpp=mpp,
    )
    dst = np.column_stack([dst_cols, dst_rows]).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(pixel_corners_xy, dst)
    warped = cv2.warpPerspective(
        obj_mask.astype(np.uint8),
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if not np.any(warped):
        return False

    rows, cols = np.nonzero(warped)
    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1
    all_objs.append((warped[r0:r1, c0:c1], (slice(r0, r1), slice(c0, c1)), tile_index))
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _postprocess_and_merge(
    tile_imgs: list[Path],
    tile_metas: list[dict[str, Any]],
    tile_out_dirs: list[Path],
    mask_subdir: str | None,
    canvas_h: int,
    canvas_w: int,
    g_min_x: float,
    g_max_y: float,
    mpp: float,
    *,
    obj_straight: bool = False,
    mode_out: Path,
    mode: str,
    text_prompt_label: str,
    input_dir: Path,
) -> int:
    """Per-tile CC filter + overlap dedup + cross-tile merge → save results.

    If *mask_subdir* is not None, masks are read from
    ``tile_out_dir/{mask_subdir}/masks.npz`` (multi-prompt layout).

    Returns the number of final merged objects.
    """
    cc_angle = 15.0 if obj_straight else None
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]] = []
    tile_summaries: list[dict[str, object]] = []
    use_corner_warp = all(_tile_has_corner_mapping(tmeta) for tmeta in tile_metas)

    for idx, (tp, tmeta, tod) in enumerate(
        zip(tile_imgs, tile_metas, tile_out_dirs)
    ):
        mask_dir = tod / mask_subdir if mask_subdir else tod
        mask_file = mask_dir / "masks.npz"
        if not mask_file.is_file():
            print(
                f"[instance-seg] [{idx+1}/{len(tile_imgs)}] missing masks for {tp.name}"
                + (f" prompt={mask_subdir}" if mask_subdir else ""),
                flush=True,
            )
            tile_summaries.append({"tile": tp.name, "status": "missing_masks"})
            continue

        masks = load_masks(mask_file)
        print(
            f"[instance-seg] [{idx+1}/{len(tile_imgs)}] {tp.name} masks={masks.shape[0]}"
            + (f" prompt={mask_subdir}" if mask_subdir else ""),
            flush=True,
        )

        masks = get_connected_components_filtered_masks(masks, angle_threshold=cc_angle)
        masks = get_overlap_reduced_masks(masks, overlap_regions=[])

        if not use_corner_warp:
            col_off, row_off = _tile_offset_in_canvas(tmeta, g_min_x, g_max_y, mpp)

        accepted = 0
        for oi in range(masks.shape[0]):
            obj_mask = masks[oi]
            if not np.any(obj_mask):
                continue

            if use_corner_warp:
                accepted += int(
                    _append_warped_object(
                        all_objs,
                        obj_mask,
                        tmeta,
                        canvas_h=canvas_h,
                        canvas_w=canvas_w,
                        global_min_x=g_min_x,
                        global_max_y=g_max_y,
                        mpp=mpp,
                        tile_index=idx,
                    )
                )
                continue

            oh, ow = obj_mask.shape
            r_start, r_end = row_off, row_off + oh
            c_start, c_end = col_off, col_off + ow

            cr_start = max(0, r_start)
            cr_end = min(canvas_h, r_end)
            cc_start = max(0, c_start)
            cc_end = min(canvas_w, c_end)
            if cr_start >= cr_end or cc_start >= cc_end:
                continue

            sr_start = cr_start - r_start
            sr_end = sr_start + (cr_end - cr_start)
            sc_start = cc_start - c_start
            sc_end = sc_start + (cc_end - cc_start)

            cropped = obj_mask[sr_start:sr_end, sc_start:sc_end].astype(bool)
            if not np.any(cropped):
                continue

            dst_slices = (slice(cr_start, cr_end), slice(cc_start, cc_end))
            all_objs.append((cropped, dst_slices, idx))
            accepted += 1

        tile_summaries.append({
            "tile": tp.name,
            "num_input_masks": int(masks.shape[0]),
            "num_accepted_masks": accepted,
            "status": "ok",
        })

    print(f"[instance-seg] cross-tile merge: {len(all_objs)} objs", flush=True)
    final_masks = cross_tile_merge(all_objs, image_hw=(canvas_h, canvas_w))
    final_obj_num = int(final_masks.max()) + 1 if final_masks.max() >= 0 else 0

    # Save results
    mode_out.mkdir(parents=True, exist_ok=True)
    np.save(mode_out / "final_masks.npy", np.asarray(final_masks, dtype=np.int32))
    vis_rgba = render_objs_image(final_masks, canvas_h, canvas_w)
    Image.fromarray(vis_rgba).save(mode_out / "objs.png")

    summary = {
        "input_dir": str(input_dir),
        "mode": mode,
        "text_prompt": text_prompt_label,
        "canvas_size": [canvas_w, canvas_h],
        "meters_per_pixel": mpp,
        "global_origin_xy": [g_min_x, g_max_y],
        "final_obj_num": final_obj_num,
        "tile_summaries": tile_summaries,
    }
    with (mode_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[instance-seg] mode={mode} prompt={text_prompt_label!r} done: "
        f"{final_obj_num} objects -> {mode_out}",
        flush=True,
    )
    return final_obj_num


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_sam3_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    modes: list[str] | None = None,
    prompts: list[str] | None = None,
    prompt_dir_map: dict[str, str] | None = None,
    sam3_dir: Path | str = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    obj_straight: bool = False,
    th: float | None = None,
    score_th: float | None = None,
) -> dict[str, dict[str, Path]]:
    """Run SAM3 on tiler BEV tiles for all fixed prompts.

    For each mode (rgb, intensity), one SAM3 batch call runs all four prompts
    (model loads once).  Post-processing and cross-tile merge are done per
    prompt.

    Output layout::

        output_dir/objs/{prompt_dir}/{mode}/
            final_masks.npy (H,W int32 label_map), objs.png, summary.json

    Returns ``{prompt_dir: {mode: output_path}}``.
    """
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    sam3_dir = Path(sam3_dir)

    if modes is None:
        modes = ["rgb", "intensity"]
    if prompts is None:
        prompts = list(PROMPTS)
    if prompt_dir_map is None:
        prompt_dir_map = {prompt: _default_prompt_dir(prompt) for prompt in prompts}
    else:
        prompt_dir_map = {str(k): str(v) for k, v in prompt_dir_map.items()}
    missing_prompt_dirs = [prompt for prompt in prompts if prompt not in prompt_dir_map]
    if missing_prompt_dirs:
        raise ValueError(f"Missing prompt_dir_map entries for prompts: {missing_prompt_dirs}")

    bev_dir = input_dir / "bev"
    if not bev_dir.is_dir():
        raise FileNotFoundError(f"BEV directory not found: {bev_dir}")

    geo_meta_path = input_dir / "geo_meta.json"
    if not geo_meta_path.is_file():
        raise FileNotFoundError(f"geo_meta.json not found: {geo_meta_path}")
    geo_meta = _load_geo_meta(geo_meta_path)
    tile_lookup: dict[str, dict[str, Any]] = {
        t["tile_name"]: t for t in geo_meta["tiles"]
    }

    # {prompt_dir: {mode: path}}
    results: dict[str, dict[str, Path]] = {
        prompt_dir_map[prompt]: {} for prompt in prompts
    }

    for mode in modes:
        tile_paths = sorted(bev_dir.glob(f"*_{mode}.png"))
        if not tile_paths:
            print(f"[instance-seg] no BEV tiles found for mode={mode}, skipping", flush=True)
            continue

        # Match BEV images to geo_meta entries
        matched: list[tuple[Path, dict[str, Any]]] = []
        for tp in tile_paths:
            tname = _tile_name_from_bev(tp, mode)
            info = tile_lookup.get(tname)
            if info is None:
                print(f"[instance-seg] WARNING: no geo_meta for {tp.name}, skipping", flush=True)
                continue
            matched.append((tp, info))

        if not matched:
            continue

        tile_imgs = [m[0] for m in matched]
        tile_metas = [m[1] for m in matched]

        # Compute global canvas
        if all(_tile_has_corner_mapping(tmeta) for tmeta in tile_metas):
            canvas_h, canvas_w, g_min_x, g_max_y, mpp = _compute_global_canvas_from_corners(tile_metas)
            canvas_mode = "corner-warp"
        else:
            canvas_h, canvas_w, g_min_x, g_max_y, mpp = _compute_global_canvas(tile_metas)
            canvas_mode = "axis-aligned"
        print(
            f"[instance-seg] mode={mode} tiles={len(matched)} canvas={canvas_w}x{canvas_h} mpp={mpp} merge={canvas_mode}",
            flush=True,
        )

        # ---- Phase 0: SAM3 batch inference (one call per prompt to limit GPU memory) ----
        per_tile_out = output_dir / "objs" / "_tiles" / mode
        tile_out_dirs = [per_tile_out / tp.stem for tp in tile_imgs]

        for prompt in prompts:
            prompt_dir_name = prompt_dir_map[prompt]
            print(
                f"[instance-seg] running SAM3 batch inference for mode={mode} prompt={prompt!r} ...",
                flush=True,
            )
            run_sam3_tile_inference(
                tile_paths=tile_imgs,
                text_prompt=prompt,
                out_dirs=[d / prompt_dir_name for d in tile_out_dirs],
                sam3_dir=sam3_dir,
                conda_env=conda_env,
                th=th,
                score_th=score_th,
            )

        # ---- Phase 1+2: Per-prompt post-processing + merge ----
        for prompt in prompts:
            prompt_dir_name = prompt_dir_map[prompt]
            prompt_mode_out = output_dir / "objs" / prompt_dir_name / mode
            _postprocess_and_merge(
                tile_imgs, tile_metas,
                [d / prompt_dir_name for d in tile_out_dirs],
                mask_subdir=None,
                canvas_h=canvas_h, canvas_w=canvas_w,
                g_min_x=g_min_x, g_max_y=g_max_y, mpp=mpp,
                obj_straight=obj_straight,
                mode_out=prompt_mode_out, mode=mode,
                text_prompt_label=prompt,
                input_dir=input_dir,
            )
            results[prompt_dir_name][mode] = prompt_mode_out

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run instance segmentation on tiler BEV outputs for road-marking prompts.",
    )
    parser.add_argument("input_dir", help="Tiler output directory (contains bev/ and geo_meta.json).")
    parser.add_argument("output_dir", help="Output directory for SAM3 results.")
    parser.add_argument(
        "--modes", nargs="+", default=["rgb", "intensity"],
        help="BEV modes to process (default: rgb intensity).",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Optional custom text prompts. Default: built-in road-marking prompts.",
    )
    parser.add_argument("--sam3-dir", default=str(DEFAULT_SAM3_DIR), help="SAM3 project directory.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="SAM3 conda environment name.")
    parser.add_argument(
        "--th",
        type=float,
        default=None,
        help="Optional SAM3 inference threshold passed through to sam3_inference.py.",
    )
    parser.add_argument(
        "--score-th",
        type=float,
        default=None,
        help="Optional SAM3 score threshold passed through to sam3_inference.py.",
    )
    parser.add_argument(
        "--obj-straight", action="store_true", default=False,
        help="Enforce direction alignment in CC filtering.",
    )
    args = parser.parse_args()

    results = run_sam3_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        modes=args.modes,
        prompts=args.prompts,
        sam3_dir=args.sam3_dir,
        conda_env=args.conda_env,
        obj_straight=args.obj_straight,
        th=args.th,
        score_th=args.score_th,
    )

    for prompt_dir, mode_paths in results.items():
        for mode, p in mode_paths.items():
            print(f"  {prompt_dir}/{mode} → {p}")


if __name__ == "__main__":
    main()
