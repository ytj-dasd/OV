"""SAM3 tiled inference pipeline.

Runs SAM3 inference in the ``sam3`` conda environment, then performs
per-tile post-processing and cross-tile merging.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
import cv2
import colorsys


Image.MAX_IMAGE_PIXELS = None


TILE_NAME_PATTERN = re.compile(r"_x(-?\d+)_y(-?\d+)")
DEFAULT_CONDA_ENV = "sam3"


def _default_sam3_dir() -> Path:
    candidates = [
        Path.cwd().parent / "sam3",
        Path.home() / "workspace" / "sam3",
        Path.home() / "zyh_workspace" / "OpenSource" / "sam3",
        Path.home() / "Data" / "zyh_workspace" / "OpenSource" / "sam3",
    ]
    for candidate in candidates:
        if (candidate / "sam3").is_dir() and (candidate / "model" / "sam3.pt").is_file():
            return candidate
    return candidates[0]


DEFAULT_SAM3_DIR = _default_sam3_dir()


def _resolve_sam3_python(conda_env: str) -> list[str]:
    env_name = str(conda_env).strip()
    candidate_paths = [
        Path(env_name).expanduser(),
        Path.home() / "miniconda3" / "envs" / env_name / "python.exe",
        Path.home() / "anaconda3" / "envs" / env_name / "python.exe",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return [str(candidate)]
    return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]


def _resolve_sam3_entrypoint(sam3_dir: Path) -> Path:
    candidates = [
        sam3_dir / "sam3_inference.py",
        sam3_dir / "masks.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    adapter = Path(__file__).with_name("sam3_inference_adapter.py")
    if (sam3_dir / "sam3").is_dir() and (sam3_dir / "model" / "sam3.pt").is_file():
        return adapter
    raise FileNotFoundError(f"Not a usable SAM3 checkout: {sam3_dir}")


def parse_tile_minxy(tile_path: Path) -> tuple[int, int]:
    match = TILE_NAME_PATTERN.search(tile_path.stem)
    if match is None:
        raise ValueError(f"Cannot parse minxy from tile name: {tile_path.name}")
    return (int(match.group(1)), int(match.group(2)))


def run_sam3_tile_inference(
    tile_paths: list[Path],
    text_prompt: str | list[str],
    out_dirs: list[Path],
    sam3_dir: Path = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    th: float | None = None,
    score_th: float | None = None,
) -> list[Path]:
    """Call a SAM3 inference entrypoint via conda run.

    Prefers ``sam3_inference.py`` and falls back to the older ``masks.py``.
    """
    if len(tile_paths) != len(out_dirs):
        raise ValueError("tile_paths and out_dirs must have the same length")
    if not tile_paths:
        return out_dirs

    sam3_dir = Path(sam3_dir).expanduser()
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    entrypoint = _resolve_sam3_entrypoint(sam3_dir)
    image_args = json.dumps([str(p.resolve()) for p in tile_paths], ensure_ascii=False)
    out_args = json.dumps([str(d.resolve()) for d in out_dirs], ensure_ascii=False)

    python_cmd = _resolve_sam3_python(conda_env)
    cmd = [*python_cmd, str(entrypoint)]
    if entrypoint.name == "sam3_inference.py":
        if isinstance(text_prompt, list):
            raise ValueError(
                "sam3_inference.py only supports a single --text prompt per batch call"
            )
        cmd.extend([
            image_args,
            "--batch",
            "--text", text_prompt,
            "--out", out_args,
        ])
        if th is not None:
            cmd.extend(["--th", str(float(th))])
        if score_th is not None:
            cmd.extend(["--score-th", str(float(score_th))])
    else:
        text_arg = json.dumps(text_prompt) if isinstance(text_prompt, list) else text_prompt
        cmd.extend([
            "--batch", image_args,
            "--text", text_arg,
            "--out", out_args,
        ])

    n_prompts = len(text_prompt) if isinstance(text_prompt, list) else 1
    print(
        f"[sam3] infer via {entrypoint.name} num_tiles={len(tile_paths)} prompts={n_prompts} "
        f"first_tile={tile_paths[0].name} last_tile={tile_paths[-1].name}",
        flush=True,
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(sam3_dir)
        subprocess.run(cmd, cwd=str(sam3_dir), check=True, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run SAM3: 'conda' not found in PATH. "
            "Ensure conda is installed/initialized and the 'sam3' environment exists."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SAM3 batch inference failed (exit code {e.returncode})") from e
    return out_dirs


def load_masks(mask_path: Path) -> np.ndarray:
    with np.load(mask_path) as data:
        if "masks" in data:
            masks = data["masks"]
        else:
            first_key = next(iter(data.files), None)
            if first_key is None:
                raise ValueError(f"No arrays found in mask file: {mask_path}")
            masks = data[first_key]

    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape in {mask_path}: {masks.shape}")
    return masks.astype(bool)


def connected_components_filter(
    img: np.ndarray, area_ratio: float = 0.1, angle_threshold: float | None = 15
) -> np.ndarray:
    """Filter connected components. Keep main + components with sufficient area.

    When *angle_threshold* is a finite number, also require direction alignment
    with the main component. When ``None``, only the area ratio is checked.
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    if num_labels <= 1:
        return img

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    max_area = areas[max_idx - 1]

    check_angle = angle_threshold is not None

    def get_info(label_idx: int):
        pts = np.column_stack(np.where(labels == label_idx))
        if len(pts) < 2:
            return 0, centroids[label_idx]
        mean, eigenvectors = cv2.PCACompute(pts.astype(np.float32), mean=None)
        angle = np.arctan2(eigenvectors[0, 0], eigenvectors[0, 1]) * 180 / np.pi
        return angle, centroids[label_idx]

    def get_angle_diff(a1: float, a2: float) -> float:
        diff = abs(a1 - a2) % 180
        return min(diff, 180 - diff)

    if check_angle:
        main_angle, main_center = get_info(max_idx)

    out_mask = np.zeros_like(img)
    out_mask[labels == max_idx] = 255

    for i in range(1, num_labels):
        if i == max_idx:
            continue
        curr_area = stats[i, cv2.CC_STAT_AREA]
        if curr_area > max_area * area_ratio:
            if check_angle:
                curr_angle, curr_center = get_info(i)
                dx = curr_center[0] - main_center[0]
                dy = curr_center[1] - main_center[1]
                link_angle = np.arctan2(dy, dx) * 180 / np.pi
                diff_self = get_angle_diff(curr_angle, main_angle)
                diff_link = get_angle_diff(link_angle, main_angle)
                if diff_self < angle_threshold and diff_link < angle_threshold:
                    out_mask[labels == i] = 255
            else:
                out_mask[labels == i] = 255

    return out_mask


def get_connected_components_filtered_masks(
    masks: np.ndarray, area_ratio: float = 0.1, angle_threshold: float | None = 15
) -> np.ndarray:
    filtered_masks = []
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks_ = masks[:, 0]
    elif masks.ndim == 3:
        masks_ = masks
    else:
        raise ValueError("masks must have shape (K,1,H,W) or (K,H,W)")
    if masks_.shape[0] == 0:
        return masks_.astype(np.bool_)

    for i in range(masks_.shape[0]):
        mask = masks_[i].astype(np.uint8) * 255
        filtered_mask = connected_components_filter(
            mask, area_ratio=area_ratio, angle_threshold=angle_threshold
        )
        filtered_mask = (filtered_mask > 0).astype(np.bool_)
        filtered_masks.append(filtered_mask)
    return np.stack(filtered_masks, axis=0)


def get_overlap_reduced_masks(
    masks: np.ndarray,
    overlap_regions: list[tuple[slice, slice]],
    *,
    cover_ratio: float = 0.9,
    min_small_area: int = 20,
) -> np.ndarray:
    """Remove small masks that are (almost) covered by a larger mask."""
    masks = np.asarray(masks)
    _ = overlap_regions  # kept for backward compatibility
    if masks.size == 0:
        return masks
    if masks.ndim != 3:
        raise ValueError("masks must have shape (K,H,W)")
    cover_ratio = float(cover_ratio)
    min_small_area = int(min_small_area)
    if cover_ratio <= 0.0 or cover_ratio > 1.0:
        raise ValueError("cover_ratio must be in (0,1]")
    if min_small_area < 0:
        raise ValueError("min_small_area must be >= 0")

    k = masks.shape[0]
    areas = masks.reshape(k, -1).sum(axis=1).astype(np.int64)
    remove = np.zeros((k,), dtype=bool)

    idxs = np.argsort(-areas)
    for ii in range(len(idxs)):
        bi = int(idxs[ii])
        if remove[bi] or areas[bi] <= 0:
            continue
        big = masks[bi]
        for jj in range(ii + 1, len(idxs)):
            si = int(idxs[jj])
            if remove[si]:
                continue
            a_small = int(areas[si])
            if a_small <= 0 or a_small < min_small_area:
                continue
            small = masks[si]
            inter = int(np.logical_and(big, small).sum())
            if inter / float(a_small) >= cover_ratio:
                remove[si] = True

    kept = np.where(~remove)[0]
    return masks[kept]


def mask_pre_filter(mask: np.ndarray) -> np.ndarray:
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    main_label = int(np.argmax(counts))
    main_area = int(counts[main_label])
    keep_labels = {main_label}

    for label_id in range(1, num_features + 1):
        if label_id == main_label:
            continue
        if counts[label_id] >= 0.1 * main_area:
            keep_labels.add(label_id)

    return np.isin(labeled, list(keep_labels))


def crop_mask_to_image(
    mask: np.ndarray,
    minxy: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[slice, slice]] | None:
    min_x, min_y = minxy
    tile_h, tile_w = mask.shape
    image_w, image_h = image_size
    max_x = min_x + tile_w
    max_y = min_y + tile_h

    dst_left = max(0, min_x)
    dst_top = max(0, min_y)
    dst_right = min(image_w, max_x)
    dst_bottom = min(image_h, max_y)
    if dst_left >= dst_right or dst_top >= dst_bottom:
        return None

    src_left = dst_left - min_x
    src_top = dst_top - min_y
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)

    cropped_mask = mask[src_top:src_bottom, src_left:src_right]
    return cropped_mask, (slice(dst_top, dst_bottom), slice(dst_left, dst_right))


def merge_mask_into_final(
    final_masks: np.ndarray,
    mask: np.ndarray,
    dst_slices: tuple[slice, slice],
    final_obj_num: int,
) -> int:
    dst_view = final_masks[dst_slices]
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return final_obj_num

    occupied_values = dst_view[mask_bool]
    existing_values = occupied_values[occupied_values != -1]
    if existing_values.size < 80:
        dst_view[mask_bool] = final_obj_num
        return final_obj_num + 1

    values, counts = np.unique(existing_values, return_counts=True)
    best_value = int(values[np.argmax(counts)])
    dst_view[mask_bool] = best_value
    return final_obj_num


def generate_distinct_colors(count: int) -> list[tuple[int, int, int]]:
    """Generate *count* visually distinct RGB colours using evenly-spaced hues."""
    if count <= 0:
        return [(200, 200, 200)]
    colors: list[tuple[int, int, int]] = []
    # golden-ratio offset keeps nearby indices far apart in hue
    golden = (1 + 5**0.5) / 2
    for i in range(count):
        h = (i / golden) % 1.0
        s = 0.75 + 0.2 * ((i % 3) / 2.0)  # vary saturation slightly
        v = 0.85 + 0.15 * ((i % 2) / 1.0)  # vary value slightly
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def render_objs_image(
    final_masks: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Render an RGBA image where each obj has a distinct colour; background is transparent."""
    vis = np.zeros((height, width, 4), dtype=np.uint8)
    max_id = int(final_masks.max())
    if max_id < 0:
        return vis

    colors = generate_distinct_colors(max_id + 1)
    for obj_id in range(max_id + 1):
        mask = final_masks == obj_id
        if not np.any(mask):
            continue
        r, g, b = colors[obj_id]
        vis[mask] = (r, g, b, 255)
    return vis


def save_objs_geotiff(
    vis_rgba: np.ndarray,
    out_path: Path,
    src_tif_path: Path,
) -> None:
    """Write *vis_rgba* (H,W,4) as a GeoTIFF, copying CRS & transform from *src_tif_path*."""
    with rasterio.open(src_tif_path) as src:
        transform = src.transform
        crs = src.crs

    h, w = vis_rgba.shape[:2]
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 4,
        "dtype": "uint8",
        "transform": transform,
        "crs": crs,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for band in range(4):
            dst.write(vis_rgba[:, :, band], band + 1)


def _slices_overlap(s1: slice, s2: slice) -> slice | None:
    """Return the overlapping sub-range of two slices, or None."""
    start = max(s1.start, s2.start)
    stop = min(s1.stop, s2.stop)
    if start >= stop:
        return None
    return slice(start, stop)


def cross_tile_merge(
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]],
    image_hw: tuple[int, int],
    *,
    tile_grid_positions: list[tuple[int, int]] | None = None,
    min_overlap_px: int = 100,
    min_overlap_ratio: float = 0.03,
) -> np.ndarray:
    """Merge objs from different tiles using Union-Find; same-tile objs stay separate.

    Parameters
    ----------
    all_objs : list of (cropped_mask, (row_slice, col_slice), tile_idx)
    image_hw : (height, width) of the full image
    tile_grid_positions : optional list mapping tile_idx to (grid_row, grid_col)
    min_overlap_px : minimum overlapping pixels required before merging
    min_overlap_ratio : minimum overlap / smaller-object-area required before merging

    Returns
    -------
    final_masks : int32 array (H, W) with merged object IDs (-1 = background)
    """
    height, width = image_hw
    n = len(all_objs)
    if n == 0:
        return np.full((height, width), -1, dtype=np.int32)

    # --- Union-Find ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    min_overlap_px = int(min_overlap_px)
    min_overlap_ratio = float(min_overlap_ratio)
    areas = np.asarray([int(mask.astype(bool).sum()) for mask, _slices, _tile in all_objs], dtype=np.int64)
    merge_edges = 0
    rejected_edges = 0

    # Pairwise overlap check (only across different tiles)
    for i in range(n):
        mask_i, (rs_i, cs_i), tile_i = all_objs[i]
        for j in range(i + 1, n):
            _, (rs_j, cs_j), tile_j = all_objs[j]
            if tile_i == tile_j:
                continue
            if tile_grid_positions is not None:
                row_i, col_i = tile_grid_positions[tile_i]
                row_j, col_j = tile_grid_positions[tile_j]
                if abs(row_i - row_j) > 1 or abs(col_i - col_j) > 1:
                    continue
            row_ov = _slices_overlap(rs_i, rs_j)
            col_ov = _slices_overlap(cs_i, cs_j)
            if row_ov is None or col_ov is None:
                continue

            # Local coords within each mask for the overlapping region
            ri_s = row_ov.start - rs_i.start
            ri_e = row_ov.stop - rs_i.start
            ci_s = col_ov.start - cs_i.start
            ci_e = col_ov.stop - cs_i.start

            rj_s = row_ov.start - rs_j.start
            rj_e = row_ov.stop - rs_j.start
            cj_s = col_ov.start - cs_j.start
            cj_e = col_ov.stop - cs_j.start

            mask_j = all_objs[j][0]
            overlap = np.logical_and(
                mask_i[ri_s:ri_e, ci_s:ci_e],
                mask_j[rj_s:rj_e, cj_s:cj_e],
            )
            overlap_px = int(overlap.sum())
            if overlap_px <= 0:
                continue
            small_area = int(min(areas[i], areas[j]))
            overlap_ratio = overlap_px / float(small_area) if small_area > 0 else 0.0
            if overlap_px >= min_overlap_px and overlap_ratio >= min_overlap_ratio:
                union(i, j)
                merge_edges += 1
            else:
                rejected_edges += 1

    # Map each root to a sequential final ID
    root_to_id: dict[int, int] = {}
    next_id = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1

    # Write merged masks (union of all objs in the same group)
    final_masks = np.full((height, width), -1, dtype=np.int32)
    for i, (mask, slices, _tile) in enumerate(all_objs):
        fid = root_to_id[find(i)]
        dst = final_masks[slices]
        dst[mask.astype(bool)] = fid

    print(
        "[sam3] cross_tile_merge: "
        f"{n} objs -> {next_id} merged groups "
        f"(merge_edges={merge_edges}, rejected_edges={rejected_edges}, "
        f"min_overlap_px={min_overlap_px}, min_overlap_ratio={min_overlap_ratio:g})",
        flush=True,
    )
    return final_masks


def run_sam3_on_tiles(
    bev_image_path: str | Path,
    tiles_dir: str | Path,
    text_prompt: str,
    out_dir: str | Path,
    sam3_dir: str | Path = DEFAULT_SAM3_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    obj_straight: bool = False,
) -> Path:
    bev_image_path = Path(bev_image_path)
    tiles_dir = Path(tiles_dir)
    out_dir = Path(out_dir)
    sam3_dir = Path(sam3_dir)

    if not bev_image_path.is_file():
        raise FileNotFoundError(f"BEV image not found: {bev_image_path}")
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_tile_out_dir = out_dir / "tiles"
    per_tile_out_dir.mkdir(parents=True, exist_ok=True)

    bev_image = Image.open(bev_image_path).convert("RGB")
    width_org, height_org = bev_image.size

    tile_paths = sorted([p for p in tiles_dir.iterdir() if p.is_file()])
    tile_summaries: list[dict[str, object]] = []
    print(
        f"[sam3] start bev={bev_image_path} tiles_dir={tiles_dir} num_tiles={len(tile_paths)} text={text_prompt!r}",
        flush=True,
    )

    # ---- Batch inference ----
    tile_out_dirs = [per_tile_out_dir / tile_path.stem for tile_path in tile_paths]
    if tile_paths:
        run_sam3_tile_inference(
            tile_paths=tile_paths,
            text_prompt=text_prompt,
            out_dirs=tile_out_dirs,
            sam3_dir=sam3_dir,
            conda_env=conda_env,
        )

    # ================================================================
    # Phase 1: Per-tile post-processing
    #   - connected-component filtering (keep main + aligned components)
    #   - coverage dedup (remove small objs mostly covered by larger ones)
    #   - crop to global coords and collect
    # ================================================================
    all_objs: list[tuple[np.ndarray, tuple[slice, slice], int]] = []

    for idx, (tile_path, tile_out_dir) in enumerate(zip(tile_paths, tile_out_dirs), start=1):
        print(f"[sam3] [{idx}/{len(tile_paths)}] processing tile={tile_path.name}", flush=True)

        mask_file = tile_out_dir / "masks.npz"
        if not mask_file.is_file():
            print(f"[sam3] [{idx}/{len(tile_paths)}] missing mask file for tile={tile_path.name}", flush=True)
            tile_summaries.append({"tile": tile_path.name, "status": "missing_masks"})
            continue

        masks = load_masks(mask_file)
        minxy = parse_tile_minxy(tile_path)
        print(
            f"[sam3] [{idx}/{len(tile_paths)}] loaded masks tile={tile_path.name} num_masks={masks.shape[0]} minxy={minxy}",
            flush=True,
        )

        # Step 1a: connected-component filtering per obj
        # obj_straight → enforce angle alignment; otherwise area-only
        cc_angle = 15.0 if obj_straight else None
        masks = get_connected_components_filtered_masks(masks, angle_threshold=cc_angle)

        # Step 1b: coverage dedup within this tile
        masks = get_overlap_reduced_masks(masks, overlap_regions=[])

        # Crop each surviving obj to global image coords and collect
        accepted_objects = 0
        for obj_idx in range(masks.shape[0]):
            obj_mask = masks[obj_idx]
            if not np.any(obj_mask):
                continue

            cropped = crop_mask_to_image(obj_mask, minxy=minxy, image_size=(width_org, height_org))
            if cropped is None:
                continue
            cropped_mask, dst_slices = cropped
            if not np.any(cropped_mask):
                continue

            tile_idx = idx - 1  # 0-based tile index
            all_objs.append((cropped_mask.astype(bool), dst_slices, tile_idx))
            accepted_objects += 1

        tile_summaries.append(
            {
                "tile": tile_path.name,
                "tile_out_dir": str(tile_out_dir),
                "num_input_masks": int(masks.shape[0]),
                "num_accepted_masks": int(accepted_objects),
                "status": "ok",
            }
        )
        print(
            f"[sam3] [{idx}/{len(tile_paths)}] done tile={tile_path.name} accepted_masks={accepted_objects}",
            flush=True,
        )

    # ================================================================
    # Phase 2: Cross-tile merge
    #   - objs from different tiles that overlap → union (merge)
    #   - objs from the same tile → never merge
    # ================================================================
    print(
        f"[sam3] phase2: cross-tile merge total_objs={len(all_objs)}",
        flush=True,
    )
    final_masks = cross_tile_merge(all_objs, image_hw=(height_org, width_org))
    final_obj_num = int(final_masks.max()) + 1 if final_masks.max() >= 0 else 0

    # ---- Save results ----
    np.save(out_dir / "final_masks.npy", final_masks)

    vis_rgba = render_objs_image(final_masks, height_org, width_org)
    Image.fromarray(vis_rgba).save(out_dir / "objs.png")
    save_objs_geotiff(vis_rgba, out_dir / "objs.tif", bev_image_path)

    summary = {
        "bev_image": str(bev_image_path),
        "tiles_dir": str(tiles_dir),
        "text_prompt": text_prompt,
        "final_obj_num": int(final_obj_num),
        "final_masks_file": "final_masks.npy",
        "objs_png": "objs.png",
        "objs_tif": "objs.tif",
        "tile_summaries": tile_summaries,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[sam3] finished final_obj_num={final_obj_num} saved objs.png & objs.tif to {out_dir}",
        flush=True,
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 inference on multiple tiles and merge masks.")
    parser.add_argument("--bev", default="outputs/rgb_bev_filled.tif", help="Path to the original BEV GeoTIFF image.")
    parser.add_argument("--tiles", default="outputs/tiles", help="Directory containing tile images.")
    parser.add_argument("--text", default="road marking", help="Text prompt for SAM3 inference.")
    parser.add_argument("--out-dir", default="outputs/sam3_output", help="Output directory.")
    parser.add_argument(
        "--sam3-dir",
        default=str(DEFAULT_SAM3_DIR),
        help="Path to sam3 project directory (contains masks.py).",
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help="Name of the conda environment with SAM3 installed.",
    )
    parser.add_argument(
        "--obj-straight",
        action="store_true",
        default=False,
        help="Treat objects as strip-shaped: enforce direction alignment in connected-component filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = run_sam3_on_tiles(
        bev_image_path=args.bev,
        tiles_dir=args.tiles,
        text_prompt=args.text,
        out_dir=args.out_dir,
        sam3_dir=args.sam3_dir,
        conda_env=args.conda_env,
        obj_straight=args.obj_straight,
    )
    print(f"Saved SAM3 merged outputs to {out_dir}")


if __name__ == "__main__":
    main()
