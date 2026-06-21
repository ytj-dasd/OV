"""pre_part tool — render whole-PLY pre-part BEV products + geo_meta.json."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

Image.MAX_IMAGE_PIXELS = None

from landmark.tools.pc_process.part.tile_part import (
    build_default_geo_tile_payload,
    write_tile_parts_json,
)
from landmark.tools.pc_process.bev import (
    _csf_ground_mask,
    _height_raster_to_rgb,
    render_bev,
    render_height_raster,
)

PC_CSF_RENDER_MPP = 0.02
BASE_RENDER_MODES = ("rgb", "intensity")
_DEFAULT_TILE_SIZE_M = 40.0
_DEFAULT_FILL_RATIO_THRESHOLD = 0.10
_DEFAULT_FILL_CELL_SIZE_M = 0.50
_REFERENCE_PLY_PATH = Path("data/cj/pc_with_preds.ply")
_FALLBACK_REFERENCE_VERTEX_COUNT = 350_000_000
_LARGE_PLY_RATIO_THRESHOLD = 1.5
_STREAM_CHUNK_SIZE = 2_000_000


def _build_mask_from_render(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        mask = img > 0
    else:
        mask = np.any(img > 0, axis=2)
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def _mode_requirement(mode: str) -> str:
    if mode == "mask":
        return "point occupancy"
    if mode == "rgb":
        return "RGB colours"
    if mode == "intensity":
        return "scalar_Intensity"
    return mode


def _write_csf_filtered_ply(
    ply_path: Path,
    output_path: Path,
) -> Path:
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise KeyError(f"'vertex' element not found in {ply_path}")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise KeyError(f"PLY vertex fields must include {sorted(required)}")

    points = np.stack([vertex.data["x"], vertex.data["y"], vertex.data["z"]], axis=-1)
    ground_mask = _csf_ground_mask(points)
    filtered_vertex = vertex.data[ground_mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_ply = PlyData(
        [PlyElement.describe(filtered_vertex, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    )
    filtered_ply.write(str(output_path))
    print(
        f"[pre_part] saved CSF-filtered PLY -> {output_path}  "
        f"({int(np.count_nonzero(ground_mask))}/{len(ground_mask)} points)",
        flush=True,
    )
    return output_path


def _ply_scalar_type(dtype: np.dtype) -> str:
    key = (dtype.kind, dtype.itemsize)
    mapping = {
        ("f", 4): "float",
        ("f", 8): "double",
        ("u", 1): "uchar",
        ("u", 2): "ushort",
        ("u", 4): "uint",
        ("i", 1): "char",
        ("i", 2): "short",
        ("i", 4): "int",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported PLY dtype: {dtype}")
    return mapping[key]


def _write_binary_ply_from_vertex_data(
    out_path: Path,
    *,
    vertex_dtype: np.dtype,
    vertex_count: int,
    body_files: list[Path],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {int(vertex_count)}",
        ]
        for name in vertex_dtype.names or ():
            header_lines.append(f"property {_ply_scalar_type(vertex_dtype[name])} {name}")
        header_lines.append("end_header")
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        for body_path in body_files:
            with body_path.open("rb") as src:
                shutil.copyfileobj(src, f, length=1024 * 1024 * 8)
    return out_path


def _iter_vertex_chunks(vertex_data: np.ndarray, chunk_size: int = _STREAM_CHUNK_SIZE):
    total = int(len(vertex_data))
    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        yield start, stop, vertex_data[start:stop]


def _reference_vertex_count(reference_ply_path: Path = _REFERENCE_PLY_PATH) -> int:
    reference_ply_path = Path(reference_ply_path).expanduser()
    if not reference_ply_path.is_file():
        return _FALLBACK_REFERENCE_VERTEX_COUNT
    ref_ply = PlyData.read(str(reference_ply_path), mmap=True)
    return int(ref_ply["vertex"].count)


def _choose_stream_grid_side(vertex_count: int, reference_count: int) -> int:
    if reference_count <= 0:
        return 1
    if float(vertex_count) <= float(reference_count) * _LARGE_PLY_RATIO_THRESHOLD:
        return 1
    side = 1
    while float(vertex_count) / float(side * side) > float(reference_count):
        side *= 2
    return side


def _scan_xy_bounds(vertex_data: np.ndarray) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for _start, _stop, chunk in _iter_vertex_chunks(vertex_data):
        x = np.asarray(chunk["x"], dtype=np.float64)
        y = np.asarray(chunk["y"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        x = x[finite]
        y = y[finite]
        min_x = min(min_x, float(np.min(x)))
        min_y = min(min_y, float(np.min(y)))
        max_x = max(max_x, float(np.max(x)))
        max_y = max(max_y, float(np.max(y)))
    if not np.isfinite(min_x) or not np.isfinite(min_y):
        raise ValueError("Failed to determine finite XY bounds from PLY")
    return min_x, min_y, max_x, max_y


def _split_large_ply_into_tiles(
    ply_path: Path,
    temp_dir: Path,
    *,
    grid_side: int,
) -> list[Path]:
    ply = PlyData.read(str(ply_path), mmap=True)
    vertex = ply["vertex"]
    vertex_data = vertex.data
    min_x, min_y, max_x, max_y = _scan_xy_bounds(vertex_data)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    tile_w = span_x / float(grid_side)
    tile_h = span_y / float(grid_side)

    raw_dir = temp_dir / "raw_tiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    body_paths = [raw_dir / f"tile_{idx:03d}.bin" for idx in range(grid_side * grid_side)]
    counts = [0 for _ in body_paths]

    handles = [path.open("wb") for path in body_paths]
    try:
        for _start, _stop, chunk in _iter_vertex_chunks(vertex_data):
            x = np.asarray(chunk["x"], dtype=np.float64)
            y = np.asarray(chunk["y"], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue
            chunk_finite = chunk[finite]
            x = x[finite]
            y = y[finite]
            col = np.floor((x - min_x) / tile_w).astype(np.int32)
            row = np.floor((y - min_y) / tile_h).astype(np.int32)
            np.clip(col, 0, grid_side - 1, out=col)
            np.clip(row, 0, grid_side - 1, out=row)
            tile_ids = row * grid_side + col
            for tile_id in np.unique(tile_ids):
                keep = tile_ids == tile_id
                tile_vertex = chunk_finite[keep]
                tile_vertex.tofile(handles[int(tile_id)])
                counts[int(tile_id)] += int(tile_vertex.shape[0])
    finally:
        for handle in handles:
            handle.close()

    tile_dir = temp_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []
    for tile_id, count in enumerate(counts):
        if count <= 0:
            continue
        row = tile_id // grid_side
        col = tile_id % grid_side
        out_path = tile_dir / f"tile_r{row:02d}_c{col:02d}.ply"
        _write_binary_ply_from_vertex_data(
            out_path,
            vertex_dtype=vertex_data.dtype,
            vertex_count=count,
            body_files=[body_paths[tile_id]],
        )
        tile_paths.append(out_path)
    return tile_paths


def _merge_filtered_tile_plys(filtered_tile_paths: list[Path], out_path: Path) -> Path:
    if not filtered_tile_paths:
        raise ValueError("No filtered tile PLYs to merge")

    dtypes: list[np.dtype] = []
    counts: list[int] = []
    for path in filtered_tile_paths:
        ply = PlyData.read(str(path), mmap=True)
        vertex = ply["vertex"]
        dtypes.append(vertex.data.dtype)
        counts.append(int(vertex.count))
    first_dtype = dtypes[0]
    if any(dt != first_dtype for dt in dtypes[1:]):
        raise ValueError("Filtered tile PLY dtypes do not match")

    raw_dir = out_path.parent / "_merge_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    body_paths: list[Path] = []
    for idx, path in enumerate(filtered_tile_paths):
        body_path = raw_dir / f"filtered_{idx:03d}.bin"
        ply = PlyData.read(str(path), mmap=True)
        ply["vertex"].data.tofile(str(body_path))
        body_paths.append(body_path)

    return _write_binary_ply_from_vertex_data(
        out_path,
        vertex_dtype=first_dtype,
        vertex_count=int(sum(counts)),
        body_files=body_paths,
    )


def _compute_canvas_from_metas(
    metas: list[dict[str, Any]],
    *,
    mpp: float,
) -> tuple[int, int, float, float]:
    global_min_x = min(float(meta["min_xy"][0]) for meta in metas)
    global_min_y = min(float(meta["min_xy"][1]) for meta in metas)
    global_max_x = max(float(meta["max_xy"][0]) for meta in metas)
    global_max_y = max(float(meta["max_xy"][1]) for meta in metas)
    canvas_w = max(int(np.ceil((global_max_x - global_min_x) / mpp)) + 1, 2)
    canvas_h = max(int(np.ceil((global_max_y - global_min_y) / mpp)) + 1, 2)
    return canvas_h, canvas_w, global_min_x, global_max_y


def _paste_tile_image(
    canvas: np.ndarray,
    tile_img: np.ndarray,
    tile_meta: dict[str, Any],
    *,
    global_min_x: float,
    global_max_y: float,
    mpp: float,
) -> None:
    tile_h, tile_w = tile_img.shape[:2]
    col_off = int(round((float(tile_meta["min_xy"][0]) - global_min_x) / mpp))
    row_off = int(round((global_max_y - float(tile_meta["max_xy"][1])) / mpp))
    canvas[row_off : row_off + tile_h, col_off : col_off + tile_w] = tile_img


def _merge_rendered_tiles(
    tile_results: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    mpp: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    metas = [meta for _img, meta in tile_results]
    _canvas_h, _canvas_w, global_min_x, global_max_y = _compute_canvas_from_metas(metas, mpp=mpp)
    max_row = 0
    max_col = 0
    for img, meta in tile_results:
        tile_h, tile_w = img.shape[:2]
        col_off = int(round((float(meta["min_xy"][0]) - global_min_x) / mpp))
        row_off = int(round((global_max_y - float(meta["max_xy"][1])) / mpp))
        max_row = max(max_row, row_off + tile_h)
        max_col = max(max_col, col_off + tile_w)
    canvas_h = max(_canvas_h, max_row)
    canvas_w = max(_canvas_w, max_col)
    sample = tile_results[0][0]
    if sample.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=sample.dtype)
    else:
        canvas = np.zeros((canvas_h, canvas_w, sample.shape[2]), dtype=sample.dtype)
    for img, meta in tile_results:
        _paste_tile_image(
            canvas,
            img,
            meta,
            global_min_x=global_min_x,
            global_max_y=global_max_y,
            mpp=mpp,
        )
    merged_meta = {
        "min_xy": [float(global_min_x), min(float(meta["min_xy"][1]) for meta in metas)],
        "max_xy": [max(float(meta["max_xy"][0]) for meta in metas), float(global_max_y)],
        "meters_per_pixel": float(mpp),
        "width": int(canvas_w),
        "height": int(canvas_h),
    }
    return canvas, merged_meta


def _merge_height_tiles(
    tile_results: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    mpp: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    metas = [meta for _values, meta in tile_results]
    _canvas_h, _canvas_w, global_min_x, global_max_y = _compute_canvas_from_metas(metas, mpp=mpp)
    max_row = 0
    max_col = 0
    offsets: list[tuple[int, int]] = []
    for values, meta in tile_results:
        tile_h, tile_w = values.shape[:2]
        col_off = int(round((float(meta["min_xy"][0]) - global_min_x) / mpp))
        row_off = int(round((global_max_y - float(meta["max_xy"][1])) / mpp))
        offsets.append((row_off, col_off))
        max_row = max(max_row, row_off + tile_h)
        max_col = max(max_col, col_off + tile_w)

    canvas_h = max(_canvas_h, max_row)
    canvas_w = max(_canvas_w, max_col)
    canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
    for (values, _meta), (row_off, col_off) in zip(tile_results, offsets):
        tile_h, tile_w = values.shape[:2]
        target = canvas[row_off : row_off + tile_h, col_off : col_off + tile_w]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        existing_valid = np.isfinite(target)
        update = valid & (~existing_valid | (values > target))
        target[update] = values[update]

    finite = np.isfinite(canvas)
    lo = float(np.nanmin(canvas)) if np.any(finite) else None
    hi = float(np.nanmax(canvas)) if np.any(finite) else None
    merged_meta = {
        "min_xy": [float(global_min_x), min(float(meta["min_xy"][1]) for meta in metas)],
        "max_xy": [max(float(meta["max_xy"][0]) for meta in metas), float(global_max_y)],
        "meters_per_pixel": float(mpp),
        "width": int(canvas_w),
        "height": int(canvas_h),
        "vis_height_lo": lo,
        "vis_height_hi": hi,
    }
    return canvas, merged_meta


def _save_height_outputs(
    *,
    output_dir: Path,
    outputs: dict[str, Path],
    height_values: np.ndarray,
    meta: dict[str, Any],
) -> None:
    height_img = _height_raster_to_rgb(
        height_values,
        lo=meta.get("vis_height_lo"),
        hi=meta.get("vis_height_hi"),
    )
    height_png_path = output_dir / "bev_pc_csf_height.png"
    Image.fromarray(height_img).save(str(height_png_path))
    height_values_path = output_dir / "bev_pc_csf_height_values.npy"
    np.save(height_values_path, np.asarray(height_values, dtype=np.float32))
    height_meta_path = output_dir / "bev_pc_csf_height_meta.json"
    height_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pre_part] saved CSF-filtered height PNG -> {height_png_path}")
    print(f"[pre_part] saved CSF-filtered height values -> {height_values_path}")
    print(f"[pre_part] saved CSF-filtered height meta -> {height_meta_path}")
    outputs["pc_csf_height_png"] = height_png_path
    outputs["pc_csf_height_values"] = height_values_path
    outputs["pc_csf_height_meta"] = height_meta_path


def _run_chunked_pre_part_renders(
    *,
    ply_path: Path,
    output_dir: Path,
    mode: str,
    mpp: float,
    requested_modes: list[str],
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    ply = PlyData.read(str(ply_path), mmap=True)
    vertex_count = int(ply["vertex"].count)
    reference_count = _reference_vertex_count()
    grid_side = _choose_stream_grid_side(vertex_count, reference_count)
    if grid_side <= 1:
        raise RuntimeError("Chunked pre-part called for non-chunked input")

    temp_dir = output_dir / "_chunked_pre_part"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[pre_part] large PLY detected ({vertex_count} points; ref={reference_count}). "
        f"Splitting into {grid_side}x{grid_side} XY tiles for CSF/BEV.",
        flush=True,
    )
    print("[pre_part] stage chunk-split: scanning XY bounds and writing tile PLYs ...", flush=True)
    tile_paths = _split_large_ply_into_tiles(ply_path, temp_dir, grid_side=grid_side)
    print(f"[pre_part] stage chunk-split: wrote {len(tile_paths)} non-empty tile PLY(s)", flush=True)

    filtered_tile_paths: list[Path] = []
    main_mode_results: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {m: [] for m in requested_modes}
    pc_csf_mode_results: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {m: [] for m in BASE_RENDER_MODES}
    height_results: list[tuple[np.ndarray, dict[str, Any]]] = []
    for idx, tile_path in enumerate(tile_paths, start=1):
        filtered_tile = temp_dir / "filtered_tiles" / f"{tile_path.stem}_csf.ply"
        filtered_tile.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pre_part] stage chunk-csf: {idx}/{len(tile_paths)} {tile_path.name}", flush=True)
        _write_csf_filtered_ply(tile_path, filtered_tile)
        filtered_tile_paths.append(filtered_tile)

        print(f"[pre_part] stage chunk-main-bev: {idx}/{len(tile_paths)} modes={requested_modes} mpp={mpp}", flush=True)
        tile_rendered = render_bev(
            filtered_tile,
            mode=requested_modes,
            mpp=mpp,
            skip_missing_fields=True,
            apply_csf=False,
        )
        assert isinstance(tile_rendered, dict)
        for rendered_mode in requested_modes:
            if rendered_mode in tile_rendered:
                main_mode_results[rendered_mode].append(tile_rendered[rendered_mode])

        print(
            f"[pre_part] stage chunk-pc-csf-bev: {idx}/{len(tile_paths)} modes={list(BASE_RENDER_MODES)} "
            f"mpp={PC_CSF_RENDER_MPP}",
            flush=True,
        )
        tile_pc_csf = render_bev(
            filtered_tile,
            mode=list(BASE_RENDER_MODES),
            mpp=PC_CSF_RENDER_MPP,
            skip_missing_fields=True,
            apply_csf=False,
        )
        assert isinstance(tile_pc_csf, dict)
        for rendered_mode in BASE_RENDER_MODES:
            if rendered_mode in tile_pc_csf:
                pc_csf_mode_results[rendered_mode].append(tile_pc_csf[rendered_mode])

        print(f"[pre_part] stage chunk-height: {idx}/{len(tile_paths)} mpp={PC_CSF_RENDER_MPP}", flush=True)
        height_values, height_meta = render_height_raster(
            filtered_tile,
            mpp=PC_CSF_RENDER_MPP,
            apply_csf=False,
        )
        height_results.append((height_values, height_meta))

    merged_outputs: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    print("[pre_part] stage merge-main-bev: merging rendered tile images ...", flush=True)
    for rendered_mode, results in main_mode_results.items():
        if results:
            merged_outputs[rendered_mode] = _merge_rendered_tiles(results, mpp=mpp)
    print("[pre_part] stage merge-pc-csf-bev: merging 0.02 rgb/intensity tile images ...", flush=True)
    for rendered_mode, results in pc_csf_mode_results.items():
        if results:
            merged_outputs[f"pc_csf_{rendered_mode}"] = _merge_rendered_tiles(results, mpp=PC_CSF_RENDER_MPP)
    if height_results:
        print("[pre_part] stage merge-height: merging 0.02 height rasters ...", flush=True)
        merged_outputs["pc_csf_height"] = _merge_height_tiles(height_results, mpp=PC_CSF_RENDER_MPP)

    print("[pre_part] stage merge-pc-csf-ply: merging filtered tile PLYs ...", flush=True)
    merged_pc_csf = _merge_filtered_tile_plys(filtered_tile_paths, output_dir / "pc_csf.ply")
    merged_outputs["pc_csf_ply_path"] = (np.empty((0, 0), dtype=np.uint8), {"path": str(merged_pc_csf)})
    return merged_outputs


def _save_rendered_mode(
    *,
    output_dir: Path,
    outputs: dict[str, Path],
    mode_name: str,
    img: np.ndarray,
    meta: dict,
    write_geotiff: bool = False,
) -> None:
    png_path = output_dir / f"bev_{mode_name}.png"
    Image.fromarray(img).save(str(png_path))
    print(f"[pre_part] saved PNG -> {png_path}  ({img.shape[1]}x{img.shape[0]} px)")

    outputs[f"{mode_name}_png"] = png_path
    if write_geotiff:
        tif_path = output_dir / f"bev_{mode_name}.tif"
        _png_to_geotiff(img, meta, tif_path)
        print(f"[pre_part] saved GeoTIFF -> {tif_path}")
        outputs[f"{mode_name}_geotiff"] = tif_path


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _is_valid_json_file(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _is_valid_npy_file(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        _ = arr.shape
    except (OSError, ValueError):
        return False
    return True


def _is_valid_image_file(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
    except (OSError, ValueError):
        return False
    return True


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _main_outputs_ready(output_dir: Path, *, mode: str) -> bool:
    required = [
        output_dir / "geo_meta.json",
        output_dir / "geo_meta_mpp-08.json",
        output_dir / f"bev_{mode}.png",
        output_dir / "bev_mask.png",
    ]
    if mode == "rgb":
        required.append(output_dir / "bev_intensity.png")
    elif mode == "intensity":
        required.append(output_dir / "bev_intensity.png")
    for path in required:
        if path.suffix == ".json":
            if not _is_valid_json_file(path):
                return False
        elif not _is_valid_image_file(path):
            return False
    return True


def _pc_csf_bev_outputs_ready(pc_csf_bev_dir: Path) -> bool:
    required = [
        pc_csf_bev_dir / "bev_pc_csf_rgb.png",
        pc_csf_bev_dir / "bev_pc_csf_rgb.tif",
        pc_csf_bev_dir / "bev_pc_csf_intensity.png",
        pc_csf_bev_dir / "bev_pc_csf_intensity.tif",
        pc_csf_bev_dir / "pc_csf_geo_meta.json",
    ]
    for path in required:
        if path.suffix == ".json":
            if not _is_valid_json_file(path):
                return False
        elif not _is_nonempty_file(path):
            return False
    return True


def _height_outputs_ready(pc_csf_bev_dir: Path) -> bool:
    return (
        _is_valid_image_file(pc_csf_bev_dir / "bev_pc_csf_height.png")
        and _is_valid_npy_file(pc_csf_bev_dir / "bev_pc_csf_height_values.npy")
        and _is_valid_json_file(pc_csf_bev_dir / "bev_pc_csf_height_meta.json")
    )


def _parts_outputs_ready(output_dir: Path) -> bool:
    return _is_valid_json_file(output_dir / "parts.json") and _is_valid_image_file(output_dir / "parts_preview.png")


def _collect_existing_outputs(output_dir: Path, outputs: dict[str, Path]) -> None:
    pc_csf_bev_dir = output_dir / "bev_pc_csf"
    candidates = {
        "geo_meta": output_dir / "geo_meta.json",
        "geo_meta_mpp-08": output_dir / "geo_meta_mpp-08.json",
        "pc_csf_ply": output_dir / "pc_csf.ply",
        "rgb_png": output_dir / "bev_rgb.png",
        "intensity_png": output_dir / "bev_intensity.png",
        "mask_png": output_dir / "bev_mask.png",
        "pc_csf_rgb_png": pc_csf_bev_dir / "bev_pc_csf_rgb.png",
        "pc_csf_rgb_geotiff": pc_csf_bev_dir / "bev_pc_csf_rgb.tif",
        "pc_csf_intensity_png": pc_csf_bev_dir / "bev_pc_csf_intensity.png",
        "pc_csf_intensity_geotiff": pc_csf_bev_dir / "bev_pc_csf_intensity.tif",
        "pc_csf_geo_meta": pc_csf_bev_dir / "pc_csf_geo_meta.json",
        "pc_csf_height_png": pc_csf_bev_dir / "bev_pc_csf_height.png",
        "pc_csf_height_values": pc_csf_bev_dir / "bev_pc_csf_height_values.npy",
        "pc_csf_height_meta": pc_csf_bev_dir / "bev_pc_csf_height_meta.json",
        "geo_tile": pc_csf_bev_dir / "geo_tile.json",
        "parts_json": output_dir / "parts.json",
        "parts_preview_png": output_dir / "parts_preview.png",
        "pc_csf_rgb_filled_png": pc_csf_bev_dir / "bev_pc_csf_rgb_filled.png",
    }
    for key, path in candidates.items():
        if _is_nonempty_file(path):
            outputs[key] = path


def _remove_stale_layout_outputs(output_dir: Path) -> None:
    stale_names = [
        "bev_rgb.png",
        "bev_rgb.tif",
        "bev_pc_csf_rgb.png",
        "bev_pc_csf_rgb.tif",
        "bev_pc_csf_rgb_filled.png",
        "bev_pc_csf_intensity.png",
        "bev_pc_csf_intensity.tif",
        "bev_pc_csf_height.png",
        "bev_pc_csf_height_values.npy",
        "bev_pc_csf_height_meta.json",
        "pc_csf_geo_meta.json",
        "geo_tile.json",
        "bev_rgb_mpp-02.png",
        "bev_rgb_mpp-02.tif",
        "bev_intensity_mpp-02.png",
        "bev_intensity_mpp-02.tif",
        "geo_meta_mpp-02.json",
    ]
    for name in stale_names:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _cleanup_transient_pre_part_dirs(output_dir: Path) -> None:
    for name in ("_chunked_pre_part", "_merge_raw"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[pre_part] cleaned transient directory -> {path}", flush=True)
        elif path.exists():
            path.unlink()
            print(f"[pre_part] cleaned transient file -> {path}", flush=True)


def fill_image_holes_with_local_mean(
    img: np.ndarray,
    *,
    radius_px: int = 10,
) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {arr.shape}")
    radius_px = int(radius_px)
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    valid_mask = np.any(arr > 0, axis=2)
    valid = valid_mask.astype(np.float32)
    holes = ~valid_mask
    if not np.any(holes):
        return np.asarray(arr, dtype=np.uint8).copy()

    kernel_size = radius_px * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius_px, radius_px), radius_px, 1.0, thickness=-1)
    filled = np.asarray(arr, dtype=np.float32).copy()
    from scipy import ndimage

    distance, nearest_idx = ndimage.distance_transform_edt(
        holes,
        return_distances=True,
        return_indices=True,
    )
    fillable = holes & (distance <= float(radius_px))
    if not np.any(fillable):
        return np.asarray(arr, dtype=np.uint8).copy()

    valid_count = cv2.filter2D(valid, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    channel_means: list[np.ndarray] = []
    for ch in range(3):
        channel = np.asarray(arr[:, :, ch], dtype=np.float32)
        channel_sum = cv2.filter2D(channel * valid, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        channel_mean = np.divide(
            channel_sum,
            valid_count,
            out=np.zeros_like(channel_sum, dtype=np.float32),
            where=valid_count > 0,
        )
        channel_means.append(channel_mean)

    target_rows, target_cols = np.where(fillable)
    nearest_rows = nearest_idx[0][target_rows, target_cols]
    nearest_cols = nearest_idx[1][target_rows, target_cols]
    nearest_pairs = np.stack([nearest_rows, nearest_cols], axis=1)
    unique_pairs, inverse = np.unique(nearest_pairs, axis=0, return_inverse=True)

    nearest_mean = np.zeros((len(unique_pairs), 3), dtype=np.float32)
    for idx, (src_row, src_col) in enumerate(unique_pairs):
        if valid_count[src_row, src_col] <= 0:
            nearest_mean[idx] = arr[src_row, src_col]
            continue
        nearest_mean[idx] = np.array(
            [channel_means[ch][src_row, src_col] for ch in range(3)],
            dtype=np.float32,
        )

    filled[target_rows, target_cols] = nearest_mean[inverse]
    return np.clip(filled, 0, 255).astype(np.uint8)


def fill_image_holes_file(
    image_path: Path | str,
    output_path: Path | str,
    *,
    radius_px: int = 10,
) -> Path:
    image_path = Path(image_path).expanduser()
    output_path = Path(output_path).expanduser()
    img = np.asarray(Image.open(image_path).convert("RGB"))
    filled = fill_image_holes_with_local_mean(img, radius_px=radius_px)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(filled).save(output_path)
    print(f"[pre_part] saved filled image -> {output_path}  (radius={radius_px}px)")
    return output_path


def render_pc_csf_rgb_at_mpp(
    pc_csf_ply_path: Path | str,
    output_dir: Path | str,
    *,
    mpp: float,
    stem_suffix: str | None = None,
) -> dict[str, Path]:
    pc_csf_ply_path = Path(pc_csf_ply_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem_suffix is None:
        stem_suffix = f"mpp-{int(round(float(mpp) * 100)):02d}"

    print(
        f"[pre_part] rendering CSF-filtered rgb BEV (mpp={mpp}) from {pc_csf_ply_path.name} ...",
        flush=True,
    )
    result = render_bev(
        pc_csf_ply_path,
        mode="rgb",
        mpp=mpp,
        apply_csf=False,
    )
    assert isinstance(result, tuple), "Expected single-mode render result"
    img, meta = result

    outputs: dict[str, Path] = {}
    _save_rendered_mode(
        output_dir=output_dir,
        outputs=outputs,
        mode_name=f"pc_csf_rgb_{stem_suffix}",
        img=img,
        meta=meta,
    )
    meta_path = output_dir / f"pc_csf_geo_meta_{stem_suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pre_part] saved CSF-filtered geo_meta -> {meta_path}")
    outputs[f"pc_csf_rgb_{stem_suffix}_meta"] = meta_path
    return outputs


def _png_to_geotiff(
    img: np.ndarray,
    meta: dict,
    tif_path: Path,
) -> None:
    """Write a 3-band GeoTIFF from an RGB array and BEV meta dict."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(
        float(meta["min_xy"][0]),
        float(meta["max_xy"][1]),
        float(meta["meters_per_pixel"]),
        float(meta["meters_per_pixel"]),
    )

    channels = img.shape[2] if img.ndim == 3 else 1
    profile = {
        "driver": "GTiff",
        "height": img.shape[0],
        "width": img.shape[1],
        "count": channels,
        "dtype": img.dtype.name,
        "transform": transform,
    }
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **profile) as ds:
        if channels == 1:
            ds.write(img, 1)
        else:
            for b in range(channels):
                ds.write(img[:, :, b], b + 1)


def run_pre_part(
    ply_path: str | Path,
    output_dir: str | Path = "outputs/pre_part",
    *,
    mpp: float = 0.08,
    mode: str = "rgb",
    tile_size_m: float = _DEFAULT_TILE_SIZE_M,
    fill_ratio_threshold: float = _DEFAULT_FILL_RATIO_THRESHOLD,
    fill_cell_size_m: float = _DEFAULT_FILL_CELL_SIZE_M,
    resume: bool = True,
) -> dict[str, Path]:
    """Render whole-PLY pre-part BEV products and write geo_meta.json files."""
    ply_path = Path(ply_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pc_csf_bev_dir = output_dir / "bev_pc_csf"
    pc_csf_bev_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        _remove_stale_layout_outputs(output_dir)

    source_ply = PlyData.read(str(ply_path), mmap=True)
    source_vertex_count = int(source_ply["vertex"].count)
    reference_count = _reference_vertex_count()
    use_chunked = _choose_stream_grid_side(source_vertex_count, reference_count) > 1
    print(
        f"[pre_part] stage init: source_points={source_vertex_count} "
        f"reference_points={reference_count} chunked={use_chunked}",
        flush=True,
    )

    modes = list(dict.fromkeys([mode, *BASE_RENDER_MODES]))
    requested_modes = modes

    pc_csf_path = output_dir / "pc_csf.ply"
    main_ready = resume and _main_outputs_ready(output_dir, mode=mode)
    pc_csf_bev_ready = resume and _pc_csf_bev_outputs_ready(pc_csf_bev_dir)
    height_ready = resume and _height_outputs_ready(pc_csf_bev_dir)

    if resume and _is_nonempty_file(pc_csf_path):
        print(f"[pre_part] resume: using existing CSF-filtered PLY -> {pc_csf_path}", flush=True)
        rendered: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        pc_csf_result: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        pc_csf_height_result: tuple[np.ndarray, dict[str, Any]] | None = None
        if main_ready:
            print("[pre_part] resume: main BEV outputs complete, skipping main render", flush=True)
        else:
            print(f"[pre_part] resume: rendering missing main BEV outputs ({', '.join(requested_modes)}, mpp={mpp}) ...", flush=True)
            result = render_bev(
                pc_csf_path,
                mode=requested_modes,
                mpp=mpp,
                skip_missing_fields=True,
                apply_csf=False,
            )
            assert isinstance(result, dict), "Expected multi-mode render result"
            rendered = result
        if pc_csf_bev_ready:
            print("[pre_part] resume: 0.02 CSF BEV outputs complete, skipping CSF BEV render", flush=True)
        else:
            print(
                f"[pre_part] resume: rendering missing 0.02 CSF BEV outputs ({', '.join(BASE_RENDER_MODES)}) ...",
                flush=True,
            )
            pc_csf_render = render_bev(
                pc_csf_path,
                mode=list(BASE_RENDER_MODES),
                mpp=PC_CSF_RENDER_MPP,
                skip_missing_fields=True,
                apply_csf=False,
            )
            assert isinstance(pc_csf_render, dict), "Expected multi-mode render result for CSF-filtered PLY"
            pc_csf_result = pc_csf_render
        if height_ready:
            print("[pre_part] resume: height outputs complete, skipping height render", flush=True)
        else:
            print(f"[pre_part] resume: rendering missing height raster (mpp={PC_CSF_RENDER_MPP}) ...", flush=True)
            pc_csf_height_result = render_height_raster(
                pc_csf_path,
                mpp=PC_CSF_RENDER_MPP,
                apply_csf=False,
            )
    elif use_chunked:
        print("[pre_part] stage chunked-prepart: start", flush=True)
        chunked = _run_chunked_pre_part_renders(
            ply_path=ply_path,
            output_dir=output_dir,
            mode=mode,
            mpp=mpp,
            requested_modes=requested_modes,
        )
        rendered = {
            rendered_mode: chunked[rendered_mode]
            for rendered_mode in requested_modes
            if rendered_mode in chunked
        }
        pc_csf_result = {
            rendered_mode: chunked[f"pc_csf_{rendered_mode}"]
            for rendered_mode in BASE_RENDER_MODES
            if f"pc_csf_{rendered_mode}" in chunked
        }
        pc_csf_height_result = chunked.get("pc_csf_height")
    else:
        print("[pre_part] stage csf: filtering full PLY ...", flush=True)
        pc_csf_path = _write_csf_filtered_ply(ply_path, pc_csf_path)
        print(f"[pre_part] stage main-bev: rendering ({', '.join(requested_modes)}, mpp={mpp}) from {pc_csf_path.name} ...")
        result = render_bev(
            pc_csf_path,
            mode=requested_modes,
            mpp=mpp,
            skip_missing_fields=True,
            apply_csf=False,
        )
        assert isinstance(result, dict), "Expected multi-mode render result"
        rendered = result
        print(
            f"[pre_part] stage pc-csf-bev: rendering ({', '.join(BASE_RENDER_MODES)}, mpp={PC_CSF_RENDER_MPP}) from {pc_csf_path.name} ...",
            flush=True,
        )
        pc_csf_result = render_bev(
            pc_csf_path,
            mode=list(BASE_RENDER_MODES),
            mpp=PC_CSF_RENDER_MPP,
            skip_missing_fields=True,
            apply_csf=False,
        )
        assert isinstance(pc_csf_result, dict), "Expected multi-mode render result for CSF-filtered PLY"
        print(f"[pre_part] stage height: rendering height raster (mpp={PC_CSF_RENDER_MPP}) from {pc_csf_path.name} ...", flush=True)
        pc_csf_height_result = render_height_raster(
            pc_csf_path,
            mpp=PC_CSF_RENDER_MPP,
            apply_csf=False,
        )

    if not main_ready and mode not in rendered:
        requirement = _mode_requirement(mode)
        raise KeyError(
            f"Primary mode '{mode}' is unavailable because required field "
            f"{requirement!r} was not found in the PLY."
        )

    geo_meta_path = output_dir / "geo_meta.json"
    geo_meta_008_path = output_dir / "geo_meta_mpp-08.json"
    outputs: dict[str, Path] = {
        "pc_csf_ply": pc_csf_path,
    }
    _collect_existing_outputs(output_dir, outputs)

    if main_ready:
        outputs["geo_meta"] = geo_meta_path
        outputs["geo_meta_mpp-08"] = geo_meta_008_path
    else:
        skipped = [m for m in requested_modes if m not in rendered]
        for skipped_mode in skipped:
            requirement = _mode_requirement(skipped_mode)
            print(
                f"[pre_part] skip {skipped_mode}: required field {requirement!r} "
                f"not found in {pc_csf_path.name}"
            )

        meta = rendered[mode][1]
        print("[pre_part] stage save-main: writing main BEV outputs ...", flush=True)
        geo_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[pre_part] saved geo_meta -> {geo_meta_path}")
        geo_meta_008_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[pre_part] saved 0.08 geo_meta -> {geo_meta_008_path}")
        outputs["geo_meta"] = geo_meta_path
        outputs["geo_meta_mpp-08"] = geo_meta_008_path

        for rendered_mode in ("rgb", "intensity"):
            if rendered_mode not in rendered:
                continue
            img, rendered_meta = rendered[rendered_mode]
            _save_rendered_mode(
                output_dir=output_dir,
                outputs=outputs,
                mode_name=rendered_mode,
                img=img,
                meta=rendered_meta,
                write_geotiff=False,
            )

        mask_source_mode = "intensity" if "intensity" in rendered else mode
        if mask_source_mode in rendered:
            mask_img = _build_mask_from_render(rendered[mask_source_mode][0])
            _save_rendered_mode(
                output_dir=output_dir,
                outputs=outputs,
                mode_name="mask",
                img=mask_img,
                meta=rendered[mask_source_mode][1],
                write_geotiff=False,
            )

    csf_meta: dict[str, Any] | None = None
    if pc_csf_bev_ready:
        _collect_existing_outputs(output_dir, outputs)
    else:
        print("[pre_part] stage save-pc-csf-bev: writing 0.02 CSF BEV outputs ...", flush=True)
        for csf_mode in BASE_RENDER_MODES:
            if csf_mode not in pc_csf_result:
                continue
            csf_img, csf_meta_item = pc_csf_result[csf_mode]
            _save_rendered_mode(
                output_dir=pc_csf_bev_dir,
                outputs=outputs,
                mode_name=f"pc_csf_{csf_mode}",
                img=csf_img,
                meta=csf_meta_item,
                write_geotiff=True,
            )
            csf_meta = csf_meta_item
        if csf_meta is not None:
            pc_csf_geo_meta_path = pc_csf_bev_dir / "pc_csf_geo_meta.json"
            pc_csf_geo_meta_path.write_text(
                json.dumps(csf_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[pre_part] saved CSF-filtered geo_meta -> {pc_csf_geo_meta_path}")
            outputs["pc_csf_geo_meta"] = pc_csf_geo_meta_path

    if height_ready:
        _collect_existing_outputs(output_dir, outputs)
    elif pc_csf_height_result is not None:
        print("[pre_part] stage save-height: writing height PNG/NPY/meta ...", flush=True)
        height_values, height_meta = pc_csf_height_result
        _save_height_outputs(
            output_dir=pc_csf_bev_dir,
            outputs=outputs,
            height_values=height_values,
            meta=height_meta,
        )

    geo_tile_path = pc_csf_bev_dir / "geo_tile.json"
    geo_tile_source = outputs.get("pc_csf_rgb_geotiff")
    if resume and _is_valid_json_file(geo_tile_path):
        print("[pre_part] resume: geo_tile.json complete, skipping geo-tile", flush=True)
        outputs["geo_tile"] = geo_tile_path
    elif geo_tile_source is not None:
        print("[pre_part] stage geo-tile: writing geo_tile.json ...", flush=True)
        geo_tile_payload = build_default_geo_tile_payload(geo_tile_source)
        geo_tile_path.write_text(
            json.dumps(geo_tile_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[pre_part] saved default geo tile -> {geo_tile_path}")
        outputs["geo_tile"] = geo_tile_path

    parts_json_path = output_dir / "parts.json"
    parts_preview_path = output_dir / "parts_preview.png"
    if resume and _parts_outputs_ready(output_dir):
        print("[pre_part] resume: parts.json and preview complete, skipping tile-part", flush=True)
        outputs["parts_json"] = parts_json_path
        outputs["parts_preview_png"] = parts_preview_path
    else:
        print("[pre_part] stage tile-part: generating parts.json and preview ...", flush=True)
        tile_payload = write_tile_parts_json(
            ply_path,
            parts_json_path,
            tile_size_m=tile_size_m,
            fill_ratio_threshold=fill_ratio_threshold,
            fill_cell_size_m=fill_cell_size_m,
            mask_bev_path=output_dir / "bev_mask.png",
            geo_meta_path=geo_meta_path,
            preview_path=parts_preview_path,
        )
        print(f"[pre_part] saved tile parts -> {parts_json_path}  ({tile_payload['num_parts']} parts)")
        outputs["parts_json"] = parts_json_path
        outputs["parts_preview_png"] = parts_preview_path

    pc_csf_rgb_png = outputs.get("pc_csf_rgb_png")
    filled_path = pc_csf_bev_dir / "bev_pc_csf_rgb_filled.png"
    if resume and _is_valid_image_file(filled_path):
        print("[pre_part] resume: filled pc_csf RGB complete, skipping fill-rgb", flush=True)
        outputs["pc_csf_rgb_filled_png"] = filled_path
    elif pc_csf_rgb_png is not None:
        print("[pre_part] stage fill-rgb: filling pc_csf RGB holes ...", flush=True)
        fill_image_holes_file(pc_csf_rgb_png, filled_path, radius_px=10)
        outputs["pc_csf_rgb_filled_png"] = filled_path

    _collect_existing_outputs(output_dir, outputs)
    _cleanup_transient_pre_part_dirs(output_dir)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render whole-PLY pre-part BEV products.",
    )
    parser.add_argument("ply_path", help="Input PLY file path.")
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/pre_part",
        help="Output directory (default: outputs/pre_part).",
    )
    parser.add_argument(
        "--mpp", type=float, default=0.08,
        help="Metres per pixel (default: 0.08).",
    )
    parser.add_argument(
        "--mode", default="rgb",
        choices=["rgb", "intensity"],
        help=(
            "Primary BEV rendering mode (default: rgb). "
            "pre_part always writes rgb/intensity/mask at the requested mpp."
        ),
    )
    parser.add_argument(
        "--tile-size", type=float, default=_DEFAULT_TILE_SIZE_M,
        help="Tile size in metres for the tile-part step (default: 40).",
    )
    parser.add_argument(
        "--fill-threshold", type=float, default=_DEFAULT_FILL_RATIO_THRESHOLD,
        help="Minimum fill ratio for the tile-part step (default: 0.10).",
    )
    parser.add_argument(
        "--fill-cell-size", type=float, default=_DEFAULT_FILL_CELL_SIZE_M,
        help="Fill-cell size metadata for the tile-part step (default: 0.50).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable stage-level resume and recompute pre-part outputs.",
    )
    args = parser.parse_args()
    run_pre_part(
        args.ply_path,
        args.output_dir,
        mpp=args.mpp,
        mode=args.mode,
        tile_size_m=args.tile_size,
        fill_ratio_threshold=args.fill_threshold,
        fill_cell_size_m=args.fill_cell_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
