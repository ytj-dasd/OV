from __future__ import annotations

import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageTk
from rasterio.transform import from_origin

Image.MAX_IMAGE_PIXELS = None


def export_geotiff_from_png(png_path: Path, geotiff_path: Path, bev_meta_path: Path) -> None:
    bev_meta = json.loads(bev_meta_path.read_text(encoding="utf-8"))
    image = Image.open(png_path)
    array = np.array(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected an RGB/RGBA image, got shape {array.shape!r}")

    transform = from_origin(
        float(bev_meta["min_xy"][0]),
        float(bev_meta["max_xy"][1]),
        float(bev_meta["meters_per_pixel"]),
        float(bev_meta["meters_per_pixel"]),
    )

    channels = array.shape[2]
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": channels,
        "dtype": array.dtype,
        "transform": transform,
    }

    with rasterio.open(geotiff_path, "w", **profile) as dataset:
        for band_idx in range(channels):
            dataset.write(array[:, :, band_idx], band_idx + 1)


@dataclass(frozen=True)
class TileCrop:
    row: int
    col: int
    minxy: tuple[int, int]
    maxxy: tuple[int, int]
    is_export: bool
    image: Image.Image


def _normalize_size(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {value!r}")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return (width, height)


def _normalize_overlap(
    overlap: float | tuple[float, float] | tuple[int, int],
    tile_size: tuple[int, int],
) -> tuple[int, int]:
    if isinstance(overlap, (int, float)):
        values = (float(overlap), float(overlap))
    else:
        if len(overlap) != 2:
            raise ValueError(f"overlap must have length 2, got {overlap!r}")
        values = (float(overlap[0]), float(overlap[1]))

    overlap_pixels: list[int] = []
    for idx, value in enumerate(values):
        if value < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap!r}")
        if value < 1.0:
            overlap_pixels.append(int(round(tile_size[idx] * value)))
        else:
            overlap_pixels.append(int(round(value)))

    overlap_width, overlap_height = overlap_pixels
    if overlap_width >= tile_size[0] or overlap_height >= tile_size[1]:
        raise ValueError(
            "overlap must be smaller than tile_size; "
            f"got overlap_pixels={tuple(overlap_pixels)!r}, tile_size={tile_size!r}"
        )
    return (overlap_width, overlap_height)


def _open_image(image_or_path: str | Path | Image.Image) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.copy()
    image = Image.open(image_or_path)
    image.load()
    return image


def _build_tile_crops(
    image: Image.Image,
    origin: tuple[int, int],
    tile_size: tuple[int, int],
    stride: tuple[int, int],
    include_edge_tiles: bool,
    is_export_map: dict[tuple[int, int], bool] | None = None,
) -> list[TileCrop]:
    width, height = image.size
    tile_width, tile_height = tile_size
    stride_width, stride_height = stride
    origin_x, origin_y = origin

    min_col = math.floor((0 - origin_x) / stride_width)
    max_col = math.ceil((width - origin_x) / stride_width)
    min_row = math.floor((0 - origin_y) / stride_height)
    max_row = math.ceil((height - origin_y) / stride_height)

    crops: list[TileCrop] = []
    for row in range(min_row, max_row + 1):
        top = origin_y + row * stride_height
        bottom = top + tile_height
        for col in range(min_col, max_col + 1):
            left = origin_x + col * stride_width
            right = left + tile_width

            if include_edge_tiles:
                crop_left = max(0, left)
                crop_top = max(0, top)
                crop_right = min(width, right)
                crop_bottom = min(height, bottom)
                if crop_left >= crop_right or crop_top >= crop_bottom:
                    continue
            else:
                if left < 0 or top < 0 or right > width or bottom > height:
                    continue
                crop_left, crop_top, crop_right, crop_bottom = left, top, right, bottom

            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            crops.append(
                TileCrop(
                    row=row,
                    col=col,
                    minxy=(left, top),
                    maxxy=(right, bottom),
                    is_export=True if is_export_map is None else bool(is_export_map.get((row, col), True)),
                    image=crop,
                )
            )

    crops.sort(key=lambda item: (item.row, item.col))
    return crops


def _build_default_export_map(
    image: Image.Image,
    origin: tuple[int, int],
    tile_size: tuple[int, int],
    stride: tuple[int, int],
    include_edge_tiles: bool,
) -> dict[tuple[int, int], bool]:
    """Compute default export flags for a fixed grid origin.

    This mirrors the interactive selector's initial lock behaviour, which
    marks mostly-black tiles as non-export by default.
    """
    image_width, image_height = image.size
    origin_x, origin_y = origin
    tile_width, tile_height = tile_size
    stride_width, stride_height = stride
    min_col = math.floor((0 - origin_x) / stride_width)
    max_col = math.ceil((image_width - origin_x) / stride_width)
    min_row = math.floor((0 - origin_y) / stride_height)
    max_row = math.ceil((image_height - origin_y) / stride_height)

    attrs: dict[tuple[int, int], bool] = {}
    for row in range(min_row, max_row + 1):
        top = origin_y + row * stride_height
        bottom = top + tile_height
        for col in range(min_col, max_col + 1):
            left = origin_x + col * stride_width
            right = left + tile_width

            if not include_edge_tiles and (
                left < 0 or top < 0 or right > image_width or bottom > image_height
            ):
                continue

            crop_left = max(0, left)
            crop_top = max(0, top)
            crop_right = min(image_width, right)
            crop_bottom = min(image_height, bottom)
            if crop_left >= crop_right or crop_top >= crop_bottom:
                continue

            tile_array = np.asarray(
                image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
            )
            black_ratio = float(np.mean(np.all(tile_array == 0, axis=2)))
            attrs[(row, col)] = black_ratio < 0.95

    return attrs


def default_grid_tile_selector(
    image_or_path: str | Path | Image.Image,
    tile_size: int | tuple[int, int] = (2048, 2048),
    overlap: float | tuple[float, float] | tuple[int, int] = 0.1,
    include_edge_tiles: bool = True,
) -> list[TileCrop]:
    """Return the default non-interactive tile selection.

    Equivalent to opening the interactive selector, keeping the default origin
    at (0, 0), locking once, and accepting the automatically filtered tiles.
    """
    image = _open_image(image_or_path)
    image = image.convert("RGB")
    tile_size = _normalize_size(tile_size, "tile_size")
    overlap_pixels = _normalize_overlap(overlap, tile_size)
    stride = (tile_size[0] - overlap_pixels[0], tile_size[1] - overlap_pixels[1])

    if stride[0] <= 0 or stride[1] <= 0:
        raise ValueError(f"stride must be positive, got {stride!r}")

    origin = (0, 0)
    is_export_map = _build_default_export_map(
        image=image,
        origin=origin,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
    )
    return _build_tile_crops(
        image=image,
        origin=origin,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
        is_export_map=is_export_map,
    )


def interactive_grid_tile_selector(
    image_or_path: str | Path | Image.Image,
    tile_size: int | tuple[int, int] = (2048, 2048),
    overlap: float | tuple[float, float] | tuple[int, int] = 0.1,
    canvas_size: tuple[int, int] = (1920, 1080),
    include_edge_tiles: bool = True,
) -> list[TileCrop]:
    image = _open_image(image_or_path)
    image = image.convert("RGB")
    tile_size = _normalize_size(tile_size, "tile_size")
    canvas_size = _normalize_size(canvas_size, "canvas_size")
    overlap_pixels = _normalize_overlap(overlap, tile_size)
    stride = (tile_size[0] - overlap_pixels[0], tile_size[1] - overlap_pixels[1])

    if stride[0] <= 0 or stride[1] <= 0:
        raise ValueError(f"stride must be positive, got {stride!r}")

    image_width, image_height = image.size
    canvas_width, canvas_height = canvas_size
    scale = min(canvas_width / image_width, canvas_height / image_height)
    display_width = max(1, int(round(image_width * scale)))
    display_height = max(1, int(round(image_height * scale)))
    display_image = image.resize((display_width, display_height), Image.Resampling.BILINEAR)
    offset_x = (canvas_width - display_width) // 2
    offset_y = (canvas_height - display_height) // 2

    root = tk.Tk()
    root.title("Interactive Grid Tile Selector")
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="black", highlightthickness=0)
    canvas.pack()

    tk_image = ImageTk.PhotoImage(display_image)
    canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_image)

    state: dict[str, Any] = {
        "locked": False,
        "current_origin": (0.0, 0.0),
        "tile_attrs": {},
        "grid_bounds": None,
        "result": None,
    }

    guide_width = stride[0] * scale
    guide_height = stride[1] * scale
    text_id = canvas.create_text(
        12,
        12,
        anchor="nw",
        fill="yellow",
        text="Move mouse to position tile grid. Left click to lock, right click to unlock, Enter to confirm.",
    )

    def event_to_origin(event_x: float, event_y: float) -> tuple[float, float]:
        image_x = (event_x - offset_x) / scale
        image_y = (event_y - offset_y) / scale
        image_x = min(max(image_x, 0.0), float(image_width - 1))
        image_y = min(max(image_y, 0.0), float(image_height - 1))
        return (image_x, image_y)

    def finalize_origin(origin: tuple[float, float]) -> tuple[int, int]:
        max_x = max(0, image_width - tile_size[0])
        max_y = max(0, image_height - tile_size[1])
        origin_x = int(math.floor(origin[0] / stride[0]) * stride[0])
        origin_y = int(math.floor(origin[1] / stride[1]) * stride[1])
        return (min(max(origin_x, 0), max_x), min(max(origin_y, 0), max_y))

    def event_to_cell(event_x: float, event_y: float, origin: tuple[float, float]) -> tuple[int, int] | None:
        if not (offset_x <= event_x <= offset_x + display_width and offset_y <= event_y <= offset_y + display_height):
            return None
        image_x = (event_x - offset_x) / scale
        image_y = (event_y - offset_y) / scale
        col = int(math.floor((image_x - origin[0]) / stride[0]))
        row = int(math.floor((image_y - origin[1]) / stride[1]))
        return (row, col)

    def draw_cell_fill(row: int, col: int, origin: tuple[float, float]) -> None:
        left = origin[0] + col * stride[0]
        right = left + stride[0]
        top = origin[1] + row * stride[1]
        bottom = top + stride[1]

        rect_left = offset_x + left * scale
        rect_top = offset_y + top * scale
        rect_right = offset_x + right * scale
        rect_bottom = offset_y + bottom * scale

        rect_left = max(rect_left, offset_x)
        rect_top = max(rect_top, offset_y)
        rect_right = min(rect_right, offset_x + display_width)
        rect_bottom = min(rect_bottom, offset_y + display_height)
        if rect_left >= rect_right or rect_top >= rect_bottom:
            return

        canvas.create_rectangle(
            rect_left,
            rect_top,
            rect_right,
            rect_bottom,
            fill="#ff0000",
            stipple="gray50",
            outline="",
            tags=("grid_fill", f"grid_fill_{row}_{col}"),
        )

    def build_locked_tile_attrs(origin: tuple[float, float]) -> dict[tuple[int, int], dict[str, Any]]:
        origin_x, origin_y = finalize_origin(origin)
        tile_width, tile_height = tile_size
        stride_width, stride_height = stride
        min_col = math.floor((0 - origin_x) / stride_width)
        max_col = math.ceil((image_width - origin_x) / stride_width)
        min_row = math.floor((0 - origin_y) / stride_height)
        max_row = math.ceil((image_height - origin_y) / stride_height)

        attrs: dict[tuple[int, int], dict[str, Any]] = {}
        for row in range(min_row, max_row + 1):
            top = origin_y + row * stride_height
            bottom = top + tile_height
            for col in range(min_col, max_col + 1):
                left = origin_x + col * stride_width
                right = left + tile_width

                if not include_edge_tiles and (left < 0 or top < 0 or right > image_width or bottom > image_height):
                    continue

                crop_left = max(0, left)
                crop_top = max(0, top)
                crop_right = min(image_width, right)
                crop_bottom = min(image_height, bottom)
                if crop_left >= crop_right or crop_top >= crop_bottom:
                    continue

                tile_array = np.asarray(image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB"))
                black_ratio = float(np.mean(np.all(tile_array == 0, axis=2)))
                attrs[(row, col)] = {
                    "minxy": (left, top),
                    "maxxy": (right, bottom),
                    "is_export": black_ratio < 0.95,
                }
        return attrs

    def redraw(origin: tuple[float, float]) -> None:
        canvas.delete("grid_line")
        canvas.delete("grid_fill")

        min_col = math.floor((0.0 - origin[0]) / stride[0]) - 1
        max_col = math.ceil((image_width - origin[0]) / stride[0]) + 1
        min_row = math.floor((0.0 - origin[1]) / stride[1]) - 1
        max_row = math.ceil((image_height - origin[1]) / stride[1]) + 1
        state["grid_bounds"] = (min_row, max_row, min_col, max_col)

        if state["locked"]:
            for row, col in state["tile_attrs"]:
                if not state["tile_attrs"][(row, col)]["is_export"]:
                    draw_cell_fill(row, col, origin)

        for col in range(min_col, max_col + 1):
            x = offset_x + (origin[0] + col * stride[0]) * scale
            if x < offset_x - 1 or x > offset_x + display_width + 1:
                continue
            canvas.create_line(
                x,
                offset_y,
                x,
                offset_y + display_height,
                fill="yellow",
                width=1,
                tags="grid_line",
            )

        for row in range(min_row, max_row + 1):
            y = offset_y + (origin[1] + row * stride[1]) * scale
            if y < offset_y - 1 or y > offset_y + display_height + 1:
                continue
            canvas.create_line(
                offset_x,
                y,
                offset_x + display_width,
                y,
                fill="yellow",
                width=1,
                tags="grid_line",
            )

        canvas.itemconfigure(
            text_id,
            text=(
                "Move mouse to shift the full grid. Left click to lock, right click to unlock, Enter to confirm. "
                f"origin=({origin[0]:.1f}, {origin[1]:.1f}), stride={stride}, locked={state['locked']}"
            ),
        )

    def on_motion(event: tk.Event) -> None:
        if state["locked"]:
            return
        if not (offset_x <= event.x <= offset_x + display_width and offset_y <= event.y <= offset_y + display_height):
            return
        state["current_origin"] = event_to_origin(event.x, event.y)
        redraw(state["current_origin"])

    def on_left_click(event: tk.Event) -> None:
        if state["locked"]:
            cell = event_to_cell(event.x, event.y, state["current_origin"])
            if cell is not None:
                row, col = cell
                if (row, col) in state["tile_attrs"]:
                    state["tile_attrs"][(row, col)]["is_export"] = False
                    canvas.delete(f"grid_fill_{row}_{col}")
                    draw_cell_fill(row, col, state["current_origin"])
            return
        else:
            if offset_x <= event.x <= offset_x + display_width and offset_y <= event.y <= offset_y + display_height:
                state["current_origin"] = event_to_origin(event.x, event.y)
            snapped_origin = finalize_origin(state["current_origin"])
            state["current_origin"] = (float(snapped_origin[0]), float(snapped_origin[1]))
            state["locked"] = True
            state["tile_attrs"] = build_locked_tile_attrs(state["current_origin"])
        redraw(state["current_origin"])

    def on_right_click(event: tk.Event) -> None:
        state["locked"] = False
        state["tile_attrs"] = {}
        if offset_x <= event.x <= offset_x + display_width and offset_y <= event.y <= offset_y + display_height:
            state["current_origin"] = event_to_origin(event.x, event.y)
        redraw(state["current_origin"])

    def finalize(_: tk.Event | None = None) -> None:
        snapped_origin = finalize_origin(state["current_origin"])
        is_export_map = {cell: attrs["is_export"] for cell, attrs in state["tile_attrs"].items()}
        crops = _build_tile_crops(
            image=image,
            origin=snapped_origin,
            tile_size=tile_size,
            stride=stride,
            include_edge_tiles=include_edge_tiles,
            is_export_map=is_export_map,
        )
        state["result"] = crops
        root.quit()

    root.bind("<Motion>", on_motion)
    root.bind("<Button-1>", on_left_click)
    root.bind("<Button-3>", on_right_click)
    root.bind("<Return>", finalize)

    redraw(state["current_origin"])
    root.mainloop()
    root.destroy()

    if state["result"] is None:
        return []
    return state["result"]


def save_interactive_tiles(
    image_path: str | Path,
    save_dir: str | Path,
    tile_size: int | tuple[int, int] = (2048, 2048),
    overlap: float | tuple[float, float] | tuple[int, int] = 0.1,
    canvas_size: tuple[int, int] = (1920, 1080),
    include_edge_tiles: bool = True,
) -> list[Path]:
    image_path = Path(image_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tiles = interactive_grid_tile_selector(
        image_or_path=image_path,
        tile_size=tile_size,
        overlap=overlap,
        canvas_size=canvas_size,
        include_edge_tiles=include_edge_tiles,
    )

    saved_paths: list[Path] = []
    for tile in tiles:
        if not tile.is_export:
            continue

        min_x, min_y = tile.minxy
        output_path = save_dir / f"{image_path.stem}_x{min_x}_y{min_y}{image_path.suffix}"
        tile.image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths
