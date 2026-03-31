import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VARIANTS = (
    ("rgb_raw", 3),
    ("intensity_raw", 1),
    ("rgb_inpaint", 3),
    ("intensity_inpaint", 1),
)


def _variant_img_path(input_root_dir: Path, stem: str, variant: str) -> Path:
    return input_root_dir / variant / f"{stem}.png"


@dataclass
class TileInfo:
    las_name: str
    stem: str
    left_top_x: float
    left_top_y: float
    left_top_z: float
    offset_x: int = 0
    offset_y: int = 0
    width: int = 0
    height: int = 0


def cv2_imread_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def cv2_imwrite_any(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def parse_positions(positions_txt: Path) -> list[TileInfo]:
    if not positions_txt.exists():
        raise FileNotFoundError(f"positions file not found: {positions_txt}")

    tiles: list[TileInfo] = []
    with positions_txt.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"positions file is empty: {positions_txt}")

    start_idx = 0
    first_cols = lines[0].split()
    if first_cols and first_cols[0].lower() == "las_name":
        start_idx = 1

    for line in lines[start_idx:]:
        cols = line.split()
        if len(cols) < 4:
            continue
        las_name = cols[0]
        stem = Path(las_name).stem
        tiles.append(
            TileInfo(
                las_name=las_name,
                stem=stem,
                left_top_x=float(cols[1]),
                left_top_y=float(cols[2]),
                left_top_z=float(cols[3]),
            )
        )
    if not tiles:
        raise ValueError(f"no valid tile rows in positions file: {positions_txt}")
    return tiles


def resolve_tile_size(tiles: list[TileInfo], input_root_dir: Path) -> None:
    for tile in tiles:
        found_shape = None
        for variant, _ in VARIANTS:
            img_path = _variant_img_path(input_root_dir, tile.stem, variant)
            if not img_path.exists():
                continue
            img = cv2_imread_any(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            found_shape = (w, h)
            break
        if found_shape is None:
            raise FileNotFoundError(
                f"No projection images found for LAS '{tile.las_name}' in {input_root_dir}"
            )
        tile.width, tile.height = found_shape


def compute_layout(tiles: list[TileInfo], resolution: float) -> tuple[int, int]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    min_x = min(t.left_top_x for t in tiles)
    max_y = max(t.left_top_y for t in tiles)

    max_w = 0
    max_h = 0
    for tile in tiles:
        tile.offset_x = int(round((tile.left_top_x - min_x) / resolution))
        tile.offset_y = int(round((max_y - tile.left_top_y) / resolution))
        max_w = max(max_w, tile.offset_x + tile.width)
        max_h = max(max_h, tile.offset_y + tile.height)

    if max_w <= 0 or max_h <= 0:
        raise ValueError("computed canvas size is invalid")
    return max_h, max_w


def paste_nonzero(dst: np.ndarray, src: np.ndarray, y0: int, x0: int) -> None:
    h, w = src.shape[:2]
    roi = dst[y0 : y0 + h, x0 : x0 + w]
    if src.ndim == 2:
        mask = src != 0
        roi[mask] = src[mask]
    else:
        mask = np.any(src != 0, axis=2)
        roi[mask] = src[mask]
    dst[y0 : y0 + h, x0 : x0 + w] = roi


def stitch_variant(
    tiles: list[TileInfo],
    input_root_dir: Path,
    canvas_h: int,
    canvas_w: int,
    variant: str,
    channels: int,
) -> np.ndarray:
    if channels == 1:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.uint8)

    for tile in tqdm(tiles, desc=f"Stitch {variant}", unit="tile"):
        img_path = _variant_img_path(input_root_dir, tile.stem, variant)
        if not img_path.exists():
            print(f"Warning: skip missing image: {img_path}")
            continue
        img = cv2_imread_any(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read image: {img_path}")
            continue

        if channels == 3:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
        else:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        paste_nonzero(canvas, img, tile.offset_y, tile.offset_x)
    return canvas


def run_stitch(
    input_root_dir: Path,
    positions_txt: Path,
    output_dir: Path,
    resolution: float,
    prefix: str,
) -> None:
    tiles = parse_positions(positions_txt)
    resolve_tile_size(tiles, input_root_dir)
    canvas_h, canvas_w = compute_layout(tiles, resolution)
    print(f"Canvas size: {canvas_w} x {canvas_h}")

    for variant, channels in VARIANTS:
        stitched = stitch_variant(
            tiles=tiles,
            input_root_dir=input_root_dir,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            variant=variant,
            channels=channels,
        )
        out_path = output_dir / f"{prefix}_{variant}.png"
        ok = cv2_imwrite_any(out_path, stitched)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch 4 BEV projection types into 4 large mosaics using LAS position txt."
    )
    parser.add_argument(
        "input_root_dir",
        type=str,
        help="Root directory containing 4 subfolders: rgb_raw, intensity_raw, rgb_inpaint, intensity_inpaint",
    )
    parser.add_argument("positions_txt", type=str, help="Path to las_positions.txt")
    parser.add_argument("output_dir", type=str, help="Output directory for stitched images")
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Projection resolution (meters/pixel), must match pc2img generation",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="stitched",
        help="Output file name prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_root_dir = Path(args.input_root_dir).expanduser().absolute()
    positions_txt = Path(args.positions_txt).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()

    if not input_root_dir.exists() or not input_root_dir.is_dir():
        raise NotADirectoryError(f"input_root_dir not found: {input_root_dir}")

    for variant, _ in VARIANTS:
        variant_dir = input_root_dir / variant
        if not variant_dir.exists() or not variant_dir.is_dir():
            raise NotADirectoryError(f"missing variant folder: {variant_dir}")

    run_stitch(
        input_root_dir=input_root_dir,
        positions_txt=positions_txt,
        output_dir=output_dir,
        resolution=args.resolution,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
