"""Bird's-eye-view (BEV) rendering utilities.

Supported modes
---------------
* ``rgb``          – average RGB colour per pixel
* ``intensity``    – scalar_Intensity mapped to grayscale
* ``height``       – normalized ground height mapped to grayscale
* ``spt_pred``     – pred_spt labels coloured via PCSS_COLORMAP
* ``m2f_pred``     – pred_m2f labels coloured via PCSS_COLORMAP
* ``randla_pred``  – pred_randla labels coloured via PCSS_COLORMAP

Usage
-----
    from landmark.tools.pc_process.bev import render_bev, render_bev_to_file

    # Single mode
    img, meta = render_bev("tile.ply", mode="rgb", mpp=0.02)

    # Multiple modes (one CSF pass)
    results = render_bev("tile.ply", mode=["rgb", "intensity"], mpp=0.02)
    # results == {"rgb": (img, meta), "intensity": (img, meta)}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from landmark.tools.pc_process.ply_io import read_ply

# PCSS semantic label colour palette (RGB, index = class id)
PCSS_COLORMAP = [
    [0, 0, 0],           # 图像 没有看到
    [0, 0, 0],           # 其他
    [128, 0, 0],         # 围栏
    [128, 64, 64],       # 墙
    [128, 128, 128],     # 建筑
    [0, 128, 0],         # 植被
    [255, 255, 0],       # 道路标线
    [128, 128, 0],       # 道路
    [192, 192, 192],     # 人行道
    [255, 128, 0],       # 杆
    [255, 0, 0],         # 交通灯
    [255, 0, 255],       # 交通标志
    [255, 192, 203],     # 人
    [0, 0, 255],         # 汽车
    [0, 64, 128],        # 卡车
    [0, 255, 0],         # 自行车
    [0, 255, 128],       # 摩托车
    [128, 128, 255],     # 电力线
    [0, 128, 128],       # 电线杆
    [192, 128, 0],       # 路灯
    [255, 255, 128],     # 广告牌
    [64, 64, 64],        # 井盖
    [192, 64, 0],        # 桥
    [64, 0, 128],        # 隧道
    [128, 0, 255],       # 路沿
    [64, 128, 128],      # 路侧挡块
    [128, 64, 0],        # 路中挡块
    [192, 0, 128],       # 护栏
    [0, 192, 64],        # 非机动车道
    [0, 128, 64],        # 服务车道
    [128, 192, 0],       # 道路隔离带
    [128, 0, 128],       # 垃圾桶
    [0, 0, 128],         # 集水池
    [64, 0, 64],         # 接线盒
    [0, 64, 64],         # 闭路电视摄像机
    [255, 64, 64],       # 消防栓
    [128, 128, 64],      # 长凳
    [0, 192, 192],       # 电话亭
]

_PCSS_PALETTE = np.asarray(PCSS_COLORMAP, dtype=np.uint8)

# Mode → preferred/current PLY field name plus legacy aliases
_PRED_MODE_FIELDS: dict[str, tuple[str, ...]] = {
    "spt_pred": ("spt_pred", "pred_spt"),
    "m2f_pred": ("m2f_pred", "pred_m2f"),
    "randla_pred": ("randla_pred", "pred_randla"),
}

VALID_MODES = {"rgb", "intensity", "height", "spt_pred", "m2f_pred", "randla_pred"}


def _resolve_property_key(properties: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in properties:
            return key
    return None


def _csf_ground_mask(
    points: np.ndarray,
    *,
    cloth_resolution: float = 0.5,
    class_threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean ground mask using Cloth Simulation Filter (CSF).

    Parameters
    ----------
    points : (N, 3) array
        XYZ coordinates.
    cloth_resolution : float
        CSF cloth grid resolution in metres (smaller = finer).
    class_threshold : float
        Height threshold for ground classification.

    Returns
    -------
    mask : (N,) bool array – True for ground points.
    """
    import CSF

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = cloth_resolution
    csf.params.class_threshold = class_threshold
    csf.setPointCloud(points[:, :3].astype(np.float64))

    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)

    mask = np.zeros(len(points), dtype=bool)
    mask[list(ground_idx)] = True
    return mask


def _project_xy(
    xy: np.ndarray,
    min_xy: np.ndarray,
    mpp: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project XY world coords → pixel (row, col) and return valid mask."""
    col = np.floor((xy[:, 0] - min_xy[0]) / mpp).astype(np.int64)
    row = (height - 1) - np.floor((xy[:, 1] - min_xy[1]) / mpp).astype(np.int64)
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return row, col, valid


def _render_single(
    mode: str,
    *,
    idx: np.ndarray,
    height: int,
    width: int,
    colors_ground_valid: np.ndarray | None,
    properties: dict[str, Any],
    ground: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Render one BEV mode from pre-projected pixel indices."""
    if mode == "rgb":
        if colors_ground_valid is None:
            raise KeyError("RGB colours not found in PLY")
        c = colors_ground_valid.astype(np.float32)
        sums = np.zeros((height * width, 3), dtype=np.float32)
        counts = np.zeros(height * width, dtype=np.float32)
        np.add.at(sums, idx, c)
        np.add.at(counts, idx, 1.0)
        has = counts > 0
        rgb_flat = np.zeros((height * width, 3), dtype=np.float32)
        rgb_flat[has] = sums[has] / counts[has, None]
        return np.clip(rgb_flat, 0, 255).astype(np.uint8).reshape(height, width, 3)

    if mode == "intensity":
        key = "scalar_Intensity"
        if key not in properties:
            raise KeyError(f"'{key}' not found in PLY properties")
        intensity = np.asarray(properties[key])[ground][valid].astype(np.float32)
        sums = np.zeros(height * width, dtype=np.float32)
        counts = np.zeros(height * width, dtype=np.float32)
        np.add.at(sums, idx, intensity)
        np.add.at(counts, idx, 1.0)
        has = counts > 0
        avg = np.zeros(height * width, dtype=np.float32)
        avg[has] = sums[has] / counts[has]
        lo, hi = avg[has].min(), avg[has].max()
        if hi > lo:
            avg[has] = (avg[has] - lo) / (hi - lo) * 255.0
        gray = np.clip(avg, 0, 255).astype(np.uint8).reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if mode == "height":
        raise RuntimeError("height mode is handled by render_height_bev / render_bev")

    # pred modes
    field_candidates = _PRED_MODE_FIELDS[mode]
    field = _resolve_property_key(properties, field_candidates)
    if field is None:
        raise KeyError(
            f"None of {field_candidates!r} found in PLY properties"
        )
    labels = np.asarray(properties[field])[ground][valid].astype(np.int32)
    flat = np.full(height * width, -1, dtype=np.int32)
    np.maximum.at(flat, idx, labels)
    label_img = flat.reshape(height, width)
    palette = _PCSS_PALETTE
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = label_img >= 0
    if np.any(mask):
        idx_mod = (label_img[mask] % len(palette)).astype(np.int64)
        img[mask] = palette[idx_mod]
    return img


def render_bev(
    ply_path: str | Path,
    *,
    mode: str | list[str] = "intensity",
    mpp: float = 0.02,
    skip_missing_fields: bool = False,
    apply_csf: bool = True,
) -> tuple[np.ndarray, dict[str, Any]] | dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Render BEV image(s) from a PLY point cloud.

    Parameters
    ----------
    ply_path : path
        Input PLY file.
    mode : str or list[str]
        One or more of: ``"rgb"``, ``"intensity"``, ``"height"``, ``"spt_pred"``,
        ``"m2f_pred"``, ``"randla_pred"``.
        When a **single string** is given, returns ``(img, meta)``.
        When a **list** is given, returns ``{mode: (img, meta), ...}``.
    mpp : float
        Metres per pixel (default 0.02).
    skip_missing_fields : bool
        When rendering multiple modes, skip modes whose required PLY fields
        are unavailable instead of raising ``KeyError``. Single-mode calls
        remain strict.
    """
    ply_path = Path(ply_path).expanduser()

    # Normalise modes
    if isinstance(mode, str):
        modes = [mode.strip().lower()]
        single = True
    else:
        modes = [m.strip().lower() for m in mode]
        single = False

    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got '{m}'")

    # Read PLY once
    points, colors, properties = read_ply(ply_path, is_property=True)

    # CSF ground filter once unless the input point cloud is already ground-only.
    if apply_csf:
        ground = _csf_ground_mask(points)
    else:
        ground = np.ones(points.shape[0], dtype=bool)
    pts = points[ground]

    if pts.shape[0] == 0:
        empty = np.zeros((2, 2, 3), dtype=np.uint8)
        meta: dict[str, Any] = {
            "min_xy": [0.0, 0.0],
            "max_xy": [0.0, 0.0],
            "meters_per_pixel": float(mpp),
            "width": 2,
            "height": 2,
        }
        if single:
            return empty, meta
        return {m: (empty.copy(), dict(meta)) for m in modes}

    xy = pts[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    ext = max_xy - min_xy
    width = max(int(np.ceil(ext[0] / mpp)) + 1, 2)
    height = max(int(np.ceil(ext[1] / mpp)) + 1, 2)

    row, col, valid = _project_xy(xy, min_xy, mpp, height, width)
    row, col = row[valid], col[valid]
    pixel_idx = row * width + col

    colors_ground_valid = (
        np.asarray(colors)[ground][valid] if colors is not None else None
    )

    meta = {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "meters_per_pixel": float(mpp),
        "width": int(width),
        "height": int(height),
    }

    render_kwargs = dict(
        idx=pixel_idx,
        height=height,
        width=width,
        colors_ground_valid=colors_ground_valid,
        properties=properties,
        ground=ground,
        valid=valid,
    )

    if single:
        if modes[0] == "height":
            height_values, lo, hi = _render_height_raster_from_points(
                points=pts,
                idx=pixel_idx,
                height=height,
                width=width,
                valid=valid,
            )
            img = _height_raster_to_rgb(height_values, lo=lo, hi=hi)
        else:
            img = _render_single(modes[0], **render_kwargs)
        return img, meta

    results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for m in modes:
        try:
            if m == "height":
                height_values, lo, hi = _render_height_raster_from_points(
                    points=pts,
                    idx=pixel_idx,
                    height=height,
                    width=width,
                    valid=valid,
                )
                img = _height_raster_to_rgb(height_values, lo=lo, hi=hi)
            else:
                img = _render_single(m, **render_kwargs)
        except KeyError:
            if not skip_missing_fields:
                raise
            continue
        results[m] = (img, dict(meta))
    return results


def _robust_height_mask(z_values: np.ndarray) -> np.ndarray:
    z = np.asarray(z_values, dtype=np.float32)
    finite = np.isfinite(z)
    if not np.any(finite):
        return finite

    zf = z[finite]
    q1, q3 = np.percentile(zf, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0:
        lo, hi = np.percentile(zf, [1.0, 99.0])
    else:
        lo = max(float(np.percentile(zf, 1.0)), float(q1 - 3.0 * iqr))
        hi = min(float(np.percentile(zf, 99.0)), float(q3 + 3.0 * iqr))
    mask = finite & (z >= lo) & (z <= hi)
    if not np.any(mask):
        return finite
    return mask


def _render_height_raster_from_points(
    *,
    points: np.ndarray,
    idx: np.ndarray,
    height: int,
    width: int,
    valid: np.ndarray,
) -> tuple[np.ndarray, float | None, float | None]:
    z_values = np.asarray(points[:, 2], dtype=np.float32)
    inlier_mask = _robust_height_mask(z_values)
    if not np.any(inlier_mask):
        return np.full((height, width), np.nan, dtype=np.float32), None, None

    z_valid = z_values[valid]
    inlier_valid = inlier_mask[valid]
    if not np.any(inlier_valid):
        return np.full((height, width), np.nan, dtype=np.float32), None, None

    flat = np.full(height * width, -np.inf, dtype=np.float32)
    np.maximum.at(flat, idx[inlier_valid], z_valid[inlier_valid])
    has = np.isfinite(flat) & (flat > -np.inf)
    if not np.any(has):
        return np.full((height, width), np.nan, dtype=np.float32), None, None

    lo = float(np.min(flat[has]))
    hi = float(np.max(flat[has]))
    raster = np.full(height * width, np.nan, dtype=np.float32)
    raster[has] = flat[has]
    return raster.reshape(height, width), lo, hi


def _height_raster_to_rgb(
    height_values: np.ndarray,
    *,
    lo: float | None,
    hi: float | None,
) -> np.ndarray:
    gray = np.zeros(height_values.shape, dtype=np.uint8)
    has = np.isfinite(height_values)
    if np.any(has) and lo is not None and hi is not None and hi > lo:
        scaled = (height_values[has] - lo) / (hi - lo) * 255.0
        gray[has] = np.clip(scaled, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def render_height_raster(
    ply_path: str | Path,
    *,
    mpp: float = 0.02,
    apply_csf: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a numeric height raster from a PLY point cloud.

    Returns a float32 ``(H, W)`` array in metres with ``NaN`` for empty pixels,
    plus the usual BEV meta dict. The raster uses the max inlier z per pixel.
    """
    ply_path = Path(ply_path).expanduser()
    points, _colors, _properties = read_ply(ply_path, is_property=True)

    if apply_csf:
        ground = _csf_ground_mask(points)
    else:
        ground = np.ones(points.shape[0], dtype=bool)
    pts = points[ground]

    if pts.shape[0] == 0:
        meta: dict[str, Any] = {
            "min_xy": [0.0, 0.0],
            "max_xy": [0.0, 0.0],
            "meters_per_pixel": float(mpp),
            "width": 2,
            "height": 2,
            "vis_height_lo": None,
            "vis_height_hi": None,
        }
        return np.full((2, 2), np.nan, dtype=np.float32), meta

    xy = pts[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    ext = max_xy - min_xy
    width = max(int(np.ceil(ext[0] / mpp)) + 1, 2)
    height = max(int(np.ceil(ext[1] / mpp)) + 1, 2)

    row, col, valid = _project_xy(xy, min_xy, mpp, height, width)
    row, col = row[valid], col[valid]
    pixel_idx = row * width + col
    height_values, lo, hi = _render_height_raster_from_points(
        points=pts,
        idx=pixel_idx,
        height=height,
        width=width,
        valid=valid,
    )
    meta = {
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "meters_per_pixel": float(mpp),
        "width": int(width),
        "height": int(height),
        "vis_height_lo": lo,
        "vis_height_hi": hi,
    }
    return height_values, meta


def render_height_bev(
    ply_path: str | Path,
    output_path: str | Path,
    *,
    mpp: float = 0.02,
    apply_csf: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Render a height BEV PNG from PLY z-values.

    The function removes significant z outliers using a robust IQR/percentile
    filter, then normalizes the remaining max height per pixel to [0, 255].
    It also writes a float32 ``*_values.npy`` raster in metres for downstream use.
    """
    height_values, meta = render_height_raster(
        ply_path,
        mpp=mpp,
        apply_csf=apply_csf,
    )
    img = _height_raster_to_rgb(
        height_values,
        lo=meta.get("vis_height_lo"),
        hi=meta.get("vis_height_hi"),
    )
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(out))
    values_out = out.with_name(f"{out.stem}_values.npy")
    np.save(values_out, np.asarray(height_values, dtype=np.float32))
    return out, values_out, meta


def render_bev_to_file(
    ply_path: str | Path,
    output_path: str | Path | dict[str, str | Path] | None = None,
    *,
    mode: str | list[str] = "intensity",
    mpp: float = 0.02,
) -> Path | dict[str, Path]:
    """Render BEV and save as PNG.

    Parameters
    ----------
    ply_path : path
        Input PLY file.
    output_path : path, dict, or None
        * For a single mode string: a single output path.
        * For a list of modes: either a ``{mode: path}`` dict, or ``None``
          to auto-name as ``<ply_stem>_<mode>.png`` next to the input.
    mode : str or list[str]
        See :func:`render_bev`.
    mpp : float
        Metres per pixel.

    Returns
    -------
    Path or dict[str, Path] matching the *mode* shape.
    """
    ply_path = Path(ply_path).expanduser()
    single = isinstance(mode, str)

    result = render_bev(ply_path, mode=mode, mpp=mpp)

    if single:
        assert isinstance(result, tuple)
        img, _meta = result
        out = Path(output_path).expanduser() if output_path else ply_path.with_name(f"{ply_path.stem}_{mode}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out))
        return out

    # Multiple modes
    assert isinstance(result, dict)
    modes = list(mode) if isinstance(mode, list) else [mode]
    out_map: dict[str, str | Path] = {}
    if isinstance(output_path, dict):
        out_map = output_path
    elif output_path is None:
        for m in modes:
            out_map[m] = ply_path.with_name(f"{ply_path.stem}_{m}.png")
    else:
        raise TypeError("output_path must be a dict or None when mode is a list")

    written: dict[str, Path] = {}
    for m in modes:
        img, _meta = result[m]
        out = Path(out_map[m]).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out))
        written[m] = out
    return written


def main() -> None:
    """CLI entrypoint for BEV rendering."""
    import argparse

    parser = argparse.ArgumentParser(description="Render a BEV image from a PLY point cloud.")
    parser.add_argument("ply", help="Input PLY file path.")
    parser.add_argument(
        "-m",
        "--mode",
        nargs="+",
        choices=sorted(VALID_MODES),
        default=["intensity"],
        help="Rendering mode(s) (default: intensity). Multiple modes share one CSF pass.",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=0.02,
        help="Metres per pixel (default: 0.02).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path (only for single mode; multi-mode auto-names).",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply).expanduser()
    modes = args.mode

    if len(modes) == 1:
        out_path = Path(args.output).expanduser() if args.output else None
        if out_path is None:
            out_path = ply_path.with_name(f"{ply_path.stem}_{modes[0]}.png")
        out = render_bev_to_file(ply_path, out_path, mode=modes[0], mpp=args.mpp)
        print(f"Saved BEV ({modes[0]}) → {out}")
    else:
        written = render_bev_to_file(ply_path, None, mode=modes, mpp=args.mpp)
        assert isinstance(written, dict)
        for m, p in written.items():
            print(f"Saved BEV ({m}) → {p}")


if __name__ == "__main__":
    main()
