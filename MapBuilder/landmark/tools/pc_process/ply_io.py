"""PLY file reading/writing and ground mask extraction."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Optional

import numpy as np
from plyfile import PlyData, PlyElement

PRED_PALETTE = np.asarray(
    [
        (128, 128, 128),
        (255, 99, 71),
        (135, 206, 235),
        (255, 215, 0),
        (50, 205, 50),
        (238, 130, 238),
        (255, 140, 0),
        (70, 130, 180),
        (255, 105, 180),
        (154, 205, 50),
        (0, 191, 255),
        (255, 160, 122),
        (106, 90, 205),
        (46, 139, 87),
        (255, 69, 0),
        (72, 209, 204),
    ],
    dtype=np.uint8,
)


def _get_property(
    properties: dict[str, Any],
    *names: str,
) -> np.ndarray:
    for name in names:
        if name in properties:
            return np.asarray(properties[name])
    raise KeyError(f"None of {names!r} found in properties")


def get_ground_mask(properties: dict[str, Any]) -> np.ndarray:
    randla = _get_property(properties, "randla_pred", "pred_randla")
    spt = _get_property(properties, "spt_pred", "pred_spt")
    randla_ground = (randla == 6) | (randla == 7)
    spt_sidewalk = spt == 8
    _ = spt_sidewalk  # kept for readability / future logic
    return randla_ground


def read_ply(
    fp: PathLike | str, is_property: bool = False
) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    plydata = PlyData.read(fp)
    vertex_data = plydata["vertex"]
    names = vertex_data.data.dtype.names

    points = np.stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=-1
    )

    has_rgb = {"red", "green", "blue"}.issubset(names)
    colors = None
    if has_rgb:
        colors = np.stack(
            [vertex_data["red"], vertex_data["green"], vertex_data["blue"]], axis=-1
        )
    if is_property:
        properties: dict[str, Any] = {}
        for prop in vertex_data.data.dtype.names:
            if prop not in ["x", "y", "z", "red", "green", "blue"]:
                properties[prop] = vertex_data[prop]
        return points, colors, properties
    else:
        return points, colors


def _ensure_supported_dtype(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with a plyfile-supported dtype, casting if loss-less."""
    supported = {
        ("i", 1): np.int8,
        ("i", 2): np.int16,
        ("i", 4): np.int32,
        ("u", 1): np.uint8,
        ("u", 2): np.uint16,
        ("u", 4): np.uint32,
        ("f", 4): np.float32,
        ("f", 8): np.float64,
    }

    kind, itemsize = arr.dtype.kind, arr.dtype.itemsize
    if (kind, itemsize) in supported:
        return arr

    if kind == "i":
        target = np.int32
    elif kind == "u":
        target = np.uint32
    else:
        raise ValueError(
            f"Unsupported dtype {arr.dtype}. plyfile allows int/uint ≤32-bit "
            "or float32/float64."
        )

    info = np.iinfo(target)
    if arr.min() < info.min or arr.max() > info.max:
        raise ValueError(
            f"Values in array range [{arr.min()}, {arr.max()}] exceed {target} "
            "bounds; cannot safely cast."
        )
    return arr.astype(target, copy=False)


def _new_vertex_dtype(
    old_dtype: np.dtype, new_field: str, new_dtype: np.dtype
) -> np.dtype:
    """Create a new structured dtype with *new_field* appended."""
    if old_dtype.names is None:
        raise ValueError("Vertex element has no named properties to extend.")
    if new_field in old_dtype.names:
        raise ValueError(f"Property '{new_field}' already exists in vertex data.")

    descr: list[Any] = old_dtype.descr + [(new_field, new_dtype.str)]
    return np.dtype(descr)


def write_ply(
    fp: PathLike,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    intensity: Optional[np.ndarray] = None,
    label: Optional[np.ndarray] = None,
    properties: Optional[dict[str, np.ndarray]] = None,
    text: bool = False,
    byte_order: str = "<",
) -> None:
    """Write a point cloud to a PLY file."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n = points.shape[0]

    dtype_fields: list[tuple[str, str]] = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    columns: dict[str, np.ndarray] = {
        "x": np.asarray(points[:, 0], dtype=np.float32),
        "y": np.asarray(points[:, 1], dtype=np.float32),
        "z": np.asarray(points[:, 2], dtype=np.float32),
    }

    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != n:
            raise ValueError("colors must have shape (N, 3) and match points")
        colors_u8 = colors.astype(np.uint8, copy=False)
        dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        columns["red"] = colors_u8[:, 0]
        columns["green"] = colors_u8[:, 1]
        columns["blue"] = colors_u8[:, 2]

    def _add_1d_property(name: str, arr: np.ndarray) -> None:
        if name in columns:
            raise ValueError(f"Duplicate property name: {name}")
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"Property '{name}' must be a 1-D array")
        if arr.shape[0] != n:
            raise ValueError(
                f"Property '{name}' length {arr.shape[0]} does not match points length {n}"
            )
        arr = np.ascontiguousarray(arr)
        arr = _ensure_supported_dtype(arr)
        dtype_fields.append((name, arr.dtype.str))
        columns[name] = arr

    if intensity is not None:
        _add_1d_property("intensity", intensity)

    if label is not None:
        _add_1d_property("label", label)

    if properties:
        for k, v in properties.items():
            _add_1d_property(str(k), v)

    vertex = np.empty(n, dtype=np.dtype(dtype_fields))
    for name, arr in columns.items():
        vertex[name] = arr

    el = PlyElement.describe(vertex, "vertex")
    ply = PlyData([el], text=text, byte_order=byte_order)

    out_path = Path(fp).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_path))
