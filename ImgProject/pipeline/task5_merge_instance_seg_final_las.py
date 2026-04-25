from __future__ import annotations

import argparse
import sys
from pathlib import Path

import laspy
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import utils as task5_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge scene-level fusion/*_instance_seg_final.las files into one "
            "benchmark-level LAS, and add per-point global_gid."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Benchmark root directory that contains per-scene folders.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="instance_seg_final_merged.las",
        help="Output LAS filename written under data-root.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def discover_final_las(data_root: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        fusion_dir = scene_dir / "fusion"
        if not fusion_dir.is_dir():
            continue

        preferred = fusion_dir / f"{scene_dir.name}_instance_seg_final.las"
        if preferred.exists():
            entries.append((scene_dir.name, preferred))
            continue

        matches = sorted(fusion_dir.glob("*_instance_seg_final.las"))
        if not matches:
            continue
        entries.append((scene_dir.name, matches[0]))
        if len(matches) > 1:
            print(
                f"[warn] {scene_dir.name}: found multiple final LAS files, "
                f"use {matches[0].name}"
            )
    return entries


def _has_extra_dim(path: Path, dim_name: str) -> bool:
    with laspy.open(path) as reader:
        extra_dims = set(reader.header.point_format.extra_dimension_names)
    return dim_name in extra_dims


def _build_output_header(first_path: Path, include_cls_id: bool) -> laspy.LasHeader:
    with laspy.open(first_path) as reader:
        first_header = reader.header
        point_format_id = int(first_header.point_format.id)
        version = first_header.version
        scales = np.asarray(first_header.scales, dtype=np.float64)
        offsets = np.asarray(first_header.offsets, dtype=np.float64)

    header = laspy.LasHeader(point_format=point_format_id, version=version)
    header.scales = scales
    header.offsets = offsets

    extra_bytes_params = getattr(laspy, "ExtraBytesParams", None)
    if extra_bytes_params is not None:
        if include_cls_id:
            header.add_extra_dim(extra_bytes_params(name="cls_id", type=np.uint8))
        header.add_extra_dim(extra_bytes_params(name="global_gid", type=np.int32))
    return header


def _safe_int32_array(values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32).reshape(-1)
    if arr.shape[0] == n:
        return arr
    out = np.full((n,), -1, dtype=np.int32)
    copy_n = min(n, arr.shape[0])
    out[:copy_n] = arr[:copy_n]
    return out


def _compute_global_gid_for_scene(
    *,
    scene_name: str,
    scene_id_map: dict[str, int],
    las_in: laspy.LasData,
) -> tuple[np.ndarray, dict[str, int]]:
    scene_id = scene_id_map.get(scene_name, None)
    if scene_id is None:
        raise KeyError(f"Scene missing in scene_id_map: {scene_name}")

    n = int(len(las_in.x))
    if hasattr(las_in, "point_scene_instance_id"):
        scene_inst_id = _safe_int32_array(np.asarray(las_in.point_scene_instance_id), n)
    else:
        scene_inst_id = np.full((n,), -1, dtype=np.int32)

    if hasattr(las_in, "point_pole_group_id"):
        pole_group_id = _safe_int32_array(np.asarray(las_in.point_pole_group_id), n)
    else:
        pole_group_id = np.full((n,), -1, dtype=np.int32)

    pole_mask = pole_group_id >= 0
    scene_mask = (~pole_mask) & (scene_inst_id >= 0)
    valid_mask = pole_mask | scene_mask

    object_id = np.full((n,), -1, dtype=np.int32)
    object_id[pole_mask] = pole_group_id[pole_mask]
    object_id[scene_mask] = scene_inst_id[scene_mask]

    if np.any(object_id[valid_mask] > int(task5_utils.GLOBAL_GID_OBJECT_MAX)):
        max_obj = int(np.max(object_id[valid_mask]))
        raise ValueError(
            f"{scene_name}: object_id exceeds global_gid capacity "
            f"({max_obj} > {int(task5_utils.GLOBAL_GID_OBJECT_MAX)})"
        )

    global_gid = np.full((n,), -1, dtype=np.int32)
    if np.any(valid_mask):
        type_code = np.zeros((n,), dtype=np.int32)
        type_code[pole_mask] = int(task5_utils.GLOBAL_GID_TYPE_POLE_GROUP)

        gid64 = (
            (np.int64(scene_id) << np.int64(task5_utils.GLOBAL_GID_SCENE_SHIFT))
            | (type_code.astype(np.int64) << np.int64(task5_utils.GLOBAL_GID_TYPE_SHIFT))
            | object_id.astype(np.int64)
        )
        gid_valid = gid64[valid_mask]
        if gid_valid.size > 0 and int(np.max(gid_valid)) > int(np.iinfo(np.int32).max):
            raise ValueError(f"{scene_name}: global_gid exceeds int32 positive range.")
        global_gid[valid_mask] = gid_valid.astype(np.int32, copy=False)

    stats = {
        "scene_points": int(np.count_nonzero(scene_mask)),
        "pole_points": int(np.count_nonzero(pole_mask)),
        "invalid_points": int(np.count_nonzero(~valid_mask)),
    }
    return global_gid, stats


def _copy_to_output_chunk(
    *,
    las_in: laspy.LasData,
    header: laspy.LasHeader,
    include_cls_id: bool,
    global_gid: np.ndarray,
) -> laspy.LasData:
    out = laspy.LasData(header)
    out.x = np.asarray(las_in.x)
    out.y = np.asarray(las_in.y)
    out.z = np.asarray(las_in.z)
    n = int(len(out.x))

    if hasattr(las_in, "red"):
        out.red = np.asarray(las_in.red, dtype=np.uint16)
        out.green = np.asarray(las_in.green, dtype=np.uint16)
        out.blue = np.asarray(las_in.blue, dtype=np.uint16)
    else:
        zeros_rgb = np.zeros((n,), dtype=np.uint16)
        out.red = zeros_rgb
        out.green = zeros_rgb
        out.blue = zeros_rgb

    if hasattr(las_in, "classification"):
        cls = np.asarray(las_in.classification, dtype=np.uint8)
    else:
        cls = np.zeros((n,), dtype=np.uint8)
    out.classification = cls

    if include_cls_id and hasattr(out, "cls_id"):
        if hasattr(las_in, "cls_id"):
            out.cls_id = np.asarray(las_in.cls_id, dtype=np.uint8)
        else:
            out.cls_id = cls

    if hasattr(out, "global_gid"):
        out.global_gid = np.asarray(global_gid, dtype=np.int32).reshape(-1)

    return out


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    entries = discover_final_las(data_root)
    if not entries:
        raise FileNotFoundError(
            f"No *_instance_seg_final.las found under scene fusion folders: {data_root}"
        )

    scene_id_map = task5_utils.build_scene_id_map(data_root)
    output_path = data_root / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (set --overwrite to replace)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_cls_id = any(_has_extra_dim(path, "cls_id") for _, path in entries)
    header = _build_output_header(entries[0][1], include_cls_id=include_cls_id)

    total_points = 0
    with laspy.open(output_path, mode="w", header=header) as writer:
        for scene_name, in_path in entries:
            las_in = laspy.read(in_path)
            global_gid, gid_stats = _compute_global_gid_for_scene(
                scene_name=scene_name,
                scene_id_map=scene_id_map,
                las_in=las_in,
            )
            out_chunk = _copy_to_output_chunk(
                las_in=las_in,
                header=header,
                include_cls_id=include_cls_id,
                global_gid=global_gid,
            )
            writer.write_points(out_chunk.points)
            count = int(len(las_in.x))
            total_points += count
            print(
                f"[merge] {scene_name}: points={count} "
                f"scene_points={gid_stats['scene_points']} "
                f"pole_points={gid_stats['pole_points']} "
                f"invalid_points={gid_stats['invalid_points']}"
            )

    print(f"[done] scenes={len(entries)} total_points={total_points}")
    print(f"[done] output={output_path}")


if __name__ == "__main__":
    main()

