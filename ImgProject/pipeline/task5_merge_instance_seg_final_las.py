from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge scene-level fusion/*_instance_seg_final.las files into one "
            "benchmark-level LAS."
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


def _has_cls_id(path: Path) -> bool:
    with laspy.open(path) as reader:
        extra_dims = set(reader.header.point_format.extra_dimension_names)
    return "cls_id" in extra_dims


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

    if include_cls_id:
        extra_bytes_params = getattr(laspy, "ExtraBytesParams", None)
        if extra_bytes_params is not None:
            header.add_extra_dim(extra_bytes_params(name="cls_id", type=np.uint8))
    return header


def _copy_to_output_chunk(
    las_in: laspy.LasData,
    header: laspy.LasHeader,
    include_cls_id: bool,
) -> laspy.LasData:
    out = laspy.LasData(header)
    out.x = np.asarray(las_in.x)
    out.y = np.asarray(las_in.y)
    out.z = np.asarray(las_in.z)

    n = len(las_in.x)

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

    output_path = data_root / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (set --overwrite to replace)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_cls_id = any(_has_cls_id(path) for _, path in entries)
    header = _build_output_header(entries[0][1], include_cls_id=include_cls_id)

    total_points = 0
    with laspy.open(output_path, mode="w", header=header) as writer:
        for scene_name, in_path in entries:
            las_in = laspy.read(in_path)
            out_chunk = _copy_to_output_chunk(
                las_in=las_in,
                header=header,
                include_cls_id=include_cls_id,
            )
            writer.write_points(out_chunk.points)
            count = len(las_in.x)
            total_points += count
            print(f"[merge] {scene_name}: {count} points")

    print(f"[done] scenes={len(entries)} total_points={total_points}")
    print(f"[done] output={output_path}")


if __name__ == "__main__":
    main()
