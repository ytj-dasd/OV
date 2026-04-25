from __future__ import annotations

import argparse
from pathlib import Path

import laspy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-scene LAS files by filename suffix. "
            "Default suffix matches files like road1-1_pole_groups_merged.las."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Benchmark root that contains scene folders.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_pole_groups_merged.las",
        help="Filename suffix to match in each scene fusion directory.",
    )
    parser.add_argument(
        "--scene-subdir",
        type=str,
        default="fusion",
        help="Subdirectory inside each scene directory to search LAS files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="pole_groups_merged_all.las",
        help="Output LAS filename written under data-root.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite output file when it already exists.",
    )
    return parser.parse_args()


def discover_scene_las_by_suffix(
    data_root: Path,
    *,
    suffix: str,
    scene_subdir: str,
) -> list[tuple[str, Path]]:
    scene_entries: list[tuple[str, Path]] = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        subdir = scene_dir / scene_subdir
        if not subdir.is_dir():
            continue

        preferred = subdir / f"{scene_dir.name}{suffix}"
        if preferred.exists():
            scene_entries.append((scene_dir.name, preferred))
            continue

        matches = sorted(subdir.glob(f"*{suffix}"))
        if not matches:
            continue
        scene_entries.append((scene_dir.name, matches[0]))
        if len(matches) > 1:
            print(
                f"[warn] {scene_dir.name}: found multiple '*{suffix}' files, "
                f"use {matches[0].name}"
            )
    return scene_entries


def _file_signature(path: Path) -> tuple[int, tuple[str, ...]]:
    with laspy.open(path) as reader:
        point_format_id = int(reader.header.point_format.id)
        extra_dims = tuple(reader.header.point_format.extra_dimension_names)
    return point_format_id, extra_dims


def _validate_compatible(entries: list[tuple[str, Path]]) -> None:
    if not entries:
        return
    base_scene, base_path = entries[0]
    base_sig = _file_signature(base_path)
    for scene_name, path in entries[1:]:
        sig = _file_signature(path)
        if sig != base_sig:
            raise ValueError(
                "Incompatible LAS schema for merge:\n"
                f"  base scene={base_scene} path={base_path.name} sig={base_sig}\n"
                f"  this scene={scene_name} path={path.name} sig={sig}"
            )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    suffix = str(args.suffix).strip()
    if not suffix:
        raise ValueError("--suffix cannot be empty.")
    if not suffix.lower().endswith(".las"):
        raise ValueError("--suffix must end with '.las'.")

    entries = discover_scene_las_by_suffix(
        data_root,
        suffix=suffix,
        scene_subdir=str(args.scene_subdir),
    )
    if not entries:
        raise FileNotFoundError(
            f"No LAS files found by suffix '*{suffix}' under {data_root}."
        )

    _validate_compatible(entries)

    output_path = data_root / str(args.output_name)
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Output exists: {output_path} (set --overwrite to replace)."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(entries[0][1]) as reader:
        header = reader.header

    total_points = 0
    with laspy.open(output_path, mode="w", header=header) as writer:
        for scene_name, las_path in entries:
            las_data = laspy.read(las_path)
            writer.write_points(las_data.points)
            count = int(len(las_data.x))
            total_points += count
            print(f"[merge] {scene_name}: points={count} file={las_path.name}")

    print(f"[done] scenes={len(entries)} total_points={total_points}")
    print(f"[done] output={output_path}")


if __name__ == "__main__":
    main()

