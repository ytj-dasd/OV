from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm


DEFAULT_CONCEPTS = ("arrow", "lane_line", "manhole")
DEFAULT_CLASS_ID = {
    "arrow": 16,
    "lane_line": 17,
    "manhole": 18,
}
DEFAULT_RGB8 = {
    "arrow": (190, 212, 0),
    "lane_line": (190, 212, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SAM prompt_results masks (pixel space) into a standalone LAS point cloud "
            "using las_positions.txt and BEV resolution."
        )
    )
    parser.add_argument(
        "--prompt-results-dir",
        required=True,
        help="Folder containing concept subfolders (e.g. arrow/lane_line/manhole).",
    )
    parser.add_argument(
        "--las-positions",
        type=str,
        default=None,
        help="Path to las_positions.txt. Default: <prompt-results-dir>/las_positions.txt",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=list(DEFAULT_CONCEPTS),
        help="Concept subfolder names to convert.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.02,
        help="BEV resolution in meters/pixel.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=3.5,
        help="Uniform Z value for all exported points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for per-instance random colors (used for manhole).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output LAS path.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite output LAS if it already exists.",
    )
    return parser.parse_args()


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"Expected masks shape (N,1,H,W), got {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks shape (N,H,W), got {arr.shape}")
    return arr.astype(bool, copy=False)


def _extract_image_stem(npz_path: Path, data: np.lib.npyio.NpzFile, concept_name: str) -> str:
    if "image_stem" in data.files:
        raw = np.asarray(data["image_stem"])
        if raw.shape == ():
            stem = str(raw.item())
            if stem:
                return stem

    stem = npz_path.stem
    if stem.endswith("_merged"):
        stem = stem[: -len("_merged")]
    tail = f"_{concept_name}"
    if stem.endswith(tail):
        stem = stem[: -len(tail)]
    return stem


def _parse_las_positions(path: Path) -> dict[str, tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"las_positions not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"las_positions is empty: {path}")

    start_idx = 0
    if lines[0].split()[0].lower() == "las_name":
        start_idx = 1

    mapping: dict[str, tuple[float, float, float]] = {}
    for line in lines[start_idx:]:
        cols = line.split()
        if len(cols) < 4:
            continue
        stem = Path(cols[0]).stem
        mapping[stem] = (float(cols[1]), float(cols[2]), float(cols[3]))
    if not mapping:
        raise ValueError(f"No valid rows in las_positions: {path}")
    return mapping


def _rgb8_to_rgb16(rgb8: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(rgb8, dtype=np.uint16).reshape(3)
    return int(arr[0] * 256), int(arr[1] * 256), int(arr[2] * 256)


def _write_points(
    writer: laspy.LasWriter,
    header: laspy.LasHeader,
    x: np.ndarray,
    y: np.ndarray,
    *,
    z: float,
    rgb8: tuple[int, int, int],
    class_id: int,
) -> None:
    n = int(x.shape[0])
    if n <= 0:
        return

    rec = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    rec.x = x
    rec.y = y
    rec.z = np.full((n,), float(z), dtype=np.float64)

    r16, g16, b16 = _rgb8_to_rgb16(rgb8)
    rec.red = np.full((n,), r16, dtype=np.uint16)
    rec.green = np.full((n,), g16, dtype=np.uint16)
    rec.blue = np.full((n,), b16, dtype=np.uint16)
    rec.classification = np.full((n,), int(class_id), dtype=np.uint8)

    writer.write_points(rec)


def main() -> None:
    args = parse_args()

    prompt_root = Path(args.prompt_results_dir).expanduser().resolve()
    if not prompt_root.is_dir():
        raise NotADirectoryError(f"prompt_results dir not found: {prompt_root}")

    las_positions_path = (
        Path(args.las_positions).expanduser().resolve()
        if args.las_positions
        else (prompt_root / "las_positions.txt")
    )
    positions = _parse_las_positions(las_positions_path)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path} (set --overwrite to replace)")

    if args.resolution <= 0:
        raise ValueError("--resolution must be > 0")

    rng = np.random.default_rng(int(args.seed))

    # LAS point format 3 includes RGB + classification.
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    total_points = 0
    total_instances = 0
    missing_position_stems: dict[str, int] = {}
    per_concept_points: dict[str, int] = {}
    per_concept_instances: dict[str, int] = {}

    with laspy.open(output_path, mode="w", header=header) as writer:
        for concept in args.concepts:
            concept_dir = prompt_root / concept
            npz_files = sorted(p for p in concept_dir.glob("*.npz") if p.is_file())
            if not npz_files:
                print(f"[warn] no npz files for concept: {concept_dir}")
                continue

            class_id = int(DEFAULT_CLASS_ID.get(concept, 31))
            concept_points = 0
            concept_instances = 0

            for npz_path in tqdm(npz_files, desc=f"Convert [{concept}]", unit="file"):
                data = np.load(npz_path, allow_pickle=True)
                if "masks" not in data.files:
                    continue

                image_stem = _extract_image_stem(npz_path, data, concept_name=concept)
                if image_stem not in positions:
                    missing_position_stems[image_stem] = missing_position_stems.get(image_stem, 0) + 1
                    continue

                left_top_x, left_top_y, _ = positions[image_stem]
                masks = _normalize_masks(data["masks"])
                if masks.shape[0] == 0:
                    continue

                for mask in masks:
                    ys, xs = np.nonzero(mask)
                    if xs.size == 0:
                        continue

                    x_world = float(left_top_x) + xs.astype(np.float64, copy=False) * float(args.resolution)
                    y_world = float(left_top_y) - ys.astype(np.float64, copy=False) * float(args.resolution)

                    if concept == "manhole":
                        rgb = tuple(int(v) for v in rng.integers(0, 256, size=3, dtype=np.uint8))
                        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
                            rgb = (255, 255, 255)
                    else:
                        rgb = DEFAULT_RGB8.get(concept, (0, 255, 255))

                    _write_points(
                        writer=writer,
                        header=header,
                        x=x_world,
                        y=y_world,
                        z=float(args.z),
                        rgb8=rgb,
                        class_id=class_id,
                    )

                    n_pts = int(x_world.shape[0])
                    concept_points += n_pts
                    concept_instances += 1
                    total_points += n_pts
                    total_instances += 1

            per_concept_points[concept] = concept_points
            per_concept_instances[concept] = concept_instances
            print(
                f"[concept] {concept}: instances={concept_instances} points={concept_points} class_id={class_id}"
            )

    print(f"[done] output={output_path}")
    print(f"[done] total_instances={total_instances} total_points={total_points} z={float(args.z):.3f}")
    if missing_position_stems:
        missing_items = sorted(missing_position_stems.items())
        print(f"[warn] stems missing in las_positions: {len(missing_items)}")
        for stem, count in missing_items[:20]:
            print(f"[warn]  - {stem}: {count} files")
        if len(missing_items) > 20:
            print(f"[warn]  ... {len(missing_items) - 20} more")


if __name__ == "__main__":
    main()
