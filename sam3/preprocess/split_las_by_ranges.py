import argparse
import csv
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import laspy
except ImportError:  # pragma: no cover
    laspy = None


@dataclass(frozen=True)
class RangeSpec:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


READ_CHUNK_SIZE = 1_000_000


def _build_compatible_output_header(reader_header: Any) -> Any:
    """
    Build a writer header compatible with point format / LAS version constraints.
    """
    output_header = reader_header.copy()
    point_format_id = int(output_header.point_format.id)
    file_version = str(output_header.version)

    from laspy.point import dims as las_dims

    if not las_dims.is_point_fmt_compatible_with_version(point_format_id, file_version):
        min_version = las_dims.preferred_file_version_for_point_format(point_format_id)
        output_header.version = laspy.header.Version.from_str(
            min_version
        )
    return output_header


def _safe_name(name: str) -> str:
    out = name.strip().replace(" ", "_")
    out = re.sub(r"[^0-9a-zA-Z_.-]+", "_", out)
    out = out.strip("._")
    return out or "range"


def load_range_specs(inline_ranges: list[str]) -> list[RangeSpec]:
    specs: list[RangeSpec] = []

    for i, text in enumerate(inline_ranges):
        # Formats:
        # 1) name:xmin,xmax,ymin,ymax
        # 2) xmin,xmax,ymin,ymax
        s = text.strip()
        if not s:
            continue
        if ":" in s:
            name_part, values_part = s.split(":", 1)
            name = _safe_name(name_part)
        else:
            name = f"range_inline_{i:03d}"
            values_part = s
        vals = [v.strip() for v in values_part.split(",")]
        if len(vals) != 4:
            raise ValueError(
                f"Invalid --range '{text}'. Expected 4 numbers: "
                f"'xmin,xmax,ymin,ymax' or 'name:xmin,xmax,ymin,ymax'."
            )
        x0, x1, y0, y1 = [float(v) for v in vals]
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
        specs.append(
            RangeSpec(
                name=name,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        )

    if not specs:
        raise ValueError("No ranges provided. Use --range and repeat it for multiple ranges.")
    return specs


def split_single_las_by_ranges(
    las_path: Path,
    output_dir: Path,
    ranges: list[RangeSpec],
    *,
    include_empty: bool = False,
) -> list[dict[str, Any]]:
    """
    Split one LAS into multiple LAS files by XY ranges.
    Reads by chunks to support very large LAS files.
    """
    if laspy is None:
        raise ImportError("laspy is required. Please install laspy first.")

    out_root = output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(las_path), mode="r") as reader:
        total_points = int(reader.header.point_count)
        
        # 调试：检查第一个 chunk 的 RGB 值
        first_chunk = next(reader.chunk_iterator(READ_CHUNK_SIZE))
        if hasattr(first_chunk, 'red'):
            print(f"RGB range - R: {first_chunk.red.min()}-{first_chunk.red.max()}, "
                  f"G: {first_chunk.green.min()}-{first_chunk.green.max()}, "
                  f"B: {first_chunk.blue.min()}-{first_chunk.blue.max()}")
        
        # 重新打开以重置迭代器
        reader.close()
        
    with laspy.open(str(las_path), mode="r") as reader:
        total_points = int(reader.header.point_count)
        counts = [0 for _ in ranges]
        out_paths = [out_root / f"{las_path.stem}_{r.name}.las" for r in ranges]
        writer_header = _build_compatible_output_header(reader.header)

        with ExitStack() as stack:
            writers = [
                stack.enter_context(
                    laspy.open(
                        str(out_paths[i]),
                        mode="w",
                        header=writer_header.copy(),
                    )
                )
                for i in range(len(ranges))
            ]

            pbar = tqdm(
                total=total_points,
                desc=f"Split {las_path.name}",
                unit="pt",
            )
            try:
                for points in reader.chunk_iterator(READ_CHUNK_SIZE):
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    for i, r in enumerate(ranges):
                        mask = (
                            (x >= r.x_min)
                            & (x <= r.x_max)
                            & (y >= r.y_min)
                            & (y <= r.y_max)
                        )
                        n = int(np.count_nonzero(mask))
                        if n == 0:
                            continue
                        writers[i].write_points(points[mask])
                        counts[i] += n
                    pbar.update(len(points))
            finally:
                pbar.close()

    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ranges):
        out_path = out_paths[i]
        n = counts[i]
        if n == 0 and not include_empty:
            if out_path.exists():
                out_path.unlink()
            continue
        rows.append(
            {
                "source_las": str(las_path),
                "output_las": str(out_path),
                "range_name": r.name,
                "x_min": r.x_min,
                "x_max": r.x_max,
                "y_min": r.y_min,
                "y_max": r.y_max,
            }
        )
    return rows


def run_split_las_by_ranges(
    input_las: Path,
    output_dir: Path,
    ranges: list[RangeSpec],
    *,
    include_empty: bool,
    summary_csv: str,
) -> None:
    if not input_las.exists() or not input_las.is_file():
        raise FileNotFoundError(f"Input LAS not found: {input_las}")
    if input_las.suffix.lower() != ".las":
        raise ValueError(f"Input file must be .las: {input_las}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = split_single_las_by_ranges(
        input_las,
        output_dir,
        ranges,
        include_empty=include_empty,
    )

    summary_path = output_dir / summary_csv
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_las",
                "output_las",
                "range_name",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Done. Summary saved to: {summary_path}")
    print(f"Output LAS files: {len(all_rows)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one LAS file by multiple XY ranges and save each range as a separate LAS."
        )
    )
    parser.add_argument(
        "input_las",
        type=str,
        help="Input .las file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--range",
        dest="inline_ranges",
        action="append",
        default=[],
        help=(
            "Inline range. Format: "
            "'xmin,xmax,ymin,ymax' or 'name:xmin,xmax,ymin,ymax'. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep empty output LAS files (0 points). By default empty files are removed.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="split_las_summary.csv",
        help="Summary CSV filename in output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if laspy is None:
        raise ImportError("laspy is required for LAS processing. Please install it first.")

    input_las = Path(args.input_las).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    ranges = load_range_specs(list(args.inline_ranges))

    run_split_las_by_ranges(
        input_las=input_las,
        output_dir=output_dir,
        ranges=ranges,
        include_empty=bool(args.include_empty),
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()
