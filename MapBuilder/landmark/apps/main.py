"""Full landmark pipeline from point cloud to target SHP outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from landmark.apps.crosswalk import run_crosswalk
from landmark.apps.laneline import run_laneline
from landmark.apps.road_arrow import run_road_arrow
from landmark.tools.pc_process.pre_part import run_pre_part
from landmark.tools.pc_process.part.part import split_ply_by_parts
from landmark.tools.pc_process.part.tile_part import write_tile_parts_json
from landmark.tools.sam3.fuse_masks import fuse_cross_mode_masks
from landmark.tools.sam3.instance_seg import (
    DEFAULT_CONDA_ENV,
    DEFAULT_SAM3_DIR,
    run_sam3_pipeline,
)

_DEFAULT_OUTPUT_DIR = Path("outputs/apps")
_DEFAULT_VERTICES = Path("asserts/arrow_line/arrow_vertices.json")
_CHECKPOINT_CHOICES = (
    "pre-part",
    "tile-part",
    "part",
    "sam3",
    "masks",
    "road-arrow",
    "laneline",
    "crosswalk",
)
_TO_SHP_MASK_SOURCE_CHOICES = ("fused", "intensity")


def _auto_discover(base: Path) -> dict[str, Path | None]:
    discovered: dict[str, Path | None] = {}
    v = base / _DEFAULT_VERTICES
    discovered["vertices_path"] = v if v.is_file() else None
    return discovered


def _find_any_summary(objs_dir: Path) -> Path:
    for p in objs_dir.rglob("summary.json"):
        return p
    raise FileNotFoundError(f"No summary.json found under {objs_dir}")


def _normalize_checkpoint_name(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).strip().lower().replace("_", "-")


def _stop_if_requested(
    *,
    checkpoint: str,
    stop_after: str | None,
    results: dict[str, Path | str],
) -> bool:
    normalized = _normalize_checkpoint_name(stop_after)
    if normalized != checkpoint:
        return False
    results["stopped_after"] = checkpoint
    return True


def run_pipeline(
    ply_path: Path | str,
    *,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    parts_json_path: Path | str | None = None,
    tile_size_m: float = 40.0,
    fill_ratio_threshold: float = 0.10,
    fill_cell_size_m: float = 0.50,
    mpp: float = 0.02,
    modes: list[str] | None = None,
    sam3_dir: str | None = None,
    conda_env: str | None = None,
    force: bool = False,
    vertices_path: Path | None = None,
    stop_after: str | None = None,
    to_shp_mask_source: str = "fused",
) -> dict[str, Path | str]:
    if modes is None:
        modes = ["rgb", "intensity"]
    if sam3_dir is None:
        sam3_dir = str(DEFAULT_SAM3_DIR)
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV

    ply_path = Path(ply_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_part_dir = output_dir / "pre-part"
    parts_dir = output_dir / "parts"
    objs_dir = output_dir / "objs"
    masks_dir = output_dir / "masks"
    shp_dir = output_dir / "shp"
    parts_path = (
        Path(parts_json_path).expanduser()
        if parts_json_path is not None
        else pre_part_dir / "parts.json"
    )

    stop_after = _normalize_checkpoint_name(stop_after)
    if stop_after is not None and stop_after not in _CHECKPOINT_CHOICES:
        raise ValueError(
            f"Unsupported stop_after={stop_after!r}; expected one of {_CHECKPOINT_CHOICES}"
        )
    if to_shp_mask_source not in _TO_SHP_MASK_SOURCE_CHOICES:
        raise ValueError(
            "Unsupported to_shp_mask_source="
            f"{to_shp_mask_source!r}; expected one of {_TO_SHP_MASK_SOURCE_CHOICES}"
        )

    results: dict[str, Path | str] = {}

    pre_part_ran = force or not (pre_part_dir / "geo_meta.json").is_file()
    if pre_part_ran:
        run_pre_part(ply_path, pre_part_dir, mpp=0.08, mode="rgb")
    results["pre_part_dir"] = pre_part_dir
    if _stop_if_requested(checkpoint="pre-part", stop_after=stop_after, results=results):
        return results

    if not pre_part_ran and (force or not parts_path.is_file()):
        if parts_json_path is None:
            write_tile_parts_json(
                ply_path,
                parts_path,
                tile_size_m=tile_size_m,
                fill_ratio_threshold=fill_ratio_threshold,
                fill_cell_size_m=fill_cell_size_m,
                mask_bev_path=pre_part_dir / "bev_mask.png",
                geo_meta_path=pre_part_dir / "geo_meta.json",
                preview_path=pre_part_dir / "parts_preview.png",
            )
    results["parts_json"] = parts_path
    if parts_json_path is None:
        results["parts_preview_png"] = pre_part_dir / "parts_preview.png"
    if _stop_if_requested(checkpoint="tile-part", stop_after=stop_after, results=results):
        return results

    geo_meta_path = parts_dir / "geo_meta.json"
    if force or not geo_meta_path.is_file():
        written = split_ply_by_parts(
            ply_path,
            parts_path,
            parts_dir,
            render_bev=True,
            mpp=mpp,
        )
        print(f"[main] wrote {len(written)} parts to {parts_dir}", flush=True)
    results["parts_dir"] = parts_dir
    if _stop_if_requested(checkpoint="part", stop_after=stop_after, results=results):
        return results

    has_objs = objs_dir.is_dir() and any(objs_dir.rglob("summary.json"))
    if force or not has_objs:
        run_sam3_pipeline(
            input_dir=parts_dir,
            output_dir=output_dir,
            modes=modes,
            sam3_dir=sam3_dir,
            conda_env=conda_env,
        )
    results["objs_dir"] = objs_dir
    if _stop_if_requested(checkpoint="sam3", stop_after=stop_after, results=results):
        return results

    masks_dir.mkdir(parents=True, exist_ok=True)
    for prompt_dir in ("arrow", "laneline", "crosswalk"):
        if to_shp_mask_source == "fused":
            fuse_cross_mode_masks(objs_dir, prompt_dir, modes)
            label_map_path = objs_dir / prompt_dir / "fused" / "final_masks.npy"
        else:
            label_map_path = objs_dir / prompt_dir / "intensity" / "final_masks.npy"
        if not label_map_path.is_file():
            raise FileNotFoundError(
                f"Missing label_map for prompt={prompt_dir!r} with source={to_shp_mask_source!r}: "
                f"{label_map_path}"
            )
        label_map = np.load(label_map_path)
        np.save(masks_dir / f"{prompt_dir}_label_map.npy", np.asarray(label_map, dtype=np.int32))
    results["masks_dir"] = masks_dir
    results["to_shp_mask_source"] = to_shp_mask_source
    if _stop_if_requested(checkpoint="masks", stop_after=stop_after, results=results):
        return results

    summary_path = _find_any_summary(objs_dir)
    arrow_shp = run_road_arrow(
        masks_dir / "arrow_label_map.npy",
        summary_path,
        shp_dir / "road_arrow",
        ply_path=ply_path,
        vertices_path=vertices_path,
    )
    results["road_arrow_shp"] = arrow_shp
    if _stop_if_requested(checkpoint="road-arrow", stop_after=stop_after, results=results):
        return results

    laneline_shp = run_laneline(
        masks_dir / "laneline_label_map.npy",
        summary_path,
        ply_path,
        shp_dir / "laneline",
    )
    results["laneline_shp"] = laneline_shp
    if _stop_if_requested(checkpoint="laneline", stop_after=stop_after, results=results):
        return results

    crosswalk_shp = run_crosswalk(
        masks_dir / "crosswalk_label_map.npy",
        summary_path,
        laneline_shp,
        shp_dir / "crosswalk",
    )
    results["crosswalk_shp"] = crosswalk_shp
    if _stop_if_requested(checkpoint="crosswalk", stop_after=stop_after, results=results):
        return results

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full point-cloud to SHP pipeline.")
    parser.add_argument("ply_path", help="Input PLY path.")
    parser.add_argument("--parts-json", default=None, help="Optional existing parts.json path.")
    parser.add_argument("--tile-size", type=float, default=40.0, help="Tile size in meters when auto-generating parts.")
    parser.add_argument("--fill-threshold", type=float, default=0.10, help="Minimum occupancy ratio for auto-generated parts.")
    parser.add_argument("--fill-cell-size", type=float, default=0.50, help="Occupancy cell size for auto-generated parts.")
    parser.add_argument("--out", default=str(_DEFAULT_OUTPUT_DIR), help="Output root directory.")
    parser.add_argument("--mpp", type=float, default=0.02, help="Meters per pixel for BEV generation.")
    parser.add_argument(
        "--to-shp-mask-source",
        default="fused",
        choices=_TO_SHP_MASK_SOURCE_CHOICES,
        help="Choose whether to use fused SAM3 label maps or only intensity SAM3 label maps for to_shp.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run all stages.")
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=_CHECKPOINT_CHOICES,
        help="Stop after a named checkpoint and only output current-stage artifacts.",
    )
    args = parser.parse_args()

    auto = _auto_discover(Path.cwd())
    results = run_pipeline(
        args.ply_path,
        output_dir=args.out,
        parts_json_path=args.parts_json,
        tile_size_m=args.tile_size,
        fill_ratio_threshold=args.fill_threshold,
        fill_cell_size_m=args.fill_cell_size,
        mpp=args.mpp,
        force=args.force,
        vertices_path=auto["vertices_path"],
        stop_after=args.stop_after,
        to_shp_mask_source=args.to_shp_mask_source,
    )
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
