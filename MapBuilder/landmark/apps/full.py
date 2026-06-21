"""Spec-driven full pipeline from one point cloud to map vector outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapefile
from plyfile import PlyData, PlyElement

from landmark.apps.crosswalk import run_crosswalk
from landmark.apps.laneline import filter_laneline_shp_by_crosswalk, run_laneline
from landmark.apps.manhole import DEFAULT_MANIFEST as DEFAULT_MANHOLE_MANIFEST
from landmark.apps.manhole import run_manhole
from landmark.apps.road_arrow import run_road_arrow
from landmark.apps.sidewalk_v2 import run_sidewalk_v2
from landmark.tools.pc_process.pre_part import run_pre_part
from landmark.tools.sam3.instance_seg_v2 import DEFAULT_CONDA_ENV, DEFAULT_SAM3_DIR, run_instance_seg_v2
from landmark.tools.to_shp.geometry import pixel_to_xy

_DEFAULT_OUTPUT_DIR = Path("outputs/full")
_DEFAULT_VERTICES = Path("asserts/arrow_line/arrow_vertices.json")
_CHECKPOINT_CHOICES = (
    "pre-part",
    "instance-seg-v2",
    "vectorize",
    "manhole",
)
_TARGETS: tuple[dict[str, str | float], ...] = (
    {"name": "road", "prompt": "road", "image": "rgb", "score_th": 0.2},
    {"name": "sidewalk", "prompt": "sidewalk", "image": "rgb", "score_th": 0.2},
    {"name": "green_vege", "prompt": "green vege", "image": "rgb", "score_th": 0.2},
    {"name": "laneline", "prompt": "lane line", "image": "intensity", "score_th": 0.5},
    {"name": "road_marking", "prompt": "road marking", "image": "intensity", "score_th": 0.5},
    {"name": "crosswalk", "prompt": "crosswalk", "image": "intensity", "score_th": 0.5},
    {"name": "arrow", "prompt": "arrow", "image": "intensity", "score_th": 0.5},
)
_REQUIRED_PLY_FIELDS = ("x", "y", "z", "red", "green", "blue", "scalar_Intensity")
_REQUIRED_LAS_FIELDS = ("x", "y", "z", "red", "green", "blue", "intensity")
_CONVERTED_INPUT_VERSION = 4
_CONVERTED_INPUT_NAME = "input_converted.ply"
_CONVERTED_INPUT_META_NAME = "input_converted.meta.json"


def _auto_discover(base: Path) -> dict[str, Path | None]:
    vertices_path = base / _DEFAULT_VERTICES
    return {"vertices_path": vertices_path if vertices_path.is_file() else None}


def _normalize_checkpoint_name(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).strip().lower().replace("_", "-")


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_path": str(path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _format_missing_fields_error(
    path: Path,
    *,
    input_type: str,
    missing: list[str],
    required: tuple[str, ...],
) -> ValueError:
    return ValueError(
        f"{input_type} point cloud is missing required fields: {missing}; "
        f"input={path}; required={list(required)}"
    )


def _validate_ply_fields(ply_path: Path) -> None:
    ply = PlyData.read(str(ply_path), mmap=True)
    if "vertex" not in ply:
        raise ValueError(f"PLY point cloud is missing 'vertex' element: input={ply_path}")
    names = set(ply["vertex"].data.dtype.names or ())
    missing = [name for name in _REQUIRED_PLY_FIELDS if name not in names]
    if missing:
        raise _format_missing_fields_error(
            ply_path,
            input_type="PLY",
            missing=missing,
            required=_REQUIRED_PLY_FIELDS,
        )


def _las_field_names(las: Any) -> set[str]:
    names = {str(name).lower() for name in las.point_format.dimension_names}
    names.update({"x", "y", "z"})
    return names


def _validate_las_fields(las_path: Path, las: Any) -> None:
    names = _las_field_names(las)
    missing = [name for name in _REQUIRED_LAS_FIELDS if name.lower() not in names]
    if missing:
        raise _format_missing_fields_error(
            las_path,
            input_type="LAS/LAZ",
            missing=missing,
            required=_REQUIRED_LAS_FIELDS,
        )


def _converted_input_meta_matches(meta_path: Path, source_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = _load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return False
    expected = _source_signature(source_path)
    return (
        meta.get("conversion_version") == _CONVERTED_INPUT_VERSION
        and all(meta.get(key) == value for key, value in expected.items())
    )


def _las_to_full_ply(las_path: Path, output_path: Path) -> Path:
    import laspy

    las = laspy.read(str(las_path))
    _validate_las_fields(las_path, las)
    rgb_channels = [np.asarray(getattr(las, name), dtype=np.uint16) for name in ("red", "green", "blue")]
    rgb_uses_16_bit_range = any(channel.size and int(channel.max()) > 255 for channel in rgb_channels)
    vertex = np.empty(
        len(las.x),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("scalar_Intensity", "u2"),
        ],
    )
    vertex["x"] = np.asarray(las.x, dtype=np.float32)
    vertex["y"] = np.asarray(las.y, dtype=np.float32)
    vertex["z"] = np.asarray(las.z, dtype=np.float32)
    for name, channel in zip(("red", "green", "blue"), rgb_channels, strict=True):
        vertex[name] = channel // 256 if rgb_uses_16_bit_range else channel
    vertex["scalar_Intensity"] = np.asarray(las.intensity, dtype=np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))
    _validate_ply_fields(output_path)
    return output_path


def _prepare_full_input_ply(input_path: Path, pre_part_dir: Path, *, force: bool) -> Path:
    suffix = input_path.suffix.lower()
    if suffix == ".ply":
        _validate_ply_fields(input_path)
        return input_path
    if suffix not in {".las", ".laz"}:
        raise ValueError(f"Unsupported point cloud input type: {input_path}. Expected .ply, .las, or .laz")

    converted_path = pre_part_dir / _CONVERTED_INPUT_NAME
    converted_meta_path = pre_part_dir / _CONVERTED_INPUT_META_NAME
    if not force and converted_path.is_file() and _converted_input_meta_matches(converted_meta_path, input_path):
        _validate_ply_fields(converted_path)
        return converted_path

    converted = _las_to_full_ply(input_path, converted_path)
    converted_meta = {
        **_source_signature(input_path),
        "conversion_version": _CONVERTED_INPUT_VERSION,
        "converted_path": str(converted),
        "required_las_fields": list(_REQUIRED_LAS_FIELDS),
        "required_ply_fields": list(_REQUIRED_PLY_FIELDS),
    }
    converted_meta_path.write_text(json.dumps(converted_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return converted


def _stop_if_requested(
    *,
    checkpoint: str,
    stop_after: str | None,
    results: dict[str, Path | str],
) -> bool:
    if _normalize_checkpoint_name(stop_after) != checkpoint:
        return False
    results["stopped_after"] = checkpoint
    return True


def _pc_csf_dir(pre_part_dir: Path) -> Path:
    return pre_part_dir / "bev_pc_csf"


def _pre_part_assets(pre_part_dir: Path) -> dict[str, Path]:
    pc_csf_dir = _pc_csf_dir(pre_part_dir)
    return {
        "rgb": pc_csf_dir / "bev_pc_csf_rgb_filled.png",
        "intensity": pc_csf_dir / "bev_pc_csf_intensity.png",
        "height_png": pc_csf_dir / "bev_pc_csf_height.png",
        "height_values": pc_csf_dir / "bev_pc_csf_height_values.npy",
        "height_meta": pc_csf_dir / "bev_pc_csf_height_meta.json",
        "geo_meta": pc_csf_dir / "pc_csf_geo_meta.json",
    }


def _pre_part_ready(pre_part_dir: Path) -> bool:
    return all(path.is_file() for path in _pre_part_assets(pre_part_dir).values())


def _target_result_dir(objs_dir: Path, target_name: str) -> Path:
    return objs_dir / target_name / "result"


def _target_label_map_path(objs_dir: Path, target_name: str) -> Path:
    return _target_result_dir(objs_dir, target_name) / "label_map.npy"


def _target_ready(objs_dir: Path, target_name: str) -> bool:
    return _target_label_map_path(objs_dir, target_name).is_file()


def _normalize_target_outputs(objs_dir: Path, target_name: str, *, force: bool = False) -> dict[str, Path]:
    result_dir = _target_result_dir(objs_dir, target_name)
    label_map_path = result_dir / "label_map.npy"
    if not label_map_path.is_file():
        raise FileNotFoundError(label_map_path)

    objs_png = result_dir / "objs.png"
    vis_png = result_dir / "vis.png"
    if objs_png.is_file() and (force or not vis_png.is_file()):
        shutil.copyfile(objs_png, vis_png)
    return {"label_map": label_map_path, "vis": vis_png}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_label_map_polygon_shp(
    label_map_path: Path,
    geo_meta_path: Path,
    output_dir: Path,
    *,
    shp_stem: str,
    min_area_px: int = 4,
) -> Path:
    label_map = np.load(label_map_path, mmap_mode="r")
    if label_map.ndim != 2:
        raise ValueError(f"label_map must have shape (H,W), got {label_map.shape}: {label_map_path}")
    meta = _load_json(geo_meta_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shp_base = output_dir / shp_stem

    writer = shapefile.Writer(str(shp_base))
    writer.shapeType = shapefile.POLYGON
    writer.field("id", "N", decimal=0)
    writer.field("area_px", "N", decimal=0)

    feature_count = 0
    ids = [int(v) for v in np.unique(label_map) if int(v) >= 0]
    for label_id in ids:
        mask = np.asarray(label_map == label_id, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area_px = int(round(float(cv2.contourArea(contour))))
            if area_px < int(min_area_px):
                continue
            approx = cv2.approxPolyDP(contour, epsilon=1.0, closed=True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            xy = pixel_to_xy(approx.astype(np.float32), meta).astype(np.float64)
            ring = [[float(x), float(y)] for x, y in xy]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            writer.poly([ring])
            writer.record(id=int(label_id), area_px=area_px)
            feature_count += 1

    writer.close()
    shp_path = shp_base.with_suffix(".shp")
    summary = {
        "label_map": str(label_map_path),
        "geo_meta": str(geo_meta_path),
        "shp": str(shp_path),
        "feature_count": int(feature_count),
        "min_area_px": int(min_area_px),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return shp_path


def _copy_shapefile_to_product(shp_path: Path | str, product_dir: Path, *, stem: str) -> Path:
    source = Path(shp_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    product_dir.mkdir(parents=True, exist_ok=True)
    target_shp = product_dir / f"{stem}.shp"
    for sidecar in source.parent.glob(f"{source.stem}.*"):
        shutil.copyfile(sidecar, product_dir / f"{stem}{sidecar.suffix}")
    return target_shp


def _write_product_outputs(results: dict[str, Path | str], product_dir: Path) -> dict[str, Any]:
    product_specs = (
        ("sidewalk", "sidewalk_shp"),
        ("belt", "belt_shp"),
        ("arrow", "arrow_shp"),
        ("crosswalk", "crosswalk_shp"),
        ("laneline", "laneline_shp"),
        ("laneline_centerline", "laneline_centerline_shp"),
        ("manhole", "manhole_shp"),
    )
    copied: dict[str, str] = {}
    missing: list[str] = []
    for product_name, result_key in product_specs:
        shp_path = results.get(result_key)
        if shp_path is None:
            missing.append(product_name)
            continue
        try:
            copied[product_name] = str(_copy_shapefile_to_product(shp_path, product_dir, stem=product_name))
        except FileNotFoundError:
            missing.append(product_name)

    summary = {
        "product_dir": str(product_dir),
        "copied": copied,
        "missing": missing,
    }
    product_dir.mkdir(parents=True, exist_ok=True)
    (product_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _run_instance_seg_targets(
    *,
    assets: dict[str, Path],
    objs_dir: Path,
    tile_size_m: float,
    fill_ratio_threshold: float,
    tile_overlap_ratio: float,
    sam3_dir: str | Path,
    conda_env: str,
    th: float | None,
    score_th: float | None,
    force: bool,
) -> dict[str, Path]:
    label_maps: dict[str, Path] = {}
    for target in _TARGETS:
        target_name = str(target["name"])
        prompt = str(target["prompt"])
        image_key = str(target["image"])
        target_score_th = float(target["score_th"]) if score_th is None else score_th
        target_dir = objs_dir / target_name
        if force or not _target_ready(objs_dir, target_name):
            run_instance_seg_v2(
                assets[image_key],
                prompt,
                assets["geo_meta"],
                tile_size_m,
                target_dir,
                spt_road_path=None,
                sam3_dir=sam3_dir,
                conda_env=conda_env,
                th=th,
                score_th=target_score_th,
                fill_ratio_threshold=fill_ratio_threshold,
                tile_overlap_ratio=tile_overlap_ratio,
            )
        outputs = _normalize_target_outputs(objs_dir, target_name, force=force)
        label_maps[target_name] = outputs["label_map"]
    return label_maps


def run_full_pipeline(
    ply_path: Path | str,
    *,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    tile_size_m: float = 40.0,
    fill_ratio_threshold: float = 0.10,
    fill_cell_size_m: float = 0.50,
    sam3_dir: str | Path | None = None,
    conda_env: str | None = None,
    force: bool = False,
    vertices_path: Path | str | None = None,
    stop_after: str | None = None,
    th: float | None = 0.5,
    score_th: float | None = None,
    tile_overlap_ratio: float = 0.10,
    laneline_box_debug: bool = False,
    manhole_samples: Path | str = DEFAULT_MANHOLE_MANIFEST,
    manhole_score_th: float = 0.5,
    manhole_iou_th: float = 0.1,
    manhole_min_radius_m: float = 0.15,
    manhole_max_radius_m: float = 1.20,
    manhole_circle_points: int = 64,
    disable_manhole: bool = False,
) -> dict[str, Path | str]:
    """Run the full pipeline described by docs/full-plan.md."""
    if sam3_dir is None:
        sam3_dir = DEFAULT_SAM3_DIR
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV

    stop_after = _normalize_checkpoint_name(stop_after)
    if stop_after is not None and stop_after not in _CHECKPOINT_CHOICES:
        raise ValueError(f"Unsupported stop_after={stop_after!r}; expected one of {_CHECKPOINT_CHOICES}")

    input_path = Path(ply_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_part_dir = output_dir / "pre-part"
    objs_dir = output_dir / "objs"
    shp_dir = output_dir / "shp"
    product_dir = output_dir / "product"
    results: dict[str, Path | str] = {"output_dir": output_dir}
    pipeline_ply_path = _prepare_full_input_ply(input_path, pre_part_dir, force=force)
    results["input_source_path"] = input_path
    results["input_ply_path"] = pipeline_ply_path

    pre_part_ready = _pre_part_ready(pre_part_dir)
    pre_part_skipped = bool(pre_part_ready and not force)
    if force or not pre_part_ready:
        run_pre_part(
            pipeline_ply_path,
            pre_part_dir,
            mpp=0.02,
            mode="rgb",
            tile_size_m=tile_size_m,
            fill_ratio_threshold=fill_ratio_threshold,
            fill_cell_size_m=fill_cell_size_m,
        )
    if not _pre_part_ready(pre_part_dir):
        missing = [str(path) for path in _pre_part_assets(pre_part_dir).values() if not path.is_file()]
        raise FileNotFoundError(f"Missing pre-part assets: {missing}")
    assets = _pre_part_assets(pre_part_dir)
    results["pre_part_dir"] = pre_part_dir
    results["pre_part_skipped"] = str(pre_part_skipped)
    results.update({f"pre_part_{key}": value for key, value in assets.items()})
    if _stop_if_requested(checkpoint="pre-part", stop_after=stop_after, results=results):
        return results

    label_maps = _run_instance_seg_targets(
        assets=assets,
        objs_dir=objs_dir,
        tile_size_m=tile_size_m,
        fill_ratio_threshold=fill_ratio_threshold,
        tile_overlap_ratio=tile_overlap_ratio,
        sam3_dir=sam3_dir,
        conda_env=conda_env,
        th=th,
        score_th=score_th,
        force=force,
    )
    results["objs_dir"] = objs_dir
    for target_name, label_map_path in label_maps.items():
        results[f"{target_name}_label_map"] = label_map_path
    if _stop_if_requested(checkpoint="instance-seg-v2", stop_after=stop_after, results=results):
        return results

    road_shp = _write_label_map_polygon_shp(
        label_maps["road"],
        assets["geo_meta"],
        shp_dir / "road",
        shp_stem="road",
    )
    results["road_shp"] = road_shp

    green_shp = _write_label_map_polygon_shp(
        label_maps["green_vege"],
        assets["geo_meta"],
        shp_dir / "green_vege",
        shp_stem="green_vege",
    )
    results["green_vege_shp"] = green_shp

    sidewalk_outputs = run_sidewalk_v2(
        pre_part_dir,
        shp_dir / "sidewalk",
        label_map_path=label_maps["sidewalk"],
        road_label_map_path=label_maps["road"],
        green_veg_label_map_path=label_maps["green_vege"],
    )
    results["sidewalk_shp"] = sidewalk_outputs["sidewalk_boundary"]

    arrow_shp = run_road_arrow(
        label_maps["arrow"],
        assets["geo_meta"],
        shp_dir / "arrow",
        ply_path=None,
        vertices_path=vertices_path,
        min_mask_area=2000,
        max_match_score=8.0,
        max_overflow=0.70,
    )
    results["arrow_shp"] = arrow_shp

    laneline_shp = run_laneline(
        label_maps["laneline"],
        assets["geo_meta"],
        pipeline_ply_path,
        shp_dir / "laneline",
        road_marking_label_map_path=label_maps["road_marking"],
        arrow_label_map_path=label_maps["arrow"],
        box_debug=laneline_box_debug,
    )
    results["laneline_shp"] = laneline_shp

    crosswalk_shp = run_crosswalk(
        label_maps["crosswalk"],
        assets["geo_meta"],
        laneline_shp,
        shp_dir / "crosswalk",
    )
    results["crosswalk_shp"] = crosswalk_shp
    laneline_shp = filter_laneline_shp_by_crosswalk(laneline_shp, crosswalk_shp)
    results["laneline_shp"] = laneline_shp
    laneline_centerline_shp = Path(laneline_shp).with_name("laneline_centerline.shp")
    if laneline_centerline_shp.is_file():
        results["laneline_centerline_shp"] = laneline_centerline_shp
    product_summary = _write_product_outputs(results, product_dir)
    results["product_dir"] = product_dir
    results["product_summary"] = product_dir / "summary.json"
    for product_name, product_path in product_summary["copied"].items():
        results[f"product_{product_name}_shp"] = Path(product_path)
    if _stop_if_requested(checkpoint="vectorize", stop_after=stop_after, results=results):
        return results

    if not disable_manhole:
        manhole_outputs = run_manhole(
            objs_dir / "road" / "parts" / "parts.json",
            manhole_samples,
            assets["rgb"],
            assets["geo_meta"],
            output_dir / "manhole",
            sam3_dir=sam3_dir,
            conda_env=conda_env,
            score_threshold=manhole_score_th,
            iou_threshold=manhole_iou_th,
            min_radius_m=manhole_min_radius_m,
            max_radius_m=manhole_max_radius_m,
            circle_points=manhole_circle_points,
            force=force,
        )
        results["manhole_label_map"] = manhole_outputs["label_map"]
        results["manhole_shp"] = manhole_outputs["shp"]
        product_summary = _write_product_outputs(results, product_dir)
        results["product_summary"] = product_dir / "summary.json"
        for product_name, product_path in product_summary["copied"].items():
            results[f"product_{product_name}_shp"] = Path(product_path)
    if _stop_if_requested(checkpoint="manhole", stop_after=stop_after, results=results):
        return results

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the spec-driven full point-cloud to SHP pipeline.")
    parser.add_argument("ply_path", help="Input point-cloud path (.ply, .las, or .laz).")
    parser.add_argument("--out", default=str(_DEFAULT_OUTPUT_DIR), help="Output root directory.")
    parser.add_argument("--tile-size", type=float, default=40.0, help="Physical tile size in meters for instance_seg_v2.")
    parser.add_argument("--fill-threshold", type=float, default=0.10, help="Minimum valid-pixel ratio for generated tiles.")
    parser.add_argument("--fill-cell-size", type=float, default=0.50, help="Occupancy cell size metadata for pre-part.")
    parser.add_argument("--sam3-dir", default=str(DEFAULT_SAM3_DIR), help="SAM3 repository directory.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="SAM3 conda env or python path.")
    parser.add_argument("--th", type=float, default=0.5, help="SAM3 mask threshold for instance_seg_v2.")
    parser.add_argument(
        "--score-th",
        type=float,
        default=None,
        help="Override SAM3 score threshold for all instance_seg_v2 targets. Defaults: terrain=0.2, instance=0.5.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.10, help="Tile overlap ratio for instance_seg_v2.")
    parser.add_argument("--laneline-box-debug", action="store_true", help="Skip short-box point-cloud refine for laneline.")
    parser.add_argument("--manhole-samples", default=str(DEFAULT_MANHOLE_MANIFEST), help="Manhole visual sample manifest.")
    parser.add_argument("--manhole-score-th", type=float, default=0.5, help="SAM3 visual retrieval confidence threshold.")
    parser.add_argument("--manhole-iou-th", type=float, default=0.1, help="Cross-part manhole overlap ratio threshold.")
    parser.add_argument("--manhole-min-radius-m", type=float, default=0.15, help="Minimum accepted manhole radius.")
    parser.add_argument("--manhole-max-radius-m", type=float, default=1.20, help="Maximum accepted manhole radius.")
    parser.add_argument("--manhole-circle-points", type=int, default=64, help="Sampled polygon vertices per manhole circle.")
    parser.add_argument("--disable-manhole", action="store_true", help="Skip the fourth manhole stage.")
    parser.add_argument("--force", action="store_true", help="Force re-run all stages.")
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=_CHECKPOINT_CHOICES,
        help="Stop after a named checkpoint and only output current-stage artifacts.",
    )
    args = parser.parse_args()

    auto = _auto_discover(Path.cwd())
    results = run_full_pipeline(
        args.ply_path,
        output_dir=args.out,
        tile_size_m=args.tile_size,
        fill_ratio_threshold=args.fill_threshold,
        fill_cell_size_m=args.fill_cell_size,
        sam3_dir=args.sam3_dir,
        conda_env=args.conda_env,
        force=args.force,
        vertices_path=auto["vertices_path"],
        stop_after=args.stop_after,
        th=args.th,
        score_th=args.score_th,
        tile_overlap_ratio=args.tile_overlap,
        laneline_box_debug=args.laneline_box_debug,
        manhole_samples=args.manhole_samples,
        manhole_score_th=args.manhole_score_th,
        manhole_iou_th=args.manhole_iou_th,
        manhole_min_radius_m=args.manhole_min_radius_m,
        manhole_max_radius_m=args.manhole_max_radius_m,
        manhole_circle_points=args.manhole_circle_points,
        disable_manhole=args.disable_manhole,
    )
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
