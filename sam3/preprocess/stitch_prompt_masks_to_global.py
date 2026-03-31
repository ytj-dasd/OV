import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class PositionInfo:
    stem: str
    left_top_x: float
    left_top_y: float
    left_top_z: float


@dataclass
class TileResult:
    prompt_key: str
    prompt_id: str
    stem: str
    masks: np.ndarray  # (N, H, W) bool
    boxes: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,) float32
    h: int
    w: int


@dataclass
class PromptSpec:
    prompt_dir: Path
    prompt_key: str
    prompt_id: str


def cv2_imwrite_any(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def cv2_imread_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _normalize_mask_array(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"Expected masks (N,1,H,W), got {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks (N,H,W), got {arr.shape}")
    return arr.astype(bool, copy=False)


def _scalar(v: np.ndarray | str | int | float | None) -> str:
    if v is None:
        return ""
    arr = np.asarray(v)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return str(v)


def parse_prompt_entry(entry: str) -> PromptSpec:
    raw_entry = entry.strip()
    if "," not in raw_entry:
        raise ValueError(
            f"Invalid prompt entry '{entry}'. Expected '<prompt_result_folder>,<id>' format."
        )

    folder_str, prompt_id = raw_entry.rsplit(",", 1)
    folder_str = folder_str.strip().strip('"').strip("'")
    prompt_id = prompt_id.strip().strip('"').strip("'")
    if not folder_str or not prompt_id:
        raise ValueError(
            f"Invalid prompt entry '{entry}'. Both folder and id must be non-empty."
        )

    prompt_dir = Path(folder_str).expanduser().absolute()
    prompt_key = prompt_dir.name
    return PromptSpec(prompt_dir=prompt_dir, prompt_key=prompt_key, prompt_id=prompt_id)


def _id_color(prompt_id: str) -> np.ndarray:
    # Stable pseudo-random color from prompt id so each id keeps one color.
    digest = hashlib.sha256(prompt_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.integers(64, 256, size=3, dtype=np.uint8)


def _prompt_id_sort_key(prompt_id: str) -> tuple[int, int | str]:
    normalized = prompt_id.strip()
    if normalized and normalized.lstrip("+-").isdigit():
        return (0, int(normalized))
    return (1, prompt_id)


def parse_positions(positions_txt: Path) -> dict[str, PositionInfo]:
    if not positions_txt.exists():
        raise FileNotFoundError(f"positions file not found: {positions_txt}")
    rows = [ln.strip() for ln in positions_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"positions file is empty: {positions_txt}")

    start = 0
    if rows[0].split()[0].lower() == "las_name":
        start = 1

    pos: dict[str, PositionInfo] = {}
    for line in rows[start:]:
        cols = line.split()
        if len(cols) < 4:
            continue
        las_name = cols[0]
        stem = Path(las_name).stem
        pos[stem] = PositionInfo(
            stem=stem,
            left_top_x=float(cols[1]),
            left_top_y=float(cols[2]),
            left_top_z=float(cols[3]),
        )
    if not pos:
        raise ValueError(f"no valid rows in positions file: {positions_txt}")
    return pos


def read_prompt_folder(prompt_dir: Path, prompt_key: str, prompt_id: str) -> list[TileResult]:
    npz_files = sorted(p for p in prompt_dir.glob("*.npz") if p.is_file())
    results: list[TileResult] = []
    for npz_path in tqdm(npz_files, desc=f"Read {prompt_key} (id={prompt_id})", unit="file"):
        data = np.load(npz_path)
        if not {"masks", "boxes", "scores"}.issubset(set(data.files)):
            continue

        masks = _normalize_mask_array(data["masks"])
        n = masks.shape[0]
        h, w = int(masks.shape[1]), int(masks.shape[2])

        boxes = np.asarray(data["boxes"], dtype=np.float32)
        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes.reshape(1, 4)
        if boxes.shape != (n, 4):
            raise ValueError(f"Invalid boxes shape in {npz_path}: {boxes.shape}, expected ({n},4)")

        scores = np.asarray(data["scores"], dtype=np.float32).reshape(-1)
        if scores.shape[0] != n:
            raise ValueError(f"Invalid scores length in {npz_path}: {scores.shape[0]}, expected {n}")

        stem = _scalar(data["image_stem"]) if "image_stem" in data else npz_path.stem
        # Backward compatibility with names like xxx_concept_merged.npz.
        if stem.endswith("_merged"):
            stem = stem[: -len("_merged")]
        results.append(
            TileResult(
                prompt_key=prompt_key,
                prompt_id=prompt_id,
                stem=stem,
                masks=masks,
                boxes=boxes,
                scores=scores,
                h=h,
                w=w,
            )
        )
    return results


def compute_canvas_and_offsets(
    positions: dict[str, PositionInfo],
    tile_sizes: dict[str, tuple[int, int]],
    resolution: float,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    valid_stems = [s for s in tile_sizes if s in positions]
    if not valid_stems:
        raise ValueError("No overlapping stems between positions and prompt results.")

    min_x = min(positions[s].left_top_x for s in valid_stems)
    max_y = max(positions[s].left_top_y for s in valid_stems)

    offsets: dict[str, tuple[int, int]] = {}
    max_w = 0
    max_h = 0
    for stem in valid_stems:
        w, h = tile_sizes[stem]
        x = int(round((positions[stem].left_top_x - min_x) / resolution))
        y = int(round((max_y - positions[stem].left_top_y) / resolution))
        offsets[stem] = (x, y)
        max_w = max(max_w, x + w)
        max_h = max(max_h, y + h)

    if max_w <= 0 or max_h <= 0:
        raise ValueError("computed canvas size is invalid")
    return max_h, max_w, offsets


def stitch_prompt_results(
    all_results: list[TileResult],
    offsets: dict[str, tuple[int, int]],
    canvas_h: int,
    canvas_w: int,
    output_dir: Path,
    base_image: np.ndarray | None = None,
    overlay_alpha: float = 0.45,
) -> None:
    global_boxes: list[np.ndarray] = []
    global_scores: list[float] = []
    global_prompts: list[str] = []
    global_prompt_ids: list[str] = []
    union_bool = np.zeros((canvas_h, canvas_w), dtype=bool)
    id_to_union_mask: dict[str, np.ndarray] = {}

    for tr in tqdm(all_results, desc="Stitch all prompts", unit="tile"):
        if tr.stem not in offsets:
            print(f"Warning: stem '{tr.stem}' not in positions; skipped.")
            continue
        x0, y0 = offsets[tr.stem]
        n = tr.masks.shape[0]

        shifted_boxes = tr.boxes.copy()
        shifted_boxes[:, [0, 2]] += float(x0)
        shifted_boxes[:, [1, 3]] += float(y0)

        for i in range(n):
            local_mask = tr.masks[i]
            union_bool[y0 : y0 + tr.h, x0 : x0 + tr.w] |= local_mask
            prompt_union = id_to_union_mask.setdefault(tr.prompt_id, np.zeros((canvas_h, canvas_w), dtype=bool))
            prompt_union[y0 : y0 + tr.h, x0 : x0 + tr.w] |= local_mask
            global_boxes.append(shifted_boxes[i])
            global_scores.append(float(tr.scores[i]))
            global_prompts.append(tr.prompt_key)
            global_prompt_ids.append(tr.prompt_id)

    if global_boxes:
        # Memory-safe metadata export: do not materialize (N, H, W) global masks.
        masks_out = np.zeros((0, 1, canvas_h, canvas_w), dtype=bool)
        boxes_out = np.stack(global_boxes, axis=0).astype(np.float32)
        scores_out = np.asarray(global_scores, dtype=np.float32)
    else:
        masks_out = np.zeros((0, 1, canvas_h, canvas_w), dtype=bool)
        boxes_out = np.zeros((0, 4), dtype=np.float32)
        scores_out = np.zeros((0,), dtype=np.float32)
        global_prompts = []
        global_prompt_ids = []

    sorted_prompt_ids = sorted(id_to_union_mask, key=_prompt_id_sort_key)
    id_colors = {prompt_id: _id_color(prompt_id) for prompt_id in sorted_prompt_ids}
    black_vis = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for prompt_id in sorted_prompt_ids:
        black_vis[id_to_union_mask[prompt_id]] = id_colors[prompt_id]

    overlay_vis: np.ndarray | None = None
    if base_image is not None:
        if base_image.ndim == 2:
            base_img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            base_img = base_image.copy()
            if base_img.shape[2] > 3:
                base_img = base_img[:, :, :3]
        if base_img.shape[0] != canvas_h or base_img.shape[1] != canvas_w:
            raise ValueError(
                f"base image shape {base_img.shape[:2]} does not match canvas {(canvas_h, canvas_w)}"
            )
        overlay_vis = base_img.astype(np.float32)
        for prompt_id in sorted_prompt_ids:
            mask = id_to_union_mask[prompt_id]
            color_f = id_colors[prompt_id].astype(np.float32)
            overlay_vis[mask] = (1.0 - overlay_alpha) * overlay_vis[mask] + overlay_alpha * color_f
        overlay_vis = np.clip(overlay_vis, 0, 255).astype(np.uint8)

    out_npz = output_dir / "all_prompts_global_instances.npz"
    # out_black_png = output_dir / "all_prompts_global_mask_black.png"
    out_legacy_png = output_dir / "all_prompts_global_mask.png"
    out_overlay_png = output_dir / "all_prompts_global_overlay.png"
    np.savez_compressed(
        out_npz,
        masks=masks_out,
        boxes=boxes_out,
        scores=scores_out,
        prompt_names=np.asarray(global_prompts, dtype=np.str_),
        prompt_ids=np.asarray(global_prompt_ids, dtype=np.str_),
        unique_prompt_ids=np.asarray(sorted_prompt_ids, dtype=np.str_),
        unique_prompt_id_colors=(
            np.asarray([id_colors[pid] for pid in sorted_prompt_ids], dtype=np.uint8)
            if sorted_prompt_ids
            else np.zeros((0, 3), dtype=np.uint8)
        ),
        union_mask=union_bool.astype(np.uint8),
        mask_storage=np.array("union_only"),
        canvas_height=np.array(canvas_h, dtype=np.int32),
        canvas_width=np.array(canvas_w, dtype=np.int32),
    )
    # if not cv2_imwrite_any(out_black_png, black_vis):
    #     raise RuntimeError(f"Failed to save mask image: {out_black_png}")
    # Keep backward-compatible output name.
    if not cv2_imwrite_any(out_legacy_png, black_vis):
        raise RuntimeError(f"Failed to save mask image: {out_legacy_png}")
    if overlay_vis is not None:
        if not cv2_imwrite_any(out_overlay_png, overlay_vis):
            raise RuntimeError(f"Failed to save overlay image: {out_overlay_png}")
    print(f"Saved: {out_npz}")
    # print(f"Saved: {out_black_png}")
    print(f"Saved: {out_legacy_png}")
    if overlay_vis is not None:
        print(f"Saved: {out_overlay_png}")


def run(
    positions_txt: Path,
    output_dir: Path,
    resolution: float,
    prompt_entries: list[str],
    base_image_path: Path | None = None,
    overlay_alpha: float = 0.45,
) -> None:
    positions = parse_positions(positions_txt)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_specs = [parse_prompt_entry(entry) for entry in prompt_entries]
    merged_all: list[TileResult] = []
    tile_sizes: dict[str, tuple[int, int]] = {}

    for spec in prompt_specs:
        prompt_dir = spec.prompt_dir
        if not prompt_dir.exists() or not prompt_dir.is_dir():
            raise NotADirectoryError(f"prompt result folder not found: {prompt_dir}")
        results = read_prompt_folder(prompt_dir, prompt_key=spec.prompt_key, prompt_id=spec.prompt_id)
        merged_all.extend(results)
        for tr in results:
            if tr.stem not in tile_sizes:
                tile_sizes[tr.stem] = (tr.w, tr.h)

    canvas_h, canvas_w, offsets = compute_canvas_and_offsets(
        positions=positions,
        tile_sizes=tile_sizes,
        resolution=resolution,
    )
    print(f"Global canvas: {canvas_w} x {canvas_h}")

    base_image: np.ndarray | None = None
    if base_image_path is not None:
        base_image = cv2_imread_any(base_image_path, cv2.IMREAD_UNCHANGED)
        if base_image is None:
            raise RuntimeError(f"Failed to read base image: {base_image_path}")

    stitch_prompt_results(
        all_results=merged_all,
        offsets=offsets,
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        output_dir=output_dir,
        base_image=base_image,
        overlay_alpha=overlay_alpha,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read multiple prompt result folders and stitch masks to one global map (color by prompt id)."
    )
    parser.add_argument("positions_txt", type=str, help="Path to las_positions.txt")
    parser.add_argument("output_dir", type=str, help="Output folder for global prompt masks")
    parser.add_argument(
        "prompt_entries",
        nargs="+",
        type=str,
        help="Prompt entries in '<prompt_result_folder>,<id>' format (e.g. /path/to/curb,1)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Projection resolution (meters/pixel), must match pc2img generation",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help="Optional stitched full image path. If provided, also outputs overlay visualization.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha when drawing prompt masks on full image",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        positions_txt=Path(args.positions_txt).expanduser().absolute(),
        output_dir=Path(args.output_dir).expanduser().absolute(),
        resolution=args.resolution,
        prompt_entries=args.prompt_entries,
        base_image_path=Path(args.base_image).expanduser().absolute() if args.base_image else None,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
