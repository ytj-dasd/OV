"""SAM3 adapter for single-sample left/right visual-prompt inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def run_batch(
    image_paths: list[Path],
    output_dirs: list[Path],
    *,
    boxes: list[list[float]],
    score_threshold: float = 0.5,
    checkpoint: Path | None = None,
    device: str = "cuda",
) -> None:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if len(image_paths) != len(output_dirs):
        raise ValueError("image_paths and output_dirs must have the same length")
    if len(boxes) != 1:
        raise ValueError("Single-sample visual inference requires exactly one positive box")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3 visual inference")
    checkpoint = checkpoint or (Path.cwd() / "model" / "sam3.pt")
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), device=device)
    processor = Sam3Processor(model, resolution=1008, device=device, confidence_threshold=float(score_threshold))

    for image_path, output_dir in tqdm(
        list(zip(image_paths, output_dirs, strict=True)),
        desc="SAM3 visual grids",
        unit="grid",
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        state = processor.set_image(Image.open(image_path).convert("RGB"))
        for box in boxes:
            state = processor.add_geometric_prompt(box=box, label=True, state=state)
        masks = _to_numpy(state["masks"])
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        np.savez_compressed(
            output_dir / "masks.npz",
            masks=masks.astype(bool, copy=False),
            boxes=_to_numpy(state["boxes"]).astype(np.float32),
            scores=_to_numpy(state["scores"]).astype(np.float32),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample left/right SAM3 visual-prompt inference.")
    parser.add_argument("images_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--score-th", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_batch(
        [Path(path) for path in json.loads(args.images_json)],
        [Path(path) for path in json.loads(args.out)],
        boxes=json.loads(args.boxes),
        score_threshold=args.score_th,
        checkpoint=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
