"""MapBuilder-compatible batch CLI for a local SAM3 checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _parse_json_paths(value: str, *, name: str) -> list[Path]:
    payload = json.loads(value)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"{name} must be a JSON list of paths")
    return [Path(item).expanduser().resolve() for item in payload]


def run_batch(
    image_paths: list[Path],
    output_dirs: list[Path],
    *,
    text: str,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.5,
    checkpoint: Path | None = None,
    device: str = "cuda",
) -> None:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if len(image_paths) != len(output_dirs):
        raise ValueError("image_paths and output_dirs must have the same length")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MapBuilder SAM3 batch inference")

    checkpoint = checkpoint or (Path.cwd() / "model" / "sam3.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    model = build_sam3_image_model(checkpoint_path=str(checkpoint), device=device)
    processor = Sam3Processor(
        model,
        resolution=1008,
        device=device,
        confidence_threshold=float(score_threshold),
    )

    for index, (image_path, output_dir) in enumerate(zip(image_paths, output_dirs), start=1):
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        state = processor.set_image(Image.open(image_path).convert("RGB"))
        output = processor.set_text_prompt(state=state, prompt=text)
        masks_source = output.get("masks_logits", output["masks"])
        masks = _to_numpy(masks_source)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        masks = masks > float(mask_threshold)
        boxes = _to_numpy(output.get("boxes", np.zeros((len(masks), 4), dtype=np.float32)))
        scores = _to_numpy(output.get("scores", np.ones((len(masks),), dtype=np.float32)))

        np.savez_compressed(
            output_dir / "masks.npz",
            masks=masks.astype(bool, copy=False),
            boxes=np.asarray(boxes, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
        )
        print(
            f"[sam3-adapter] {index}/{len(image_paths)} image={image_path.name} "
            f"prompt={text!r} masks={len(masks)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MapBuilder-compatible SAM3 batch inference.")
    parser.add_argument("images_json", help="JSON list of input image paths.")
    parser.add_argument("--batch", action="store_true", help="Accepted for MapBuilder CLI compatibility.")
    parser.add_argument("--text", required=True, help="Single text prompt.")
    parser.add_argument("--out", required=True, help="JSON list of output directories.")
    parser.add_argument("--th", type=float, default=0.5, help="Mask-logit sigmoid threshold.")
    parser.add_argument("--score-th", type=float, default=0.5, help="Instance confidence threshold.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SAM3 checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Inference device; full pipeline requires CUDA.")
    args = parser.parse_args()

    run_batch(
        _parse_json_paths(args.images_json, name="images_json"),
        _parse_json_paths(args.out, name="out"),
        text=args.text,
        mask_threshold=args.th,
        score_threshold=args.score_th,
        checkpoint=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
