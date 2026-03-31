from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = "/home/guitu/文档/vector/sam3/model/sam3.pt"
DEFAULT_PROMPTS = ("lane line", "arrow")


def _safe_name(text: str) -> str:
    out = text.strip().replace(" ", "_")
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in out)
    out = out.strip("._")
    return out or "query"


def _collect_image_paths(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if recursive:
        image_paths = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    else:
        image_paths = [
            p
            for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in directory: {input_path} (supported: {sorted(SUPPORTED_IMAGE_EXTS)})"
        )
    return image_paths


def run_lane_arrow_agent_inference(
    input_path: Path | str,
    *,
    prompts: list[str],
    server_url: str,
    llm_model: str,
    llm_name: str,
    api_key: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    sam_resolution: int = 1008,
    confidence_threshold: float = 0.5,
    output_root: Path | str | None = None,
    recursive: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().absolute()
    image_paths = _collect_image_paths(input_path, recursive=recursive)

    if output_root is None:
        base_dir = input_path.parent if input_path.is_file() else input_path
        output_root_path = (base_dir / "agent_lane_arrow_results").absolute()
    else:
        output_root_path = Path(output_root).expanduser().absolute()
    output_root_path.mkdir(parents=True, exist_ok=True)

    resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        resolved_device = "cpu"

    checkpoint_path = str(Path(checkpoint).expanduser().absolute())
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )
    processor = Sam3Processor(
        model,
        resolution=int(sam_resolution),
        device=resolved_device,
        confidence_threshold=float(confidence_threshold),
    )

    llm_config = {
        "provider": "vllm",
        "model": llm_model,
        "name": llm_name,
        "api_key": api_key,
    }
    send_generate_request = partial(
        send_generate_request_orig,
        server_url=server_url,
        model=llm_model,
        api_key=api_key,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    summary: dict[str, Any] = {
        "total_images": len(image_paths),
        "prompts": list(prompts),
        "runs_total": len(image_paths) * len(prompts),
        "runs_failed": 0,
        "failures": [],
        "output_root": str(output_root_path),
    }

    for prompt in prompts:
        prompt_dir = output_root_path / _safe_name(prompt)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(image_paths, desc=f"Agent inference [{prompt}]", unit="img"):
            try:
                run_single_image_inference(
                    image_path=str(img_path),
                    text_prompt=prompt,
                    llm_config=llm_config,
                    send_generate_request=send_generate_request,
                    call_sam_service=call_sam_service,
                    output_dir=str(prompt_dir),
                    debug=debug,
                )
            except Exception as exc:  # noqa: BLE001
                summary["runs_failed"] += 1
                summary["failures"].append(
                    {
                        "image": str(img_path),
                        "prompt": prompt,
                        "error": str(exc),
                    }
                )
                print(f"[ERROR] image={img_path} prompt={prompt} error={exc}")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 Agent with local SAM3 and local Qwen3-VL endpoint for lane lines and arrows."
    )
    parser.add_argument("input_path", type=str, help="Input image path or image directory")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="Agent text queries to run (default: lane line + arrow)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://0.0.0.0:8001/v1",
        help="OpenAI-compatible local VLM endpoint (e.g. vLLM)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-VL-8B",
        help="Local VLM model name served by endpoint",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="qwen3_vl_8b_local",
        help="Short safe model name for output file naming",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="DUMMY_API_KEY",
        help="API key passed to OpenAI-compatible client (can be dummy for local vLLM)",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="SAM3 device, e.g., cuda or cpu")
    parser.add_argument("--sam-resolution", type=int, default=1008, help="SAM3 processor square resize resolution")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Mask confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output root directory")
    parser.add_argument("--recursive", action="store_true", help="Recursively find images in directory input")
    parser.add_argument("--debug", action="store_true", help="Enable SAM3 agent debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_lane_arrow_agent_inference(
        input_path=args.input_path,
        prompts=args.prompts,
        server_url=args.server_url,
        llm_model=args.llm_model,
        llm_name=args.llm_name,
        api_key=args.api_key,
        checkpoint=args.checkpoint,
        device=args.device,
        sam_resolution=args.sam_resolution,
        confidence_threshold=args.confidence_threshold,
        output_root=args.output,
        recursive=args.recursive,
        debug=args.debug,
    )
    print(
        "Done. "
        f"images={result['total_images']} runs={result['runs_total']} "
        f"failed={result['runs_failed']} output={result['output_root']}"
    )
