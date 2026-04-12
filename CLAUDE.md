# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zero-shot multi-scene point cloud instance segmentation system using VLMs (Qwen3-VL / GLM-4.6V) + SAM3. Processes urban street scene LAS point clouds and segments 15 categories of urban objects (utility poles, street lights, traffic signs, trees, fences, etc.).

Documentation is primarily in Chinese (`pipeline.md` is the authoritative spec, `coding.md` is development notes).

## Pipeline Architecture

6-task sequential pipeline. Data flows through `benchmark/` with a fixed per-scene directory hierarchy:

```
benchmark/
  scene_{scene_id}/
    source/            # Input: LAS point clouds + camera station JSONs
    projected_images/  # Task 2 output: PNG images + NPZ point-pixel correspondences
    vlm_desc/          # Task 3 output: per-image .vlm.txt class detections
    sam_mask/          # Task 4 output: per-image .npz masks/boxes/scores
    fusion/            # Task 5 output: instance-segmented LAS + NPZ + JSON
  cross_scene/         # Task 6 output (not yet implemented)
```

All tasks support resume — they skip existing output directories.

- **Task 1**: Scene discovery (scan `scene_*/source/*.las`)
- **Task 2**: Point cloud projection to multi-view images (soft-splat rendering via `pc2img_soft` in `ImgProject/pyIMS/Core/pc2img.py`)
- **Task 3**: VLM classification of projected images — outputs `.vlm.txt` per image (format: `class_name<TAB>confidence` per line)
- **Task 4**: SAM3 segmentation per detected class — outputs `.npz` with `masks`, `boxes`, `scores`
- **Task 5** (merged 5+6+7): 2D→3D back-projection, IoU-based instance merging, dual-channel merge (point IoU + XY distance), class-weighted scoring, point conflict resolution, DBSCAN denoising, ground removal, fence re-clustering, tree trunk/crown processing
- **Task 6**: Cross-scene boundary instance merging (referenced in `pipeline.md`, not yet implemented)

## Build & Run Commands

```bash
# Install ImgProject dependencies
cd ImgProject && uv sync

# Run GUI
cd ImgProject && uv run main.py

# Task 3 - VLM classification (Qwen3-VL)
python Qwen3-VL/infer/batch_scene_classify.py \
  --data-root benchmark --model-path Qwen/Qwen3-VL-2B-Thinking --max-new-tokens 1024

# Task 3 - VLM classification (GLM-V)
python GLM-V/inference/batch_scene_classify_glm.py \
  --data-root benchmark --model-path ZhipuAI/GLM-4.6V-Flash --max-new-tokens 1024

# Task 4 - SAM segmentation
python sam3/preprocess/batch_scene_sam_from_vlm.py \
  --data-root benchmark --alpha 0.45 \
  --checkpoint sam3/model/sam3.pt --device cuda --sam-resolution 1008

# Task 5 - 2D→3D instance segmentation (many params, see .vscode/launch.json for defaults)
python ImgProject/pipeline/task5_scene_instance_seg.py --data-root benchmark \
  --iou-threshold 0.25 --merge-xy-distance 0.5 --fov-deg 90
```

## Testing

```bash
cd ImgProject && pytest          # projection, DBSCAN, instance seg tests
cd sam3 && pytest                 # SAM batch inference tests
```

## Codebase Structure

- **`ImgProject/`** — Primary custom code. PySide6 GUI + core pipeline logic.
  - `pyIMS/IMS.py` — Main module: projection, inference, terrain-adjusted camera heights
  - `pyIMS/Core/pc2img.py` — Soft-splat point cloud rendering (~31K lines)
  - `pyIMS/Core/img2pc.py` — 2D→3D back-projection
  - `pyIMS/Core/instance.py` / `merge.py` — Instance data structures and merging
  - `pipeline/task5_scene_instance_seg.py` — Task 5 entry point
  - `pipeline/utils.py` — Task 5 utility functions (~60K lines)
- **`GLM-V/`** — Forked ZhipuAI GLM-4.6V. Custom scripts in `inference/`.
- **`Qwen3-VL/`** — Forked Qwen3-VL. Custom scripts in `infer/`. Model weights in `Qwen/`.
- **`sam3/`** — Forked Meta SAM3. Custom scripts in `preprocess/`. Model weights in `model/`.

## Key Intermediate Data Formats

- `projected_images/*.npz` must contain: `dist_img`, `pts_img_indices`, `pts_indices`
- `vlm_desc/*.vlm.txt`: each line is `class_name<TAB>confidence`
- `sam_mask/*.npz`: `masks`, `boxes`, `scores`
- `fusion/*_instance_seg.npz`: `scene_instance_id`, `class_id`, `class_name`, `confidence`, `point_indices`, `point_instance_id (N,int32)`, `point_confidence (N,float32)`

## Class Vocabulary

15 classes defined in `benchmark/class_vocab.yaml` (IDs 1–15, 0 = background). Each class has `id`, `name_zh`, `name_en`, and `aliases` for VLM text normalization.

## Dependencies

- Python >=3.11, managed by `uv`
- PyTorch + CUDA, transformers (HuggingFace), cupy-cuda11x
- laspy, open3d, Pillow, scipy, PySide6
- Model weights (not in git): Qwen3-VL variants, GLM-4.6V-Flash, sam3.pt
