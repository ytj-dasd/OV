# OpenIns3D Batch Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Batch-run the six first-level scene LAS files with one model load and isolated scene outputs.

**Architecture:** Refactor single-scene inference into reusable runtime and scene
functions. Add a batch CLI that discovers `<scene>/<scene>.las`, reuses loaded
models, records failures, and writes one output folder per scene.

**Tech Stack:** Python, PyTorch, MinkowskiEngine, ODISE, NumPy

---

### Task 1: Test scene discovery

**Files:**
- Create: `OpenIns3D/tests/test_batch_zero_shot.py`

1. Create temporary scene and nested fusion LAS files.
2. Verify only matching first-level scene LAS files are returned.
3. Verify paths are sorted.

### Task 2: Refactor reusable inference

**Files:**
- Modify: `OpenIns3D/zero_shot_multi_vocs.py`

1. Add runtime construction for Mask3D, Lookup, and Snap.
2. Add a per-scene inference function accepting output directories.
3. Keep the original CLI behavior.

### Task 3: Add batch CLI

**Files:**
- Create: `OpenIns3D/batch_zero_shot_multi_vocs.py`

1. Parse input root, output root, vocabulary, voxel size, and confidence threshold.
2. Load models once.
3. Process each discovered scene sequentially.
4. Write `batch_summary.json`.

### Task 4: Verify

1. Run discovery and existing focused tests.
2. Compile both CLIs and helper modules.
3. Check command help and diff.
