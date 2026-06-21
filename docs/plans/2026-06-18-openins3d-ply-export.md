# OpenIns3D PLY Result Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured PLY and class-map outputs to multi-vocabulary OpenIns3D inference.

**Architecture:** Implement a lightweight NumPy/plyfile exporter independent of the
CUDA inference modules. Pass the active vocabulary, classification results, scores,
masks, and loader offset from `zero_shot_multi_vocs.py`.

**Tech Stack:** Python, NumPy, plyfile, pytest

---

### Task 1: Define exporter behavior with tests

**Files:**
- Create: `OpenIns3D/tests/test_result_export.py`
- Create: `OpenIns3D/openins3d/result_export.py`

1. Write tests for PLY fields, semantic IDs, background values, deterministic
   colors, overlap priority, and coordinate restoration.
2. Run the tests and verify they fail because the exporter does not exist.
3. Implement the minimum exporter needed by the tests.
4. Run the tests and verify they pass.

### Task 2: Connect the exporter to inference

**Files:**
- Modify: `OpenIns3D/zero_shot_multi_vocs.py`

1. Return both inference coordinates and the coordinate offset from the loader.
2. Export the complete point cloud after multiview aggregation.
3. Write the active vocabulary mapping beside the PLY.
4. Print both output paths at the end of inference.

### Task 3: Verify integration

**Files:**
- Test: `OpenIns3D/tests/test_result_export.py`
- Check: `OpenIns3D/zero_shot_multi_vocs.py`

1. Run the focused pytest file.
2. Compile the modified Python modules.
3. Inspect the diff for unrelated changes.
