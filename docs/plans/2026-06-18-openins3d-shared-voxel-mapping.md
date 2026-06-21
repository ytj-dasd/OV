# OpenIns3D Shared Voxel Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use one persisted voxel mapping across Mask3D, Snap, lookup, and full-resolution PLY export.

**Architecture:** Split quantization from sparse tensor construction. Run all model
and rendering operations on sampled points, then expand sampled instance masks to
the original points through `inverse_map`.

**Tech Stack:** Python, NumPy, PyTorch, MinkowskiEngine, plyfile

---

### Task 1: Test mapping and restoration

**Files:**
- Create: `OpenIns3D/tests/test_voxel_mapping.py`

1. Test deterministic representative indices and inverse mapping.
2. Test sampled mask restoration to original point order.
3. Test NPZ contents and compact integer dtypes.
4. Run tests and verify failure before implementation.

### Task 2: Implement mapping utilities

**Files:**
- Create: `OpenIns3D/openins3d/voxel_mapping.py`

1. Add quantization and mapping validation.
2. Add mask restoration.
3. Add compressed NPZ persistence.
4. Run focused tests.

### Task 3: Reuse sampled points through inference

**Files:**
- Modify: `OpenIns3D/openins3d/mask3d/__init__.py`
- Modify: `OpenIns3D/zero_shot_multi_vocs.py`

1. Add a sparse-input builder for already sampled points.
2. Run Mask3D without a second quantization.
3. Run Snap and lookup on sampled points and sampled masks.
4. Restore final masks before PLY export.
5. Print the mapping output path.

### Task 4: Verify

1. Run focused mapping, voxel-size, mask-output, and PLY-export tests.
2. Compile modified Python modules.
3. Check the diff for whitespace and unrelated changes.
