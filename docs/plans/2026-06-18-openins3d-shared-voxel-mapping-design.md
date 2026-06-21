# OpenIns3D Shared Voxel Mapping Design

## Goal

Voxelize an input point cloud once, use the sampled points for Mask3D, Snap, and
2D/3D lookup, persist the original-to-sampled mapping, and restore final instance
masks to every original point before PLY export.

## Data Flow

1. Load the full point cloud and retain its original point order.
2. Quantize XYZ with the configured voxel size.
3. Store:
   - `unique_map`: sampled point index to original point index.
   - `inverse_map`: original point index to sampled point index.
   - `coordinate_offset`: offset used to center LAS/LAZ coordinates.
   - `voxel_size`.
4. Build the MinkowskiEngine input directly from sampled points without another
   quantization pass.
5. Run Mask3D, Snap rendering, and ODISE lookup on sampled points only.
6. Restore each sampled mask with `sampled_masks[inverse_map]`.
7. Export a full-resolution PLY using original point order and restored masks.

## Mapping Storage

The mapping is stored at:

`output/results_demo/<scene>/<scene>_voxel_mapping.npz`

To reduce storage for large scenes, map arrays use the smallest unsigned integer
dtype that can represent their largest index and are written with
`np.savez_compressed`.

## Safety

- Validate that `inverse_map` has one entry per original point.
- Validate that all inverse indices refer to sampled points.
- Keep original coordinates centered in memory and add the LAS offset only at PLY
  export.
- Use sampled points for both projection and mask lookup so the point and mask
  dimensions remain aligned.
