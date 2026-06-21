# OpenIns3D Batch Inference Design

## Scope

Given a directory such as `benchmark/RXL/Front-view`, discover only first-level
scene point clouds following:

`<input_root>/<scene_name>/<scene_name>.las`

Nested `fusion` and `fusion_sam3` outputs are ignored.

## Execution

- Load Mask3D and ODISE once.
- Process the six scene LAS files sequentially.
- Reuse the same inference behavior and defaults as `zero_shot_multi_vocs.py`.
- Isolate each scene under `<output_root>/<scene_name>/`.
- Continue to the next scene if one scene fails and write a batch summary JSON.

## Per-Scene Outputs

```text
output/<scene_name>/
  <scene_name>_instances.ply
  <scene_name>_classes.json
  <scene_name>_voxel_mapping.npz
  snap/<scene_name>/...
  snap/<scene_name>_vis/...
  odise/<scene_name>/...
```

## Vocabulary

The default batch vocabulary is the current road-facility vocabulary:

`lane marking, crosswalk, manhole, utility pole, street light, signboard,
traffic sign, signal light, surveillance camera, tree, trash bin, fire hydrant,
utility box, sculpture, bench, traffic cone, bollard, fence`

`--vocab` may replace it.
