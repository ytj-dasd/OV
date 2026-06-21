# OpenIns3D PLY Result Export Design

## Goal

Export OpenIns3D inference results as a point PLY containing `x y z red green blue
instance_id semantic_id score`, while restoring LAS coordinates and preserving the
active vocabulary as numeric semantic IDs.

## Data Model

- `semantic_id` is the zero-based index of the active vocabulary.
- The default Replica vocabulary therefore uses IDs `0..47`.
- With `--vocab-only`, a vocabulary of length `N` uses IDs `0..N-1`.
- Without `--vocab-only`, `--vocab` does not extend the default 48 classes.
- Detected instances use IDs `1..K`.
- Unrecognized points use black RGB, `instance_id=0`, `semantic_id=-1`, and
  `score=0`.
- A deterministic random color is assigned per detected instance.
- If masks overlap, the instance with the higher aggregated score owns the point.

## Coordinate Handling

LAS/LAZ input is centered before inference for numerical stability. The exporter
adds the saved XYZ mean back before writing the PLY. PLY and NPY inputs use a zero
offset because their loader does not center them.

## Outputs

- `output/results_demo/<scene>/<scene>_instances.ply`
- `output/results_demo/<scene>/<scene>_classes.json`

The JSON file records the active `semantic_id -> class_name` mapping because
per-vertex string properties are not consistently supported by PLY readers.

## Verification

Unit tests inspect the generated PLY fields, coordinate restoration, background
values, deterministic instance colors, semantic IDs, scores, and overlap handling.
