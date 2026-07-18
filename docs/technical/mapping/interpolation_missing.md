---
title: Mapping interpolation and missing values
doc_id: neat-tech-mapping-interpolation
doc_type: technical_reference
functional_area: mapping
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [image stacks]
scientific_review: pending
source_paths: [NEAT/workers/batch.py]
source_symbols: [BatchFitEdgesWorker.interpolate_results, BatchFitWorker.interpolate_results]
test_paths: [tests/test_batch_mapping_outputs.py]
---

# Mapping interpolation and missing values

Without interpolation, only box-center pixels contain fitted values; all other
detector pixels remain NaN. Both “ungridded” and “gridded” CSVs still contain
every detector coordinate.

Interpolation operates only inside the selected mapping ROI and independently
for every parameter, uncertainty and edge. It does not distinguish a skipped
grid location from a failed fit; both are NaN and can be filled.

Individual-edge mapping requires at least four valid points per array. It uses
linear `scipy.interpolate.griddata`; Qhull/value failures fall back to nearest
neighbor. Linear interpolation can still return NaN outside the convex hull.

Pattern mapping attempts linear interpolation whenever any valid point exists.
It has no too-few-points check or fallback, so insufficient or collinear points
can raise an exception and prevent the final gridded save.

A focused test verifies linear filling at a point inside the convex hull for
all individual-edge result arrays.

The individual worker imports `QhullError` from the deprecated
`scipy.spatial.qhull` namespace; current SciPy emits a deprecation warning and
this import should move to `scipy.spatial`.

Interpolation also fills uncertainty arrays numerically; it does not propagate
statistical uncertainty. Scientific review must define whether failed fits may
be interpolated and how such pixels should be marked.

## Retrieval questions

- What is the difference between missing grid points and failed fits?
- Where does NEAT interpolate mapping values?
- Why are NaNs left outside the convex hull?
- Does interpolating uncertainty constitute uncertainty propagation?
