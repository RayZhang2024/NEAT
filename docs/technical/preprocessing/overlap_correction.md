---
title: Overlap (pile-up) correction
doc_id: neat-tech-preprocessing-overlap
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [classic 512-by-512 detector exports]
scientific_review: pending
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.correct_overlap, OverlapCorrectionWorker]
test_paths: [tests/test_preprocessing_workers.py]
---

# Overlap (pile-up) correction

## Scope and warning

This page describes the implemented calculation. The physical derivation,
instrument applicability and conditions under which this correction is valid
have not yet been domain-reviewed. The assistant must not recommend this step
solely from this code description.

## Inputs

Each dataset needs:

- images with sortable numeric suffixes;
- at least one `_Spectra.txt`;
- at least one `_ShutterCount.txt`; and
- image shape exactly 512×512.

The UI loads the first matching spectra and shutter file. It passes the full
spectra array and a flattened array of every nonzero shutter-table value.

## Segmentation

The first spectra column is interpreted as time of flight (ToF). A new segment
boundary is created wherever:

```text
diff(ToF) > 0.0001
```

The worker computes the mean consecutive ToF interval inside each segment. A
single-element segment receives interval `1e-5`; an empty segment would receive
zero. The first segment interval is the reference and must be nonzero.

From the flattened shutter values, only values greater than 1000 are retained.
There must be at least as many retained values as ToF segments. The first
`number-of-segments` values are used.

## Correction algorithm

Images are numerically sorted from digits in their suffix. For each segment,
the worker maintains cumulative intensity `C` up to the current frame. With
segment shutter count `N`:

```text
p(x,y) = C(x,y) / N
d(x,y) = max(1 - p(x,y), 1e-10)
corrected(x,y) = image(x,y) / d(x,y) × reference_interval / segment_interval
```

If the current interval is nonpositive, the interval scale is 1. A zero
shutter count skips that image. Any result containing NaN or infinity is
skipped.

## Outputs

Corrected frames are written as:

```text
Corrected_<base>_<digits-from-input-suffix>.fits
```

All recognized spectra and shutter-count files from the source folder are
copied unchanged. The standalone UI creates `Corrected_<dataset-name>` below
the selected output parent.

## Progress and cancellation

Progress follows the index of successfully reached frames in the sorted list.
Stop is checked before each image. Sidecar copying still occurs after the image
loop, including after a stop request.

## Known ambiguities and risks

- Segment membership uses the sorted image position, assuming one image per
  spectra row in the same order; this relationship is not independently
  validated.
- Flattening the entire two-column shutter table mixes both columns before the
  `>1000` filter. The intended column semantics require confirmation.
- Clamping `1-p` to `1e-10` can generate extremely large finite values rather
  than rejecting `p >= 1`.
- The worker records `succeeded=False` after abort, cancellation or any skipped
  frame. Successfully written frames can remain after a later failure.
- The input dictionary is stored separately from the `run()` method, so normal
  Python and Qt thread invocation are both available.
- Focused tests verify the one-segment correction equation and 512×512 shape
  rejection. Multi-segment behavior remains untested.

## Retrieval questions

- What equation does overlap correction implement?
- Why does overlap correction require 512×512 images?
- How are ToF segments and shutter counts selected?
- When is overlap correction scientifically appropriate?
