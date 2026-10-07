---
title: Overlap (pile-up) correction
doc_id: neat-tech-preprocessing-overlap
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 349e9703ddb5aeb89bf6e6b7ba9806de4543e84c
status: code-verified
instrument_applicability: [classic 512-by-512 detector exports]
scientific_review: pending
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_overlap.py]
source_symbols: [PreprocessingMixin.correct_overlap, OverlapCorrectionWorker, correct_loaded_image_run, correct_overlap_frame]
test_paths: [tests/test_preprocessing_workers.py, tests/test_preprocessing_overlap.py]
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
- the first image in mapping insertion order has shape exactly 512×512.

The UI loads the first matching spectra and shutter file. It passes the full
spectra array and a flattened array of every nonzero shutter-table value.

## Segmentation

The first spectra column is interpreted as time of flight (ToF). A new segment
boundary is created wherever:

```text
diff(ToF) > 0.0001
```

The headless service computes the mean consecutive ToF interval inside each segment. A
single-element segment receives interval `1e-5`; an empty segment would receive
zero. The first segment interval is the reference and must be nonzero.

From the flattened shutter values, only values greater than 1000 are retained.
There must be at least as many retained values as ToF segments. The first
`number-of-segments` values are used.

## Correction algorithm

Images are stably sorted by the integer formed from digits in their suffix;
suffixes without digits sort first, and equal numeric keys retain mapping order.
For each segment, the service maintains cumulative intensity `C` up to the
current frame. With
segment shutter count `N`:

```text
p(x,y) = C(x,y) / N
d(x,y) = max(1 - p(x,y), 1e-10)
corrected(x,y) = image(x,y) / d(x,y) × reference_interval / segment_interval
```

If the current interval is nonpositive, the interval scale is 1. A zero
shutter count skips that image. Any result containing NaN or infinity is
skipped. Cumulative intensity is advanced before the later NaN/write checks;
a frame that fails those checks can still affect the next frame in its segment.
Only the first frame in mapping insertion order must have shape 512×512.
Later frames are not pre-validated for shape, preserving NumPy's broadcast or
error behavior.

## Outputs

Corrected frames are written as:

```text
Corrected_<base>_<digits-from-input-suffix>.fits
```

All recognized spectra and shutter-count files from the source folder are
copied unchanged after the image loop, even after frame failure or cancellation.
The two sidecar types use separate raw-order folder enumerations. Sidecar-copy
errors are warnings and do not retroactively change the image-loop status.
The standalone UI creates `Corrected_<dataset-name>` below the selected output
parent. The service does not create that folder. Successful writes are recorded
in production order, including repeated output paths when suffixes collapse to
the same digits and a later FITS write overwrites an earlier one.

## Progress and cancellation

Progress is emitted only after a successful FITS write, using that image's
position in the sorted list rather than the count of successes. Stop is checked
before each image, not during setup or a frame. Final success/cancellation is
snapshotted after the image loop and before sidecars; a later stop request
during sidecar copying does not change that result. The Qt worker remains a
signal/state adapter and retains the structured `PreprocessingOperationResult`.

## Known ambiguities and risks

- Segment membership uses the sorted image position, assuming one image per
  spectra row in the same order; this relationship is not independently
  validated.
- Flattening the entire two-column shutter table mixes both columns before the
  `>1000` filter. The intended column semantics require confirmation.
- Clamping `1-p` to `1e-10` can generate extremely large finite values rather
  than rejecting `p >= 1`.
- The worker records `succeeded=False` after abort, cancellation or any skipped
  frame. Successfully written frames can remain after a later failure. The
  structured result counts successful corrected FITS writes and retains those
  paths, errors, and sidecar warnings.
- The input dictionary is stored separately from the `run()` method, so normal
  Python and Qt thread invocation are both available.
- Golden values were captured from the original worker at
  `0f3acf640925276d0cd8087e54fa2cd4eef092e5` using deterministic
  512×512 synthetic frames. Focused tests compare exact FITS float32 values,
  orientation, multi-segment scaling, first-N shutter selection, cumulative
  carry-forward after a write failure, messages and progress. This is code
  equivalence evidence, not a scientific review of the physical model.

## Retrieval questions

- What equation does overlap correction implement?
- Why does overlap correction require 512×512 images?
- How are ToF segments and shutter counts selected?
- When is overlap correction scientifically appropriate?
