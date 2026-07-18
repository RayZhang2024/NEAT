---
title: Filtering and masks
doc_id: neat-tech-preprocessing-filtering
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/ui/dialogs.py]
source_symbols: [FilteringWorker, MaskGeneratorDialog]
test_paths: [tests/test_preprocessing_workers.py]
---

# Filtering and masks

## Implemented mask meaning

Filtering is a spatial keep/discard operation, not smoothing. For every image
pixel:

```text
output = input, where mask == 1
output = 0,     where mask == 0
```

The mask must contain only finite values 0 and 1. Any other value aborts the
worker. The mask and image shapes must match exactly.

## UI and generated masks

Users may load one mask or open **Generate Mask**. The generator can load
FITS/TIFF data, create a finite threshold mask, invert the threshold direction,
and edit it with brush, rectangle or polygon tools. Its generated mask is
`uint8`, with nonzero pixels retained.

The selected output directory must already exist and a base name is required.

## Processing and outputs

The validated binary mask and images are converted to `float32`. Shape
mismatches and write failures mark the worker failed/incomplete; successfully
processed frames may already exist.

Output images are `<base>_<suffix>.fits`. The first spectra and first shutter
sidecar from each run are copied as `Run<index>_<original-name>`.

## Progress and cancellation

Progress increments for saved images, but a per-run percentage is also emitted
after each run. Therefore the displayed percentage can jump and is not a strict
count of successful outputs. Stop is cooperative between runs and frames.

## Limitations

- Zeroing discarded pixels may cause Clean to treat them as invalid if Clean is
  run afterward.
- No uncertainty or masked-pixel metadata are written into FITS.
- Focused tests verify binary-mask semantics, nonbinary-mask rejection,
  shape-mismatch failure and successful completion state.
  Interactive mask-editor behavior remains without dedicated coverage.
- Filtering is not part of Full Process. If used separately, it should not be
  followed by Clean because Clean treats the intentionally zeroed region as
  invalid pixels.

## Retrieval questions

- Does Filtering smooth the image?
- Which mask values keep a pixel?
- Why was an image skipped during Filtering?
- What happens if I run Clean after applying a mask?
