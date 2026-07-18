---
title: RADEN multi-page TIFF normalisation
doc_id: neat-tech-preprocessing-normalisation-raden
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: domain-reviewed
instrument_applicability: [RADEN]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/workers/batch.py]
source_symbols: [RadenNormalisationWorker, get_raden_tiff_stack_info]
test_paths: [tests/test_fitting_headless.py]
---

# RADEN multi-page TIFF normalisation

## Input validation

Both selections must be classified as RADEN TIFF stacks. The worker requires:

- equal frame counts;
- identical image shapes;
- matching ToF `bins`, `min` and `max` under `np.isclose`; and
- equal lowercased ToF unit strings.

Each TIFF page is read as `float32`; NaN and infinity are changed to zero.
Detailed `.stat`/JSON discovery and ToF interpretation are covered in Batch 2.

## Pulse scale and parameters

The worker searches metadata first for `pulses_with_data`, then `pulses`.
When both sample and open-beam values are finite and positive:

```text
scale = open_beam_pulses / sample_pulses
```

Otherwise it reports that pulse metadata are unavailable and uses scale 1.
Spatial half-window `n` and adjacent-frame half-window `m` have the same
implemented meaning as classic normalisation. The default is `n=10, m=0`;
permitted ranges are `0 <= n <= 100` and `0 <= m <= 10`.

## Algorithm

For sample page `i`, open-beam pages `i-m ... i+m`, clipped to stack limits,
are summed. The same integral-image, border-rescaled local open-beam
normalisation is then applied:

```text
normalised = K × (2n+1)^2 × sample / scaled_local_open_beam × pulse_scale
```

The frame must be at least `(2n+1)` in each dimension. Nonfinite results are
written as zero.

## Outputs

The result is one multi-page `float32` TIFF. If the entered base name does not
already end in `.tif` or `.tiff`, the sample-stack stem is appended:

```text
<base>_<sample-stack-stem>.tiff
```

The detected metadata file is copied with the output stem. The first other
sidecar found for each of `.stat`, `.json` and `.log` is also copied, avoiding a
duplicate extension.

## Cancellation and limitations

Stop is checked per page. A stop can leave a partial TIFF, and sidecars are not
copied unless the frame loop completes. The worker writes frames incrementally
and performs garbage collection every 100 pages. Its `succeeded` flag is true
only when every expected page is written without cancellation or error.

The headless test verifies a multi-page normalized output. It currently exposes
a harmless ignored Pillow `AppendingTiffWriter` closed-file warning, so writer
lifecycle deserves future cleanup. The pulse preference/scale, ToF matching
criteria and spatial/temporal equation were domain-reviewed on 2026-07-16.

## Retrieval questions

- What must match between RADEN sample and open-beam stacks?
- How does RADEN pulse scaling work?
- How is the output TIFF named?
- What remains after RADEN normalisation is cancelled?
