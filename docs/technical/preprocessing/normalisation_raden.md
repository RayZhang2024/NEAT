---
title: RADEN multi-page TIFF normalisation
doc_id: neat-tech-preprocessing-normalisation-raden
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: eb605978e4cbc52630ff1138866d6786c2208b0d
status: domain-reviewed
instrument_applicability: [RADEN]
scientific_review: completed 2026-07-16
source_paths: [NEAT/services/preprocessing_normalisation_raden.py, NEAT/services/preprocessing_normalisation_kernel.py, NEAT/services/preprocessing_normalisation.py, NEAT/workers/preprocessing.py, NEAT/workers/batch.py]
source_symbols: [normalise_raden_tiff_stack, normalise_local_open_beam_frame, RadenNormalisationWorker, get_raden_tiff_stack_info]
test_paths: [tests/test_preprocessing_normalisation_raden.py, tests/test_preprocessing_normalisation.py, tests/test_fitting_headless.py]
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

## Headless service and compatibility boundary

Issue #37 extracted the RADEN workflow into
`NEAT.services.preprocessing_normalisation_raden.normalise_raden_tiff_stack`.
The service accepts already resolved sample and open-beam stack-info mappings;
it does not parse `.stat`/JSON metadata or load complete stacks. The existing
`get_raden_tiff_stack_info()` call remains in the worker compatibility
boundary, which uses a truthy supplied `run["info"]` directly and resolves
missing/falsey info through the existing loader.
Truthy supplied info is assumed to meet the loader's normal invariants;
loader-impossible handcrafted metadata is outside this service's supported
compatibility boundary.

The worker creates the output directory before resolving metadata. The service
also creates it at operation start before validation. As before, resolution or
validation failure can therefore leave an empty output directory. The
standalone UI dispatch and the classic-only Full Process path are unchanged.
RADEN remains a separate workflow and does not use `LoadedImageRun`.

The RADEN reader still seeks one TIFF page at a time, copies it to `float32`,
and replaces NaN and both infinities with zero. It does not vertically flip
pages. Adjacent open-beam pages use absolute stack indices, clipped to the
stack bounds, and are accumulated in `float64`. The shared pure
`normalise_local_open_beam_frame()` kernel performs only the prepared-array
shape/window checks and local spatial calculation. Classic normalisation
retains its suffix-based temporal selection and delegates its already-prepared
arrays to the same kernel. RADEN sanitisation and the classic input dtype rules
remain outside that kernel.

The RADEN service returns `PreprocessingOperationResult`: each successfully
saved TIFF page increments `processed_count`, and `expected_count` remains
`None` until a validated total is established. A `normalised_stack` output is
recorded after the first successful page save, so a useful partial TIFF remains
visible on cancellation or later failure. The zero-byte writer placeholder
created before page zero is not reported as an output. Sidecars are added to
the ordered outputs only after each copy succeeds.

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

The detected metadata file is copied first with a lowercase extension and the
output stem, including unusual metadata extensions. Then the first matching
regular file is copied for each of `.stat`, `.json` and `.log`, in that order.
Each unsatisfied extension uses its own raw-order directory scan; the first
match wins. A metadata extension suppresses the corresponding later scan.
Only sample-side metadata is copied. Missing sidecars are silent. Sidecar
enumeration and copy errors are fatal, while the TIFF and any earlier copied
sidecars remain represented in the failed result.

## Cancellation and limitations

Stop is checked before each page and after the TIFF contexts close before
sidecar copying. A stop detected in the loop or at that post-close boundary
skips sidecars. Once sidecar copying starts, it runs to completion even if a
stop arrives during copying; the final status then observes the current
running state. Cancellation remains cooperative and can leave a readable
partial TIFF. Progress remains `int(100 * (page_index + 1) / total)` after the
page is saved, the writer advances, and the processed count increments. The
worker collects garbage after zero-based page indices 0, 100, 200, and so on.
These page-boundary collections execute inside the service and retain their
existing operation-failure semantics. A separate terminal collection after
the service result is established is best-effort: a failure produces a
`[WARN] Worker finalization: <error>` message when possible, leaves the result
unchanged and does not prevent the worker's single `finished` notification.
Its `succeeded` flag is true only when every expected page is written without
cancellation or error.

The adapter preserves the original message text, constructor, `stop()` method,
public worker helpers, and no-argument `finished` signal. The signal is
attempted exactly once after ordinary execution. `result` retains structured
success, failure, cancellation, counts, outputs and errors. Missing stack info
is still resolved through `get_raden_tiff_stack_info()` before calling the
headless service.

Golden values and lifecycle evidence were captured by executing the original
worker at `04bb285383de172dca308e0c9512c6640ded3ca2` using small synthetic
multi-page stacks. Focused tests cover the exact `float32` page arrays, spatial
borders, temporal clipping, pulse scale, nonfinite input sanitisation, output
naming/orientation, partial TIFF behavior, sidecar ordering/failure,
cancellation, progress, garbage collection, helper compatibility and a fresh
process without Qt/UI/worker imports. The existing ignored Pillow
`AppendingTiffWriter` closed-file warning remains; writer lifecycle was not
redesigned in this extraction. Scientific review remains completed on
2026-07-16.

## Retrieval questions

- What must match between RADEN sample and open-beam stacks?
- How does RADEN pulse scaling work?
- How is the output TIFF named?
- What remains after RADEN normalisation is cancelled?
