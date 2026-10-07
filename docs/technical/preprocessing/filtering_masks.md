---
title: Filtering and masks
doc_id: neat-tech-preprocessing-filtering
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: eb605978e4cbc52630ff1138866d6786c2208b0d
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_filtering.py, NEAT/services/image_io.py, NEAT/domain/preprocessing_inputs.py, NEAT/domain/preprocessing.py, NEAT/ui/dialogs.py]
source_symbols: [FilteringWorker, MaskGeneratorDialog, filter_loaded_image_runs, apply_binary_mask, LoadedImageRun, PreprocessingOperationResult]
test_paths: [tests/test_preprocessing_filtering.py, tests/test_preprocessing_workers.py, tests/test_image_io_orientation.py]
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
operation before outputs. Mask and image shapes must match exactly, but this
comparison occurs per frame; a mismatching frame is skipped while later frames
continue. There is no 2-D-only or boolean-only service validation.

## UI and generated masks

Users may load one mask or open **Generate Mask**. The generator can load
FITS/TIFF data, create a finite threshold mask, invert the threshold direction,
and edit it with brush, rectangle or polygon tools. Its generated mask is
`uint8`, with nonzero pixels retained.

The selected output directory must already exist and a base name is required.

## Processing and outputs

The GUI-independent `filter_loaded_image_runs()` service accepts ordered
`LoadedImageRun` inputs, validates the mask, applies it, copies sidecars,
writes FITS, and returns a `PreprocessingOperationResult`. `FilteringWorker`
is the Qt compatibility adapter: it converts the existing run dictionaries,
relays messages/progress, and retains both `result` and legacy `succeeded` and
`failed_frames` state. Filtering remains standalone, not part of Full Process.

The pure `apply_binary_mask()` transform has no I/O or Qt dependency. It
converts image data to `float32` only if needed and uses `np.where(mask == 1,
image, 0)` without mutating image or mask inputs. Kept NaN/Inf image values are
not cleaned. A successfully validated mask becomes the worker's public
`filtering_mask` as `float32`; failed validation leaves that attribute pointing
to the original supplied object.

Runs are processed in supplied order. Within each run, frame suffixes are
**sorted lexicographically**, unlike Clean's insertion order. Shape mismatches,
FITS-write failures and other individual frame errors are recorded and do not
stop later frames. A missing output directory is not created or rejected up
front by the service; writes may fail frame by frame.

Output images are `<base>_<suffix>.fits`. The first spectra and first shutter
sidecar from each run are copied as `Run<index>_<original-name>` using
`shutil.copyfile`. Sidecars use only `primary_source` (legacy `folder_path`),
not `source_folders`. One unsorted `os.listdir()` supplies both first matches
per run. Missing folders/sidecars produce informational messages. Enumeration
or copy exceptions are nonfatal warnings; a failed Spectra copy stops the
ShutterCount attempt for that run. Nonempty `load_errors` alone do not reject
Filtering.

`expected_count` is all supplied frames; `processed_count` is successfully
written filtered FITS images. Sidecars do not increment it. Successful
sidecars/images remain in the ordered `outputs` with roles `related_file_copy`
and `filtered_image` after later failure or cancellation. No rollback occurs.

## Progress and cancellation

Progress has two streams: after a successful image write it emits
`int(processed_count / expected_count * 100)`; after each run's frame loop it
emits `int(run_index / run_count * 100)`. Consequently progress can move
backward or reach 100 despite failure/cancellation. A valid zero-frame run
still handles sidecars, emits run progress and succeeds with a 0/0 summary.

Stop is cooperative before runs and frames, not within a frame or sidecar
copy. The service validates the mask and emits `Filtering started...` before
observing a pre-existing stop. It may emit the stop-observation message at a
frame boundary and again at the next run boundary; it then emits the legacy
failed/incomplete summary. No-runs and no-mask inputs keep their existing
messages and FAILED service results, but the worker now emits its public
`finished` signal once on both paths (Issue #41). Like the other preprocessing
workers, it retains the structured result and derives `succeeded` from that
result. Terminal adapter-only cleanup failure is reported as a best-effort
`[WARN] Worker finalization: <error>` without changing the scientific result;
service failures remain operation errors. The worker's public
`copy_related_files()` and `get_short_path()` remain callable;
`output_folder_short` is set only on paths reaching the normal final summary.

## Limitations

- Zeroing discarded pixels may cause Clean to treat them as invalid if Clean is
  run afterward.
- No uncertainty or masked-pixel metadata are written into FITS.
- Focused headless and worker tests verify binary-mask semantics, validation,
  sorted frame order, sidecar behavior, partial outputs, dual progress,
  cancellation and completion state.
  Interactive mask-editor behavior remains without dedicated coverage.
- Filtering is not part of Full Process. If used separately, it should not be
  followed by Clean because Clean treats the intentionally zeroed region as
  invalid pixels.

## Numerical-equivalence reference

The golden fixture in `tests/test_preprocessing_filtering.py` was executed
through the original `FilteringWorker` at the Issue #31 baseline
`44c62653f27899506dbf7d19cdf1249fafe6ed11` before extraction. It uses two
runs, mixed source dtypes, a 0/1 mask, insertion order different from sorted
suffix order, both sidecar types, and kept infinity. Recorded output arrays
and FITS orientation are compared exactly (`rtol=0`, `atol=0`); output roles,
names, messages and progress `[33, 66, 50, 100, 100]` are asserted.

## Retrieval questions

- Does Filtering smooth the image?
- Which mask values keep a pixel?
- Why was an image skipped during Filtering?
- What happens if I run Clean after applying a mask?
