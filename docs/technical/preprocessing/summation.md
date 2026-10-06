---
title: Summation
doc_id: neat-tech-preprocessing-summation
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: f48bc0b6d887a865603ddb6dfc8583512d33bbd7
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_summation.py, NEAT/services/image_io.py, NEAT/domain/preprocessing_inputs.py, NEAT/domain/preprocessing.py]
source_symbols: [PreprocessingMixin.add_summation_images, SummationWorker, sum_loaded_image_runs, sum_corresponding_frames, LoadedImageRun, PreprocessingOperationResult]
test_paths: [tests/test_preprocessing_summation.py, tests/test_preprocessing_workers.py, tests/test_image_io_orientation.py]
---

# Summation

## Purpose and use

Summation adds corresponding detector images from repeated runs pixel by
pixel. It is valid only when the runs use identical acquisition settings and
measure the same object. The headless `sum_loaded_image_runs()` service owns
validation, arithmetic, sidecar processing and output persistence. The Qt
`SummationWorker` remains the GUI adapter: its existing dictionary inputs are
converted to `LoadedImageRun` values, it delegates the operation, relays
progress/messages, and exposes the final `PreprocessingOperationResult` and
legacy `succeeded` flag.

## Inputs and UI

The user selects a parent through **Add data**, an existing output directory, a
required base name, then **Sum**. Folder structures are described in
[folder layouts](../data/folder_layouts.md). All runs in a job must expose the
same frame suffix set. Corresponding images must have the same shape. Every run
must have a readable two-column shutter-count table of the same length.

## Algorithm

For every lexicographically sorted suffix `k`, each image is converted to
`float32` and accumulated in logical run order:

```text
S_k(x, y) = image_1,k(x, y) + image_2,k(x, y) + ... + image_N,k(x, y)
```

The operation uses `image.astype(np.float32, copy=False)` for each frame and
copies the first converted frame before adding later frames in place. No
division, exposure normalization, clipping or uncertainty propagation is
performed. All suffix sets, two-dimensional image shapes, shape agreement,
loader errors and shutter-count inputs are validated before the output folder
is created; a validation failure has `processed_count=0` and
`expected_count=None`.

For each logical run, the service visits physical `source_folders` in supplied
order and uses the first unsorted `os.listdir()` entry ending exactly in
`_ShutterCount.txt`. It loads each table as `float32`, requires two columns,
and adds column 2 sequentially in place within the logical run. These per-run
arrays are then combined with NumPy's `sum(..., axis=0)` in logical-run order.
The output contains a zero-based row index and summed counts, formatted with
`%d\t%d`.

Spectra are not summed. The first unsorted entry ending exactly in
`_Spectra.txt` is copied from the first matching physical folder for each
logical run, and numbered by logical run. Missing files and copy failures are
nonfatal warnings; a copy failure retains the legacy follow-up “No Spectra
file found” message.

## Outputs

- images: `<base>_Summed_<suffix>.fits`
- shutter counts: `<base>_summed_ShutterCount.txt`
- spectra: `<base>_<run-index>_Spectra.txt`

Existing output directories are reused and matching outputs are overwritten.
Output images are `float32` and written with the shared, headless orientation
helper in `NEAT.services.image_io`; the established import through
`NEAT.workers.batch.write_fits_image_file` remains available.

The ordered `PreprocessingOperationResult.outputs` entries use the roles
`summed_image`, `summed_shutter_count`, and `spectrum_copy`. `processed_count`
counts successfully written summed images only; sidecar artifacts do not
increment it. After validation, `expected_count` is the number of suffixes,
including on later failure or cancellation. Previously written files are not
rolled back.

## Errors, progress and cancellation

The service checks for absent or malformed runs, unequal suffix sets, non-2D
or shape-mismatched images, unreadable/malformed shutter sidecars and
incompatible shutter-count lengths before writing output. Any such validation
failure produces `FAILED`, `processed_count=0`, and no expected count. Later
failures preserve already produced files and report the count of successfully
written images. Cancellation is cooperative: it is checked between suffixes,
while collecting a suffix's run frames, before sidecar phases and between
logical-run spectra copies. Existing partial files remain, with status
`CANCELLED` and the current image count. Worker progress and message signals
retain the prior user-visible behavior.

## Known limitations

- Acquisition-setting equivalence is a user prerequisite; it cannot be
  verified from the image arrays alone.
- More than one candidate sidecar has filesystem-dependent “first match” behavior;
  the directory enumeration is intentionally not sorted.
- Existing output directories are reused and matching filenames overwritten.
- The golden regression was captured from the original `SummationWorker` at
  `e6e0e8cef9806d1f5000c8f38b7b2509c8a3618e`: `gold_Summed_10.fits` is
  `[[1, 1], [10, 13]]`, `gold_Summed_2.fits` is
  `[[3.5, 6], [10.25, 4]]`, and the shutter text is `0\t1` then `1\t7`.
- Focused tests cover that numerical baseline, float32/run-order behavior,
  validation-before-output, partial failure, cancellation, spectra warnings,
  headless import, worker adaptation and FITS orientation compatibility.

## Worked example

Runs A and B each contain `scan_00001.fits` and `scan_00002.fits`. Pixel
`(10, 20)` has values 12 and 15 in frame `00001`; the summed output value is
27. The output names are `Fe_Summed_00001.fits` and
`Fe_Summed_00002.fits`.

## Review record

Code verification and domain review are complete for the behavior above.
RAG approval remains pending until the revised working tree completes the full
test and knowledge-evaluation process.

## Retrieval questions

- What exactly does Summation add?
- Why did Summation reject runs with different frame suffixes?
- Are spectra and shutter counts summed in the same way?
- When should repeated runs not be summed?
