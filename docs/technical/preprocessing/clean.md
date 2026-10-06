---
title: Clean (invalid-pixel and positive-spike replacement)
doc_id: neat-tech-preprocessing-clean
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: c63137362dbf26ee1c40481afacea57fb40e47e5
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_clean.py, NEAT/services/image_io.py, NEAT/domain/preprocessing_inputs.py, NEAT/domain/preprocessing.py]
source_symbols: [PreprocessingMixin.remove_outliers, OutlierFilteringWorker, clean_loaded_image_runs, clean_outlier_frame, LoadedImageRun, PreprocessingOperationResult]
test_paths: [tests/test_preprocessing_clean.py, tests/test_preprocessing_workers.py, tests/test_image_io_orientation.py]
---

# Clean

## Purpose

The **Clean** block replaces nonpositive/nonfinite pixels and unusually high
positive spikes using local positive-neighbor means. The GUI-independent
`clean_loaded_image_runs()` service owns processing, report and image writing,
sidecar copying, progress, and cancellation. `OutlierFilteringWorker` keeps the
existing Qt constructor and signals, converts legacy run dictionaries into
`LoadedImageRun`, and retains both its legacy state and a structured
`PreprocessingOperationResult`.

## Inputs and UI

Select one dataset folder, or a parent whose immediate children are datasets;
choose an existing output parent and enter a required base name. Each dataset
is written below `outlier_removed_<dataset-name>`.

## Invalid-pixel replacement

An input image is copied to `float32`. A pixel is invalid when:

```text
value <= 0, or value is NaN, or value is positive/negative infinity
```

For invalid pixel `(x, y)`, Clean first takes the clipped 5x5 neighborhood
centered on it. Only finite values greater than zero are valid neighbors and
the center is excluded. If no valid neighbor exists, the search expands to the
clipped 7x7 neighborhood.

The replacement is the arithmetic mean of the valid neighbors. If the 7x7
search also finds none, the report records `NaN` and the original invalid
value remains unchanged.

Invalid pixels are processed in the array order returned by `np.argwhere`.
Earlier replacements can therefore contribute to a later neighborhood.

## Positive-spike replacement

After invalid pixels are processed, Clean scans every finite positive pixel.
It calculates the clipped 5x5 positive-neighbor mean, excluding the center. If:

```text
pixel value >= 10 x positive-neighbor mean
```

the pixel is classified as a positive spike and replaced by that mean. Spike
replacement is sequential, so an earlier replacement can affect a later
neighborhood. The pure `clean_outlier_frame()` transform owns this algorithm:
it copies the input to `float32`, does no I/O, and does not mutate the loaded
frame. Its cached candidate scan retains the original order-sensitive results.

## Outputs

- image: `<base>_<suffix>.fits`
- report: `<base>_outlier_report.csv`
- first recognized sidecars copied as `Run<index>_<original-name>`

The service visits runs and frame mappings in supplied insertion order. For
each run it searches only `LoadedImageRun.primary_source` (the legacy
`folder_path`) for the first unsorted `_Spectra.txt` and `_ShutterCount.txt`
entries. It copies matches with `copy2`; missing matches are silent. It does
not use `source_folders` for Clean sidecars, and nonempty `load_errors` alone
do not reject a run.

The report columns are:

```text
frame_idx,pixel_x,pixel_y,outlier_value,replace_value
```

`frame_idx` is the filename suffix, not necessarily a zero-based frame number.
Coordinates are array column `x` and row `y`. Both invalid-pixel and positive
spike replacements are recorded.

The output directory is created/reused and the report is freshly created with
its header **before** any run or frame. Each frame appends its rows, including
a blank newline when there are no outliers, before the FITS write. A failed
FITS write does not roll back report rows or prior artifacts; later frames
continue. The ordered result records the report, successful sidecar copies,
and written images with roles `outlier_report`, `related_file_copy`, and
`cleaned_image`.

## Progress, failure and cancellation

`expected_count` is the total number of supplied frames, including zero;
`processed_count` counts only successfully written cleaned FITS frames.
An empty run sequence succeeds with a report and no progress update. Frame
failures are recorded in `errors`, set final status `FAILED`, and do not block
later frames. Report-append or sidecar-copy failures are nonfatal warnings;
sidecar directory-enumeration and report-creation failures are fatal. Partial
outputs remain listed in the result and on disk. Progress advances only after
successful image writes and is not artificially set to 100% after failure.

Stop is checked before each run and frame, not within a frame or sidecar copy.
The result is `CANCELLED` while retaining prior outputs and errors. The legacy
end-of-run summary is still emitted after frame failures or cancellation; its
cleaned-pixel count includes only frames whose FITS write succeeded. The
worker's `succeeded` flag is true only for `SUCCEEDED`.

## Limitations

- Clean is intended for the confirmed preprocessing stage where zero and
  negative values are physically invalid.
- In-place sequential replacement makes the result order-dependent.
- An invalid value remains when neither the 5x5 nor 7x7 search has a positive
  finite neighbor.
- The method does not propagate replacement uncertainty.
- Focused tests verify 5x5 replacement, 7x7 fallback, exact 10x spike
  detection, CSV coordinates and the no-valid-neighbor case.

## Numerical-equivalence reference

The golden fixture in `tests/test_preprocessing_clean.py` was run through the
original `OutlierFilteringWorker` at the Issue #29 baseline commit
`56ddb9e8e10e9faaa8505ccdfac2a43531772268`, before extraction. It contains
nonpositive/nonfinite values, clipped 5x5 and 7x7 neighborhoods, an unresolved
pixel, an exact-threshold boundary spike, a near-threshold non-spike, and
adjacent order-sensitive spikes. The recorded cleaned `float32` arrays are
compared exactly (`rtol=0`, `atol=0`), and the complete ordered report is
protected by a SHA-256 digest of its normalized text, with representative
rows asserted directly. Output names, order and progress are also asserted.

## Retrieval questions

- Which pixels does Clean classify as outliers?
- How is a bad pixel replaced?
- When does Clean expand from 5x5 to 7x7?
- How does Clean identify a high positive spike?
- Why can a NaN or negative pixel remain after Clean?

