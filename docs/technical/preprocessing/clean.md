---
title: Clean (invalid-pixel and positive-spike replacement)
doc_id: neat-tech-preprocessing-clean
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.remove_outliers, OutlierFilteringWorker]
test_paths: [tests/test_preprocessing_workers.py]
---

# Clean

## Purpose

The **Clean** block replaces nonpositive/nonfinite pixels and unusually high
positive spikes using local positive-neighbor means.

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
neighborhood.

## Outputs

- image: `<base>_<suffix>.fits`
- report: `<base>_outlier_report.csv`
- first recognized sidecars copied as `Run<index>_<original-name>`

The report columns are:

```text
frame_idx,pixel_x,pixel_y,outlier_value,replace_value
```

`frame_idx` is the filename suffix, not necessarily a zero-based frame number.
Coordinates are array column `x` and row `y`. Both invalid-pixel and positive
spike replacements are recorded.

## Progress, failure and cancellation

Progress counts frames successfully passed through `_clean_one_frame`. A frame
write or processing error makes `succeeded=False`; other frames can still be
processed. Stop is checked between runs and frames. The report is recreated at
the start of each job.

## Limitations

- Clean is intended for the confirmed preprocessing stage where zero and
  negative values are physically invalid.
- In-place sequential replacement makes the result order-dependent.
- An invalid value remains when neither the 5x5 nor 7x7 search has a positive
  finite neighbor.
- The method does not propagate replacement uncertainty.
- Focused tests verify 5x5 replacement, 7x7 fallback, exact 10x spike
  detection, CSV coordinates and the no-valid-neighbor case.

## Retrieval questions

- Which pixels does Clean classify as outliers?
- How is a bad pixel replaced?
- When does Clean expand from 5x5 to 7x7?
- How does Clean identify a high positive spike?
- Why can a NaN or negative pixel remain after Clean?

