---
title: RADEN TIFF stacks and metadata
doc_id: neat-tech-loading-raden
doc_type: technical_reference
functional_area: loading
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [RADEN]
scientific_review: pending
source_paths: [NEAT/workers/batch.py, NEAT/ui/mixins/fitting.py]
source_symbols: [parse_raden_stat_file, parse_raden_json_file, get_raden_tiff_stack_info, load_raden_tiff_stack, RadenTiffStackLoadWorker]
test_paths: [tests/test_fitting_headless.py]
---

# RADEN TIFF stacks and metadata

## TIFF and sidecar selection

Input may be a TIFF path or folder. A folder with one TIFF uses it; with several,
the largest file is chosen. Metadata preference is same-stem `.stat`, then
same-stem `.json`, then a sole file of either extension in the folder.

The TIFF must contain more than one page. Metadata ToF bin count must equal the
page count.

## Metadata parsing

`.stat` requires X, Y and TOF lines with bin count, minimum, maximum, bin size
and units. Entries, pulses and pulses-with-data are captured when present.
JSON reads `x_*`, `y_*` and `t_*` parameters, assigns millimetres to spatial
axes and milliseconds to ToF, and derives bin size.

ToF bin centers are generated uniformly between metadata minimum and maximum.
Supported units are microseconds, milliseconds and seconds (including listed
abbreviations), converted internally to microseconds.

## Frame loading

Every TIFF page becomes a `float32` frame; nonfinite values become zero and the
frame is flipped vertically. Wavelengths are computed using the chosen app
flight path. Loading supports progress callbacks and cooperative cancellation.

## Limitations

- Selecting the largest TIFF is a heuristic.
- JSON units are assumed rather than read dynamically.
- Uniform bin edges are reconstructed from min/max/bins; explicit irregular
  axes are not supported.
- Spatial X/Y metadata are parsed but not used to reshape pages.
- Instrument verification is required for orientation and metadata semantics.

## Retrieval questions

- Which RADEN TIFF and metadata files does NEAT choose?
- Why must ToF bins equal TIFF pages?
- Which RADEN ToF units are supported?
- How are RADEN TIFF pages oriented and converted?

