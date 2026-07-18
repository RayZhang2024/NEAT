---
title: Supported preprocessing images and sidecars
doc_id: neat-tech-data-images-sidecars
doc_type: technical_reference
functional_area: data
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general, IMAT-style image folders, RADEN]
scientific_review: pending
source_paths: [NEAT/workers/batch.py, NEAT/workers/preprocessing.py]
source_symbols: [load_image_file, write_fits_image_file, ImageLoadWorker, FullProcessWorker.load_run_dict]
test_paths: [tests/test_image_io_orientation.py, tests/test_preprocessing_tiff.py]
---

# Supported preprocessing images and sidecars

## Scope

This page records the file rules actually enforced by preprocessing code. It
does not establish which instrument exports are scientifically interchangeable.

## Classic image folders

The general preprocessing loader accepts `.fits`, `.fit`, `.tiff` and `.tif`
(case-insensitive). `load_image_file` can also read `.fts`, although the folder
scanners used by preprocessing do not select that extension.

Each filename must end in an underscore followed by a numeric identifier:

```text
sample_00001.fits
sample_00002.tiff
```

`ImageLoadWorker` accepts 1–10 digits. The identifier is retained as a string
and becomes the frame key. Files are first sorted lexicographically. If two
files have the same suffix, the later loaded image replaces the earlier one and
a warning is emitted.

The Full Process loader accepts the final nonempty underscore-delimited stem
component as its suffix, including nonnumeric values. For example,
`sample_frame.tiff` uses `frame`. Duplicate suffixes are reported as load
errors instead of silently replacing an earlier image.

## Orientation and output format

FITS and TIFF images are flipped along the row axis on loading so NEAT's
in-memory orientation matches the ImageJ-style display convention. FITS output
is flipped on writing. Consequently, reading and writing through the NEAT
helpers preserves the displayed orientation.

Most classic preprocessing operations write FITS even when the source is TIFF.
The original FITS header is not carried into these generated files. Arrays are
normally converted to `float32` by the numerical workers.

## Classic sidecars

Several blocks recognize files whose names end exactly with:

- `_Spectra.txt`
- `_ShutterCount.txt`

Matching is case-sensitive. Some workers copy only the first directory match;
others copy all matches. Directory enumeration order is not explicitly sorted,
so “first” should not be relied upon when several candidates exist.

The expected numeric content differs by block:

- Summation expects a two-column shutter-count table and sums column 2.
- Standard normalisation reads column 2 of the first line only.
- Overlap correction loads the full table and the UI flattens all nonzero
  values before the correction worker selects values greater than 1000.

These are separate implementation contracts, not a single universal sidecar
schema.

## RADEN stack files

RADEN normalisation uses a multi-page TIFF stack plus metadata discovered by
`get_raden_tiff_stack_info`. Sidecars can include `.stat`, `.json` and `.log`.
The detailed metadata-discovery rules belong to Batch 2. RADEN sample and
open-beam inputs must both be detected as RADEN stacks; they cannot be mixed
with classic image folders.

## Failure and skip behavior

Unreadable images and duplicate or empty suffixes are recorded as load errors.
Unsupported extensions are not selected. Full Process aborts when a required
stage sees load errors or incomplete worker output. Standalone preprocessing
workers expose a `succeeded` flag, while successfully written frames can remain
after a later failure.

## Known limitations

- Suffix parsing still differs between the general `ImageLoadWorker` numeric
  rule and Full Process's relaxed final-component rule.
- `.fts` is supported by the low-level reader but not selected by preprocessing
  folder scanners.
- Generated FITS files do not preserve source headers.
- Multiple sidecars with the same recognized suffix can be handled
  inconsistently.

## Source traceability and review

Implementation: `load_image_file`, `write_fits_image_file`,
`ImageLoadWorker.run`, and `FullProcessWorker.load_run_dict`.

Tests verify FITS orientation round trips and the Full Process TIFF helper.
There is no consolidated test of every extension, suffix rule and sidecar
variant.

Scientific/instrument review is required before RAG approval, particularly for
which classic layouts correspond to IMAT exports and which metadata are
essential for quantitative use.

## Retrieval questions

- Which image extensions can preprocessing load?
- Why did Full Process ignore files that another block loaded?
- What filename suffix does NEAT require?
- Does preprocessing preserve FITS headers and image orientation?
