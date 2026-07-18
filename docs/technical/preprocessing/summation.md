---
title: Summation
doc_id: neat-tech-preprocessing-summation
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: domain-reviewed
instrument_applicability: [classic image folders]
scientific_review: completed 2026-07-16
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.add_summation_images, SummationWorker]
test_paths: [tests/test_preprocessing_workers.py]
---

# Summation

## Purpose and use

Summation adds corresponding detector images from repeated runs pixel by
pixel. It is valid only when the runs use identical acquisition settings and
measure the same object. The code also combines shutter counts and preserves
each run's spectrum separately.

## Inputs and UI

The user selects a parent through **Add data**, an existing output directory, a
required base name, then **Sum**. Folder structures are described in
[folder layouts](../data/folder_layouts.md). All runs in a job must expose the
same frame suffix set. Corresponding images must have the same shape. Every run
must have a readable two-column shutter-count table of the same length.

## Algorithm

For every sorted suffix `k`, each image is converted to `float32` and accumulated:

```text
S_k(x, y) = image_1,k(x, y) + image_2,k(x, y) + ... + image_N,k(x, y)
```

No division, exposure normalization, clipping or uncertainty propagation is
performed. All suffix sets and image shapes are validated before output is
written; any mismatch aborts the job.

For each run, the worker locates the first `_ShutterCount.txt`, loads it as a
two-dimensional two-column table and uses column 2. Arrays with compatible
lengths are added. It writes a new two-column table containing a zero-based row
index and the summed counts. Output uses integer formatting.

Spectra are not summed. The first `_Spectra.txt` found for each run is copied
and numbered by run.

## Outputs

- images: `<base>_Summed_<suffix>.fits`
- shutter counts: `<base>_summed_ShutterCount.txt`
- spectra: `<base>_<run-index>_Spectra.txt`

Existing FITS outputs are overwritten. Output images are `float32` and written
with NEAT's orientation helper.

## Errors, progress and cancellation

The worker checks for absent or malformed runs, unequal suffix sets, non-2D or
shape-mismatched images, unreadable shutter sidecars and incompatible
shutter-count lengths before writing output. Any of these conditions makes
`succeeded=False` and aborts the job. Progress is based on processed suffixes.
Stop is cooperative and checked between units of work.

## Known limitations

- Acquisition-setting equivalence is a user prerequisite; it cannot be
  verified from the image arrays alone.
- More than one candidate sidecar has nondeterministic “first match” behavior.
- Existing output directories are reused and matching filenames overwritten.
- Focused tests cover corresponding image/shutter summation, spectra copying
  and pre-write rejection of suffix and shutter-length mismatches.

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
