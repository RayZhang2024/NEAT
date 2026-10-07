---
title: Preprocessing folder layouts and batch detection
doc_id: neat-tech-data-folder-layouts
doc_type: technical_reference
functional_area: data
audience: [user, developer]
neat_version: 4.8.0
verified_commit: 51e65cdade5fb2b9bd89f9aa4a7ddd02c523f7e5
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/services/preprocessing_layout.py, NEAT/services/preprocessing_full_process.py, NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [immediate_child_directories, discover_classic_batch, classify_standalone_summation, discover_full_process_summation, FullProcessPipeline.maybe_do_summation, PreprocessingMixin.add_outlier_images, PreprocessingMixin.add_overlap_correction_images, PreprocessingMixin.add_normalisation_data_images, PreprocessingMixin.add_summation_images, PreprocessingMixin._classify_normalisation_folder, FullProcessWorker.maybe_do_summation]
test_paths: [tests/test_preprocessing_layout.py, tests/test_preprocessing_full_process.py]
---

# Preprocessing folder layouts and batch detection

## Shared classic batch discovery

The headless `NEAT.services.preprocessing_layout` module provides
`discover_classic_batch()` for Clean, Overlap Correction, and the classic
batch-discovery part of Normalisation. If the selected folder has immediate
child directories, each child is a dataset in the order returned by
`os.listdir()`. Otherwise the selected folder is the single dataset for Clean
and Overlap; Normalisation retains its existing single-run classification.
Grandchildren are not promoted to datasets.

This discovery checks directory presence only. Empty or unrelated immediate
directories count exactly like any other child and can change batch selection.
It does not inspect files or classify folders by their contents. No sorting is
applied, so filesystem enumeration order is preserved.

RADEN-versus-classic detection remains a separate concern. Normalisation first
uses the existing `_classify_normalisation_folder()` behavior on the selected
folder; a detected RADEN TIFF stack follows the existing RADEN path and skips
classic batch discovery. Classic folders then use the shared discovery rule.

## Summation layouts

Standalone Summation has its own rules and is not the shared classic batch
rule. Selecting a folder requires at least two immediate child directories;
that check still happens when the folder is selected. It then permits one of
two uniform directory structures:

```text
parent/                    parent/
  run_1/                     sample_A/
  run_2/                       run_1/
                              run_2/
                            sample_B/
                              run_1/
                              run_2/
```

The left layout is described in code as two-level (`parent -> runs`). The right
is three-level (`parent -> samples -> runs`). Mixing immediate sample
directories where some contain immediate child directories and others do not
is rejected. This is determined only from directory presence; empty and
unrelated directories count, and image contents are not inspected by the
layout helper.

For the three-level form, each sample is a separate summation job and must have
at least two run folders. That per-sample check remains delayed until the user
starts summation, after the output-folder and base-name checks. During lazy
loading, the suffix set must match across runs. A mismatch aborts rather than
summing only the intersection.

Output subfolders are created per job:

- two-level: `Summed_<selected-parent-name>`
- three-level: `Summed_<sample-name>`

## Full Process differs

Full Process treats the sample and open-beam selections independently. It uses
the same immediate-directory enumeration primitive, but keeps its own rules:
no immediate children means summation is skipped; one or more immediate
children means it attempts summation over those children. In particular, one
child is still passed through the existing load-and-sum path; Full Process
does not require two children and does not support the extra
`parent -> sample -> runs` grouping layer. Grandchildren are not promoted to
top-level runs.

`discover_full_process_summation()` exposes the immediate folders and this
`should_sum` decision to `FullProcessPipeline`. Its decision depends only on
whether the immediate-folder list is empty; it does not impose the standalone
Summation minimum of two folders. The Qt worker delegates to the pipeline and
does not reclassify folder contents.

Empty or unrelated children still affect the discovered child count and the
existing Full Process classification. Full Process subsequently tries to load
each child and retains its current behavior for children with no valid images;
that content handling is not part of the headless layout helper.

Full Process uses the final nonempty underscore-delimited filename component
as its frame suffix. Unlike the standalone loader's numeric convention, this
suffix may be nonnumeric. Duplicate suffixes are load failures.

## Recommended deterministic layout

Keep image frames and their sidecars together in one run folder. Do not place
unrelated directories below a selected batch parent. Use identical, unique
frame suffixes across runs intended for summation and across each
sample/open-beam pair.

## Known limitations

All layout classification is based on the presence of immediate directories,
not their content. An empty or unrelated child can therefore change how a
selection is interpreted. This limitation is intentional for compatibility;
the helper does not inspect FITS/TIFF files, sidecars, spectra, shutter counts,
or scientific metadata.

## Retrieval questions

- How should I arrange folders for Summation?
- Why was a selected parent treated as a batch?
- Can Full Process use the same three-level layout as standalone Summation?
- Why did mixed-depth folders get rejected?
