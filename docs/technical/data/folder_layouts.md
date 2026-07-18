---
title: Preprocessing folder layouts and batch detection
doc_id: neat-tech-data-folder-layouts
doc_type: technical_reference
functional_area: data
audience: [user, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.add_summation_images, PreprocessingMixin.add_outlier_images, PreprocessingMixin.add_overlap_correction_images, FullProcessWorker.maybe_do_summation]
test_paths: []
---

# Preprocessing folder layouts and batch detection

## General batch convention

For Clean and Overlap Correction, selecting a folder with immediate child
folders makes every child a separate dataset processed sequentially. If there
are no child folders, the selected folder itself is one dataset. Nested
grandchildren are not recursively discovered by these blocks.

Normalisation similarly classifies selected sample datasets and uses the first
loaded open-beam run. RADEN stack detection follows a separate classification
path.

## Summation layouts

Standalone Summation requires at least two immediate children and permits one
of two uniform structures:

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
is three-level (`parent -> samples -> runs`). Mixing samples that have
subfolders with children that contain images directly is rejected.

For the three-level form, each sample is a separate summation job and must have
at least two run folders. During lazy loading, the suffix set must match across
runs. A mismatch aborts rather than summing only the intersection.

Output subfolders are created per job:

- two-level: `Summed_<selected-parent-name>`
- three-level: `Summed_<sample-name>`

## Full Process differs

Full Process treats the sample and open-beam selections independently. If a
selected folder has any immediate child folders, it attempts summation across
all valid child folders. If it has none, summation is skipped. Unlike the
standalone Summation UI, Full Process does not require at least two children and
does not support the extra `parent -> sample -> runs` grouping layer.

Full Process uses the final nonempty underscore-delimited filename component
as its frame suffix. Unlike the standalone loader's numeric convention, this
suffix may be nonnumeric. Duplicate suffixes are load failures.

## Recommended deterministic layout

Keep image frames and their sidecars together in one run folder. Do not place
unrelated directories below a selected batch parent. Use identical, unique
frame suffixes across runs intended for summation and across each
sample/open-beam pair.

## Known limitations and test gap

Folder classification is based on the presence of directories, not their
content. An empty or unrelated child can therefore change how a selection is
interpreted. No focused automated folder-classification test was identified.

## Retrieval questions

- How should I arrange folders for Summation?
- Why was a selected parent treated as a batch?
- Can Full Process use the same three-level layout as standalone Summation?
- Why did mixed-depth folders get rejected?
