---
title: Full Process preprocessing pipeline
doc_id: neat-tech-preprocessing-full-process
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [classic image folders]
scientific_review: pending
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.run_full_process, FullProcessWorker]
test_paths: [tests/test_preprocessing_tiff.py]
---

# Full Process preprocessing pipeline

## Scope

Full Process chains existing classic-folder workers. It is not the RADEN
multi-page TIFF workflow and does not include Filtering.

## UI inputs

The user supplies sample folder, open-beam folder, an existing output parent,
and optionally:

| Field | Fallback |
|---|---:|
| base name | `FullProcess` |
| spatial half-window `n` | 10 |
| adjacent-frame half-window `m` | 0 |

Both values must be integers. `n` must be between 0 and 100 inclusive; `m` must
be between 0 and 10 inclusive. Invalid values prevent the pipeline from
starting. The entered overall base name is stored but stage-specific output
names are used instead.

## Executed sequence

```text
sample:    optional Sum -> Clean -> required Overlap --\
                                                        -> Normalise
open beam: optional Sum -> Clean -> required Overlap --/
```

1. **Summation:** performed separately if the selected root contains any
   immediate child folders; otherwise skipped.
2. **Clean:** always attempted for sample and open beam.
3. **Overlap:** required for both. Missing images, spectra, shutter counts or
   incomplete correction aborts Full Process.
4. **Normalisation:** applies the selected `n` and `m`.

Full Process accepts FITS/TIFF names whose final underscore-delimited component
is a nonempty frame suffix. The suffix no longer has to contain five digits.
Duplicate suffixes and unreadable frames are recorded as load failures.

## Output layout

Stage folders are created under the selected output parent:

```text
0_summed_<input-folder>/
1_cleaned_<previous-folder>/
2_corrected_<previous-folder>/
3_normalised_original/
```

As names include the previous stage's folder name, repeated prefixes can
accumulate. The final frames are named `normalised_<suffix>.fits`; the overall
base-name field does not control them.

Summation alone is skipped when an input root has no immediate run subfolders.
Processing then begins with Clean on that root.

## Threading, progress and stop behavior

Full Process is itself a worker thread. Each stage creates another worker and
waits through a nested Qt event loop. Stage progress is forwarded directly and
reset to zero between stages; it is not a single monotonic whole-pipeline
percentage.

The parent stop flag is checked between stages and is forwarded to the active
child worker. Cancellation remains cooperative, so the child stops at its next
safe check rather than being forcibly terminated.

## Known implementation risks

- One child folder triggers summation even though standalone Summation requires
  at least two.
- Partial output from a prior run is not cleared before `exist_ok=True` folders
  are reused.
- Worker success is an in-memory flag, not a persistent manifest of expected
  and produced files.
- Tests cover relaxed suffix loading, required-overlap sidecars and forwarding
  Stop to the active child. The complete end-to-end sequence remains untested.

## Required review before use as assistant guidance

The ordering and requirement to perform overlap correction were confirmed in
the Batch 1 review. RAG approval still requires a defined persistent
output-completeness check and resolution of the remaining overlap-correction
scientific questions.

## Retrieval questions

- Which operations does Full Process run and in what order?
- When does Full Process skip Summation or Overlap Correction?
- Why do Full Process filenames differ from the base name I entered?
- Why did Stop wait until the current stage ended?
