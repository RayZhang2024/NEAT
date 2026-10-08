---
title: Full Process preprocessing pipeline
doc_id: neat-tech-preprocessing-full-process
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: d3de8d15895fa577a6a8bfcd060ea0f48c56f94f
status: code-verified
instrument_applicability: [classic image folders]
scientific_review: pending
source_paths: [NEAT/ui/preprocessing_worker_registry.py, NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_full_process.py, NEAT/services/image_io.py]
source_symbols: [PreprocessingMixin.run_full_process, PreprocessingMixin.stop_full_process, PreprocessingWorkerRegistry, FullProcessWorker, FullProcessPipeline, load_full_process_run, run_full_process]
test_paths: [tests/test_preprocessing_worker_ownership.py, tests/test_preprocessing_full_process.py, tests/test_preprocessing_tiff.py, tests/test_preprocessing_layout.py, tests/test_image_io_orientation.py]
---

# Full Process preprocessing pipeline

## Scope

Full Process orchestrates the existing classic-folder preprocessing services.
`FullProcessWorker` is a single Qt-thread/signal adapter around the headless
`FullProcessPipeline`; the pipeline does not import Qt, UI, or worker modules,
create threads/event loops, or require a `QApplication`. It is not the RADEN
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

The exact stage order is sample Summation, open-beam Summation, sample Clean,
open-beam Clean, sample Overlap, open-beam Overlap, then Normalisation. Each
Summation is skipped only when its selected root has no immediate child
directories. One child still invokes Summation. Clean always runs for sample
and open beam; Overlap is required for both; then classic Normalisation uses
the selected `n` and `m`.

The headless Full Process loader uses raw `os.listdir()` order and selects
case-insensitive `.fits`, `.fit`, `.tiff`, and `.tif` files. It excludes `.fts`
even though the generic image reader continues to support that extension. The
final underscore-delimited filename component is stripped and used as a
nonempty frame suffix; it need not be numeric. Duplicate suffixes and
unreadable frames are recorded as load failures. Loaded arrays are converted
to `float32`; the shared generic reader preserves the existing FITS/TIFF
vertical-orientation behavior, tries imageio before Pillow for TIFF, and does
not coerce dtype itself. Loading is deliberately not cancellable: once a
loader starts, it processes all eligible files.

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

Directories are created with `exist_ok=True` and are not cleared. Stale
eligible images may therefore be loaded by later stages, but stale files are
not included in the current result unless an operation service reports them.
Accumulated prefixes such as `0_summed_`, `1_cleaned_0_summed_`, and
`2_corrected_1_cleaned_0_summed_` are intentional. The overall base name
remains unused.

When Summation is skipped, Full Process rejects duplicate source Spectra or
ShutterCount sidecars before Clean applies its first-match copy rule. Sidecar
manifests follow the current run through Overlap and Normalisation, so the
normalisation scale and copied sample sidecars come from current-run metadata.
Prior generated sidecars for that run are removed from the reused final output
folder. Standalone Normalisation keeps its legacy folder-discovery behavior.

The pipeline result is immutable and Full Process-specific. It contains the
overall `PreprocessingStatus`, ordered stage records, the failed/cancelled
stage, operation-produced outputs in chronological order, and stage-attributed
errors/warnings. Each stage record contains its identity, sample/open-beam
branch, `SKIPPED`/`SUCCEEDED`/`FAILED`/`CANCELLED` outcome, input/artifact/
propagation folders, and the operation result when a service started. Folder
paths are not represented as `ProducedOutput`; output aggregation never scans
the reused stage directories and preserves duplicate paths reported by a
service.

Skipped Summation has no artifact folder and propagates its input. A setup
failure before a stage directory is created has no artifact folder; a failure
after creation records the directory. On success, artifact and propagation
folders are normally equal. If an operation succeeds after Stop was pressed
during setup/loading, its artifact folder is retained but the helper's
propagation folder falls back to the original input; the next boundary stops
before that fallback is consumed.

## Threading, progress and stop behavior

`FullProcessWorker` owns one QThread. It does not create preprocessing child
workers or use nested `QEventLoop`s. Operation services receive plain progress,
message, and cancellation callbacks. Service progress is forwarded directly;
the worker emits a zero reset after each accepted stage boundary. Progress is
not transformed into a global monotonic percentage. Load progress remains an
independent per-loader stream.

Cancellation has separate parent-boundary and active-operation states. Stop
always clears the parent-running flag. If an operation is active, Stop also
sets that operation's fresh cancellation token and emits the legacy child Stop
diagnostic before the Full Process Stop message (Clean has no child Stop
diagnostic). If Stop occurs during discovery/loading/setup before a service
starts, setup and the loader continue; a service reached afterward receives a
fresh, non-cancelled token and may complete. The next parent boundary then
stops the pipeline. No forceful thread termination is used. Pre-stopped runs
still reach the first historical stage boundary and report its stop message.

When launched from the GUI, `FullProcessWorker` is registered with the
window-owned preprocessing worker registry before start. The GUI Stop handler
delegates to the worker's existing `stop()` and returns without waiting; the
registry retains the QThread until native exit and completion handling are
both confirmed. Structured results remain available through normal completion
handling; after that, or after a cancelled/abnormal worker exits, the registry
clears a convenience reference only if it still points to the retiring worker.
As with other preprocessing families, no subsequent GUI run can reuse that
family while its worker is still retiring. If Qt does not acknowledge startup,
the registry retains ownership and reports the ambiguity instead of treating
the startup timeout as proof of failure. This changes only GUI ownership and
interactive Stop; the pipeline's stage order, one-child Summation behavior,
cancellation safe points, outputs and progress streams are unchanged.

This is not application-close coordination. The current
`closeEvent()`/`cleanup_resources()` path can still accept a close after its
legacy 1000 ms wait while a registry-owned thread remains alive. Safe
close-during-processing remains a follow-up release blocker for the next
Workstream E shutdown issue.

## Known implementation risks

- One child folder triggers Full Process Summation even though standalone
  Summation requires at least two.
- Partial output from a prior run is not cleared before `exist_ok=True` folders
  are reused; stale eligible files may affect later loaders.
- Worker success remains an in-memory result, not a persistent manifest of
  expected and produced files.
- The workflow has exact-array regression evidence, but its overlap-correction
  scientific assumptions remain pending domain review.

## Regression evidence

The current-main no-Summation and one-child paths were characterized against
Issue #39 baseline `f583a5a12110132decf4b2989fdd79e8e2920c4b` (recorded in
the regression tests). The full Summation → Clean → Overlap → Normalisation
fixture was run with the original worker at Epic baseline
`37a952a926f86244596bf9fdae73d3d147289205`, then compared against the extracted
pipeline. It uses two immediate runs per branch, four matching 512×512 frames,
two ToF segments, and valid Spectra/ShutterCount files. Selected persisted
image arrays match exactly by SHA-256, including summed, cleaned, overlap-
corrected, and all four final normalised frames. This is implementation
equivalence evidence, not scientific validation.

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
