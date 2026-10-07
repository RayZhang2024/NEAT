---
title: Worker ownership, progress, cancellation and shutdown
doc_id: neat-tech-architecture-workers-shutdown
doc_type: technical_reference
functional_area: architecture
audience: [developer, support]
neat_version: 4.8.0
verified_commit: eb605978e4cbc52630ff1138866d6786c2208b0d
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/ui/main_window.py, NEAT/ui/mixins/preprocessing.py, NEAT/ui/mixins/fitting.py, NEAT/ui/assistant_panel.py, NEAT/services/fitting_engine.py, NEAT/workers/preprocessing.py, NEAT/workers/batch.py]
source_symbols: [_finish_worker, FitsViewer.cleanup_resources, AssistantDockWidget.shutdown, SummationWorker.stop, FullProcessWorker.stop, BatchFitEdgesWorker.stop, BatchFitWorker.stop]
test_paths: [tests/test_preprocessing_workers.py, tests/test_preprocessing_normalisation.py, tests/test_preprocessing_normalisation_raden.py, tests/test_preprocessing_full_process.py, tests/test_fitting_headless.py, tests/test_assistant_panel.py, tests/test_assistant_semantic_retrieval.py, tests/test_pattern_batch_worker.py, tests/test_batch_mapping_outputs.py]
---

# Worker ownership, progress, cancellation and shutdown

## Thread model

Long image loading, preprocessing, batch fitting, update checks and assistant
requests use `QThread` workers so the Qt GUI remains responsive. The window or
owning mixin stores a worker reference, connects typed signals, then calls
`start()`.

The recurring signal pattern is:

- `progress_updated(int)` for a percentage;
- `message(str)` for the relevant message box;
- a payload signal such as `run_loaded`, `stack_loaded` or `answer_ready`; and
- `finished(...)` for UI reset and reference cleanup. The seven preprocessing
  workers expose a no-argument `finished` signal.

Qt queued connections move worker signals back to the GUI thread where needed.
Batch fitting also emits `current_box_changed` for the moving map overlay.

## Cooperative cancellation

NEAT does not forcibly terminate normal workers. Stop methods set flags that
the calculation checks at safe points:

| Worker family | Stop flag |
|---|---|
| Preprocessing workers | `_is_running = False` |
| Batch fitting workers | `stop_requested = True` |
| FITS/NeXus/RADEN loaders | `_stop_requested = True` |
| Qt fallback at shutdown | `requestInterruption()` |

Cancellation latency therefore depends on the current operation. A large file
read, image write, curve fit, OpenAI request or other blocking call may finish
before the flag is checked. `BatchFitWorker` uses an injected numerical engine
and checks cancellation after a final box fit before writing results, so a stop
requested during that fit discards unsaved partial results. A Stop
acknowledgement means that cancellation was requested, not that the worker has
already exited. `BatchFitEdgesWorker` now uses an injected fitting engine,
checks Stop between boxes and before saving, and does not add a per-edge
check. The current box may finish its remaining edge fits after Stop before
exiting without output files.

Full Process uses one `FullProcessWorker` QThread around the headless
`FullProcessPipeline`; it does not create child preprocessing workers or nested
`QEventLoop` instances. Its parent and active-operation cancellation state
remain separate, and active services retain their existing cooperative safe
points.

## Preprocessing worker lifecycle

The Clean, Summation, Overlap, classic Normalisation, Filtering, RADEN
Normalisation and Full Process adapters follow one completion contract:

```text
headless operation/pipeline result
        -> worker.result
        -> succeeded derived from result.status
        -> one public finished notification
```

This applies to direct `run()` calls and to `start()`/thread exit. On the
pinned PyQt5 baseline, plain `QThread.finished` delivers one callback after a
real thread exits. A subclass declaration of `finished = pyqtSignal()` shadows
that native signal: a subclass that declares but never emits it delivers no
callback, while one manual emit delivers one callback in either execution
mode. The preprocessing adapters retain the declared public signal and emit it
once after their run path; tests process queued Qt events after bounded
`wait()` before checking the public callback count. Filtering's no-runs and
no-mask paths now use this same one-emission final path; their FAILED results
and messages are unchanged.

Terminal adapter-only finalisation runs after a service or pipeline has
returned its structured result. A failure in terminal garbage collection or
standalone Normalisation's final memory report is best-effort: the original
result, its outputs/errors/warnings and its derived `succeeded` state remain
intact, a `[WARN] Worker finalization: <error>` message is attempted, and
completion is still attempted even if warning delivery fails. These warnings
are not added to scientific result errors.

This does not make callbacks that execute *inside* a service or pipeline
nonfatal. Classic Normalisation per-run GC and pacing, RADEN page-boundary GC,
Full Process stage-completion GC, and Full Process's in-pipeline final
Normalisation diagnostic retain their existing failure behavior. In
particular, an Issue #39 in-pipeline memory callback failure keeps the Full
Process result FAILED with its structural diagnostics and already-returned
operation outputs.

## GUI controls

Preprocessing has separate Run/Stop controls for Clean, Summation, overlap
correction, normalisation, Filtering and Full Process. Fitting has Stop controls
for both individual-edge and pattern mapping. Image-loading progress dialogs
can request cancellation.

Run buttons are normally disabled while their worker is active and restored by
completion handlers. Message text is part of the operational audit trail but
is not written to a persistent application log.

## Application shutdown

`FitsViewer.cleanup_resources` checks a fixed list of known worker attributes.
For each worker it:

1. calls `stop()` when present;
2. calls `requestInterruption()` if still running;
3. waits at most 1000 ms; and
4. clears the stored reference.

It then releases main image/spectrum arrays and disables image navigation.
There is no forceful `terminate()` call.

The assistant has a stricter close contract. It requests interruption and waits
1500 ms; if the thread remains active, the entire window close is rejected.
The assistant worker does not poll interruption while waiting for OpenAI, so
network timeout/retry settings can outlast that close wait.

## Observed implementation limitations

- `cleanup_resources` is an explicit name list. Several transient loader
  attributes used by preprocessing mixins are not listed, so comprehensive
  shutdown depends on their normal completion handlers.
- `OpenBeamLoadWorker` has no `stop()` method and does not check Qt interruption
  requests in its load loop.
- The common preprocessing completion signal indicates that an adapter run
  ended; use its structured result and `succeeded` state to distinguish
  success, cancellation and failure.
- Wait timeouts are not followed by a second warning or persistent diagnostic.
- GUI worker-reference ownership, close-during-processing behavior and
  consolidated application shutdown remain follow-up work; this worker
  lifecycle contract does not change the UI stop handlers or window cleanup.

## Retrieval questions

- Does pressing Stop immediately terminate a NEAT calculation?
- How does worker progress reach the GUI?
- Which operations run outside the GUI thread?
- What happens to active processing when NEAT closes?
- Why might closing wait for an AI request or file operation?

