---
title: Worker ownership, progress, cancellation and shutdown
doc_id: neat-tech-architecture-workers-shutdown
doc_type: technical_reference
functional_area: architecture
audience: [developer, support]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/ui/main_window.py, NEAT/ui/mixins/preprocessing.py, NEAT/ui/mixins/fitting.py, NEAT/ui/assistant_panel.py, NEAT/workers/preprocessing.py, NEAT/workers/batch.py]
source_symbols: [FitsViewer.cleanup_resources, AssistantDockWidget.shutdown, SummationWorker.stop, FullProcessWorker.stop, BatchFitEdgesWorker.stop, BatchFitWorker.stop]
test_paths: [tests/test_preprocessing_workers.py, tests/test_fitting_headless.py, tests/test_assistant_panel.py, tests/test_assistant_semantic_retrieval.py, tests/test_pattern_batch_worker.py]
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
- `finished(...)` for UI reset and reference cleanup.

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
already exited. `BatchFitEdgesWorker` still uses its existing GUI-owned fitting
path; that worker is not decoupled here.

The Full Process worker runs child preprocessing workers inside nested
`QEventLoop` instances. Its own stop flag is checked between stages, but a
currently executing child operation must reach its own safe point before the
parent can continue or exit.

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
- Completion signals do not consistently distinguish success, cancellation and
  failure. Some workers emit `finished` after logging an error.
- Wait timeouts are not followed by a second warning or persistent diagnostic.
- There is no consolidated end-to-end test covering close during every worker
  type.

## Retrieval questions

- Does pressing Stop immediately terminate a NEAT calculation?
- How does worker progress reach the GUI?
- Which operations run outside the GUI thread?
- What happens to active processing when NEAT closes?
- Why might closing wait for an AI request or file operation?

