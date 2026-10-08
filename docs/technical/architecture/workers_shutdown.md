---
title: Worker ownership, progress, cancellation and shutdown
doc_id: neat-tech-architecture-workers-shutdown
doc_type: technical_reference
functional_area: architecture
audience: [developer, support]
neat_version: 4.8.0
verified_commit: c138fc5804912984b4665821f4c4bea55c9a5b28
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/ui/preprocessing_worker_registry.py, NEAT/ui/window_thread_inventory.py, NEAT/ui/main_window.py, NEAT/ui/mixins/preprocessing.py, NEAT/ui/mixins/fitting.py, NEAT/ui/assistant_panel.py, NEAT/ui/assistant_settings_dialog.py, NEAT/services/fitting_engine.py, NEAT/workers/preprocessing.py, NEAT/workers/batch.py]
source_symbols: [PreprocessingWorkerRegistry.start_worker, PreprocessingWorkerRegistry.request_shutdown, PreprocessingWorkerRegistry._poll_workers, PreprocessingWorkerRegistry._retire, PreprocessingMixin._begin_preprocessing_workflow, FitsViewer.closeEvent, FitsViewer.cleanup_resources, WindowThreadInventory.track, WindowThreadInventory.request_shutdown, AssistantDockWidget.shutdown, AssistantSettingsDialog.has_unsettled_workers, BatchFitWorker.stop, BatchFitEdgesWorker.stop]
test_paths: [tests/test_application_shutdown.py, tests/test_preprocessing_worker_ownership.py, tests/test_preprocessing_workers.py, tests/test_preprocessing_normalisation.py, tests/test_preprocessing_normalisation_raden.py, tests/test_preprocessing_full_process.py, tests/test_fitting_headless.py, tests/test_assistant_panel.py, tests/test_assistant_settings_dialog.py, tests/test_assistant_semantic_retrieval.py, tests/test_pattern_batch_worker.py, tests/test_batch_mapping_outputs.py]
---

# Worker ownership, progress, cancellation and shutdown

## Thread model

Long image loading, preprocessing, batch fitting, update checks and assistant
requests use `QThread` workers so the Qt GUI remains responsive. All
preprocessing-owned workers and transient image loaders are registered with
the window-owned `PreprocessingWorkerRegistry` before `start()`. The registry
holds strong references independently of the mixin's convenience attributes,
and is exposed as `FitsViewer._preprocessing_worker_registry` for the later
application-close integration.

The recurring signal pattern is:

- `progress_updated(int)` for a percentage;
- `message(str)` for the relevant message box;
- a payload signal such as `run_loaded`, `stack_loaded` or `answer_ready`; and
- `finished(...)` for UI reset and reference cleanup. The seven preprocessing
  workers expose a no-argument `finished` signal.

The preprocessing registry routes progress, messages, payloads and public
completion through QObject receivers using queued connections. Registry state,
payload consumption, GUI callbacks, family-generation changes and retirement
therefore run on the GUI thread. Batch fitting also emits `current_box_changed`
for the moving map overlay.

## Cooperative cancellation

NEAT does not forcibly terminate normal workers. Stop methods set flags that
the calculation checks at safe points:

| Worker family | Stop flag |
|---|---|
| Preprocessing workers | `_is_running = False` |
| Batch fitting workers | `stop_requested = True` |
| `ImageLoadWorker` | `stop()` sets `_stop_requested = True` |
| FITS/NeXus/RADEN loaders | Their respective cooperative stop flag |
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

## Preprocessing GUI ownership and Stop

The ownership state progresses from registered/starting through acknowledged
start, optional Stop request, confirmed exit and retirement. A start is
acknowledged by the native `QThread.started` signal or observed running state;
a fast post-start `isFinished()` state also covers a thread that exits before
the GUI observes it running. The registry never treats `wait(0)` on a
never-started thread as completion: it uses that nonblocking check only after
start/exit evidence and combines it with `isFinished()` and not-running state.
The 500 ms startup check is diagnostic, not grounds for retirement: if Qt has
not acknowledged a start and the thread is not verifiably rejected, the
registry reports that startup remains unacknowledged and keeps the worker
strongly owned while its active-only poll continues. A delayed start can then
be acknowledged normally. A custom wrapper may raise
`PreprocessingWorkerStartRejected` to report a synchronous rejection; even
then, retirement requires an unchanged pre-start state and a nonblocking
`wait(0)`. An arbitrary exception may have occurred after Qt accepted the
start, so it is treated as ambiguous. If startup remains indefinitely
ambiguous, ownership and the family lock are intentionally retained rather
than risking destruction of a thread that may later start. The user receives
a diagnostic; same-family Run remains unavailable until Qt supplies safe
start/exit evidence. This is an exceptional residual limitation.

Public adapter `finished` is deliberately separate from native QThread exit.
Issue #41 workers manually emit a public signal from `run()`, and that signal
can arrive while the native thread is still running. A single active-only
25 ms GUI-thread timer polls registered threads; retirement requires verified
exit through `isFinished()`, not-running state and successful `wait(0)`. The
worker remains strongly referenced until its queued payload/completion handoff
is processed and actual exit is confirmed. No GUI Stop or completion path
uses a positive-timeout wait, `quit()` or forceful termination.

After completion handling consumes the worker's structured result, or after a
cancelled/abnormal worker's actual exit is confirmed, the registry runs a
GUI-side cleanup callback. It clears direct mixin convenience attributes only
when they still refer to that exact worker, including transient image-loader
attributes and numbered lazy Summation loaders. A stale worker therefore
cannot clear a newer worker bound to the same attribute. Cancelled workers do
not need their public completion handler to run for references to be released.
Timer callbacks and signal-proxy references are detached on retirement so they
do not keep the worker/result graph alive. Run-button state is restored from
the state recorded when that generation began, and Normalisation remains
disabled while a related Open Beam loader is still active.

Payload and public completion signals enter one queued QObject receiver.
Payloads are held for the receiver's completion handoff, then delivered before
the batch completion callback. This avoids relying on separate receivers
having a particular queue order. Worker identity and family-generation checks
guard payload, progress, message and completion callbacks; cancelled payloads
and callbacks from retired generations are discarded. A stopped batch does
not start its next loader, service worker or dataset, while natural FAILED
service outcomes keep their existing continuation policy.

When actual exit is confirmed without public completion, the registry allows a
zero-delay GUI event dispatch and then a nonblocking 250 ms grace timer for
queued signals. At expiry it emits one neutral abnormal-completion diagnostic,
discards buffered payloads, prevents success reporting and batch continuation,
and retires the worker. Late signals are ignored. Run controls become
retryable only after every worker in that family has drained.

Family locks are generation-based rather than global. One-, two- and three-
level Summation share `summation`; classic and RADEN Normalisation share
`normalisation`; Clean, Overlap, Filtering and Full Process have separate
families. A new same-family generation, including a Summation/Normalisation
mode switch, is refused until retirement. Unrelated families may proceed
independently. Superseded Open Beam loader selections use selection identities:
each remains registry-owned until exit, but only the latest selection may
update open-beam data or UI state. `OpenBeamLoadWorker` has no cooperative
Stop method, so it is retained and its obsolete payload is ignored.

Full Process GUI Stop still delegates to its existing parent/active-operation
token semantics from Issue #39. Its pipeline's in-progress loader remains
non-cancellable; no nested workers were introduced.

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

Run buttons are disabled for an active family generation and are not restored
by a public `finished` signal alone. A user Stop cancels the entire generation
and disables further Run/Stop clicks until its loaders and service worker have
actually exited. Message text is part of the operational audit trail but is
not written to a persistent application log.

## Application shutdown

Window close is a two-phase operation. On the first close request,
`FitsViewer.closeEvent` snapshots unsettled work, marks close pending, blocks
new preprocessing/fitting/update starts, and requests cooperative cancellation
from the preprocessing registry and the window's legacy-thread inventory. It
then rejects that close event without clearing data, saving settings or
destroying widgets. A GUI-thread poll keeps ownership until every worker has
verified native exit. Public completion alone is not sufficient evidence;
native exit is checked with `isFinished()` and nonblocking `wait(0)` after
startup/exit evidence. A worker whose completion notification is missing gets
a bounded queued-signal grace, but remains owned until the safe retirement
path completes.

The preprocessing registry is the source of truth for generation lifetime,
including active, starting, stop-requested, closing and ambiguous starts.
Shutdown cancels every existing generation, including one already closing or
one with no currently registered workers. New same-window preprocessing work
is blocked only while the close attempt is pending. Once all work drains, a
later close request may proceed; if a close is rejected for another reason,
the pending guard is reset so normal work can resume.

`WindowThreadInventory` covers the window-owned FITS, NeXus and RADEN image
loaders, both batch-fitting workers and the update-check thread. It records
workers before `start()`, independent of replaceable convenience attributes,
and suppresses stale queued UI callbacks after cancellation or supersession.
Each guarded signal callback is counted at emission and released after its
GUI-thread handler has run (or been suppressed). Retirement requires verified
native exit and an empty callback count, so late or reordered queued handlers
do not depend on a fixed settling delay. An update check cannot be cooperatively
interrupted, so it remains tracked until actual thread exit. Worker references
and their signal proxies are not cleared or deleted while native execution may
continue. Exceptions from stop requests are diagnostic only and never treated
as proof of exit. There is no forceful `terminate()` path.

After quiescence is verified, the second close request preserves the
Assistant's established shutdown contract: request interruption and wait up
to 1500 ms. If the assistant thread remains active, close is rejected and
application data/settings are left untouched. Child assistant settings
dialogs also veto their own close while provider-test or model-discovery
threads remain unsettled. Accepted cleanup is guarded against duplicate
execution; it does not perform a second worker-stop/clear pass.

## Observed implementation limitations

- A non-cooperative legacy loader or update-check thread can keep the first
  close attempt pending until its native call returns. The user can continue
  using the window after a rejected close; close is retried only by a later
  request, not accepted automatically in the background.
- A startup whose state remains genuinely ambiguous is retained and blocks
  close rather than being guessed dead. This favors safety over automatic
  recovery; a diagnostic identifies the outstanding work.
- `OpenBeamLoadWorker` has no `stop()` method and does not check Qt interruption
  requests in its load loop; it is retained until exit and obsolete selected
  payloads are ignored.
- The common preprocessing completion signal indicates that an adapter run
  ended; use its structured result and `succeeded` state to distinguish
  success, cancellation and failure.
- Wait timeouts are not followed by a second warning or persistent diagnostic.

## Retrieval questions

- Does pressing Stop immediately terminate a NEAT calculation?
- How does worker progress reach the GUI?
- Which operations run outside the GUI thread?
- What happens to active processing when NEAT closes?
- Why does NEAT ask the user to close again after workers drain?
- Why might closing wait for an AI request or file operation?

