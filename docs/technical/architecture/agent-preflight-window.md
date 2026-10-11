# AI Agent window: local Step 0 preflight

## Boundary and data flow

The independent, non-modal AI Agent window is opened from the application-wide
AI Agent menu. It is owned by one NEAT main window and does not replace the AI
Assistant dock. Opening the window performs no analysis. The user explicitly
starts a local, read-only preflight.

The Qt owner thread captures a versioned `ScientificDatasetHandle` and, when
valid, the already-selected fitting ROI. The ROI is read only: it is never
created, moved, or written to NEAT's fitting controls. A tracked `QThread`
uses Issue #53's observation service for metadata, at most three 512-pixel
image previews, and a measured ROI or imported-profile spectrum and plot.
The worker does not access QWidgets. `NEAT.domain.agent_preflight` classifies
only observed capabilities and quality conditions. The assessment contains
safe evidence references, not images or local file paths. Unknown companion
metadata and calibration sources remain explicit warnings.

The Agent window presents Overview and Visual Evidence together in two
resizable side-by-side panels. Overview shows the run and dataset tokens,
status, capabilities, provenance, warnings, and unsupported operations.
Visual Evidence shows actual preview PNGs, original-grid ROI overlays, and a
rendered measured spectrum where available. Frame wavelengths appear only for
a valid aligned axis. The full source spectrum count is separate from the plot
display window. The two panels share one companion window; the AI Assistant
dock remains in the main NEAT window.

## Worker and window lifetime

The main window registers the preflight thread in `WindowThreadInventory`
before starting it. Progress and completion callbacks are guarded by the
current worker attribute and run ID. The agent window checks the dataset ID
and revision before displaying evidence and monitors the revision while a
result is visible. Changed data discard prior evidence and request worker
cancellation. Stop is cooperative between bounded service calls; a currently
running preview or spectrum call finishes before cancellation completes.
Start remains disabled until the worker's native run has finished.

Closing the agent window requests cancellation and hides it for reuse. The
main window's existing close veto requests shutdown of a still-running
preflight and waits for inventory retirement before a later close can commit.
On committed main-window close, the agent window is hidden and clears its
result. The worker releases its captured handle before emitting completion;
the result retains only bounded previews and derived spectrum arrays. Future
consumers must similarly drop stale handles to release old frame stacks.

## Scope and future controller policy

This milestone has no LLM, LangGraph, LangSmith, automated fitting or mapping,
phase identification, model interpretation, report drafting, network upload,
or ten-second checkpoint. The window says “Local Read-only Preflight (no LLM
yet)” and never presents a heuristic as an AI conclusion.

A future controller should operate autonomously within granted permission,
use a ten-second pre-authorised checkpoint only for noncritical decisions,
limit refinement to three rounds per issue, leave cloud previews disabled by
default, and seek explicit approval for critical decisions. These are policy
requirements for later work, not behaviour implemented by this window.

## Manual smoke test

1. Launch NEAT from this branch. On each top-level workflow tab, open
   **AI Agent → Open AI Agent**. Confirm Overview and Visual Evidence are
   visible together in the resizable panels, the companion window moves and
   resizes independently, and the existing **AI Assistant** menu and dock
   still work. Close and reopen AI Agent; it should reuse one window.
2. With no data loaded, select **Start Preflight**. Overview should report
   invalid or no data, and Visual Evidence should contain no invented images
   or spectrum.
3. Load an image stack. Select a valid fitting ROI in NEAT, then start a new
   preflight. Overview should show real frame and wavelength metadata. Visual
   Evidence should show first/midpoint/last frame previews, the selected ROI
   overlay, and the measured spectrum and plot. Clear or reload images and
   confirm the old evidence changes to “Dataset changed — rerun preflight”.
4. Import a spectrum-only file and run preflight. The measured spectrum should
   appear, while image previews and mapping are marked not applicable.
5. Start a preflight and press **Stop**, or close AI Agent or NEAT while it is
   running. The UI should remain responsive; NEAT may ask for another close
   after the tracked worker has drained.
