# Scientific observation API and future agent boundary

## Purpose

The observation API exposes a bounded, read-only view of the scientific data
already loaded in NEAT. It is independent of the Qt interface at its core and
does not fit edges, change fitting settings, run mapping, save output, or
contact a model provider. A future assistant can use this layer as its only
scientific data source; it should not read GUI widgets, the filesystem, or
application internals directly.

## Component boundary

- `NEAT.domain.observation` defines typed dataset, axis, ROI, image, spectrum,
  plot, capability, provenance, and revision contracts. It imports no GUI code.
- `NEAT.services.scientific_observation` computes metadata, image previews,
  ROI spectra, and rendered plots. It has no Qt dependency and does not call
  `update_plots()` or any fitting method.
- `NEAT.ui.scientific_observation_adapter` is the only live-window bridge. It
  captures state on the owning Qt thread, refuses capture while image loading
  is active, and returns handles that can be read by the service layer.
- `DatasetRevisionClock` tracks the current dataset identity and revision.
  Dataset replacement changes the identity; scientific axis edits advance the
  revision. An operation checks the token before and after reading so a result
  cannot silently combine two revisions.

The existing GUI remains the owner of image arrays. Captured image frames are
read-only NumPy views, so capture does not duplicate the stack. Application
code must invalidate before replacing data or wavelength axes and must not
mutate image pixels in place while a read is active. Existing image loading and
normalization paths replace image arrays rather than editing captured pixels.
Imported wavelength and intensity vectors are copied into the handle and made
read-only.

## Scientific conventions

- Image row zero is the top row, matching NEAT's `imshow` display orientation.
  Pixel coordinates use zero-based integer pixel centres. ROI bounds are
  half-open: `x_min <= x < x_max`, `y_min <= y < y_max`.
- Preview resizing uses deterministic nearest-sample selection. The coordinate
  mapping reports how each preview pixel maps to the original pixel grid.
  Previews are at most 1024 pixels on either side and at most 1,048,576 pixels
  total. The PNG output is capped at 4 MiB.
- Image ROI intensity preserves the current GUI calculation exactly: sum each
  frame's ROI with NumPy's existing `sum` behavior, then divide the resulting
  per-frame sums by the ROI pixel area. The full wavelength and intensity
  vectors are returned. The display wavelength window is a separate mask and
  never discards source samples.
- Imported spectrum-only datasets provide a spectrum and plot without an ROI.
  Image previews and mapping are reported as not applicable. Axis order is
  preserved; invalid or non-monotonic axes are flagged rather than silently
  sorted by the observation layer.
- Plot output uses Matplotlib's non-interactive Agg canvas. Rendering is
  serialized because Matplotlib has process-wide configuration state.

## Privacy and resource limits

Observation metadata reports only a recognized input format, scientific axis
provenance, dimensions, units, safe instrument values, capabilities, and data
quality flags. It never includes local file paths, folder names, image pixels
outside an explicitly requested bounded preview, or a complete stack in a
model-ready summary. No output is persisted by default. The core performs no
network access or provider upload.

Spectrum vectors and image frame counts are capped at 250,000. Plot requests
are capped at 1600 by 1200 pixels (1,920,000 pixels), 200 DPI, and 8 MiB PNG.
An ROI request is capped at 250,000,000 frame-pixel operations per call.
Metadata flags are allowlisted and capped at 64 entries.
ROI extraction allocates output proportional to the number of frames and
reduces each requested ROI from the existing frame view; it does not copy the
full image stack. Preview allocation is bounded by the preview pixel cap.
Resource-limit failures are returned as typed validation errors instead of
silently truncating scientific observations.

## Revision and concurrency rules

Live-window capture and revision invalidation run on the Qt owner thread.
Image previews, ROI extraction, and plot rendering consume a captured handle
and are independent of Qt. Capturing while a loader runs is rejected. Replacing
an image/profile dataset or changing its wavelength axis invalidates prior
handles. Display-only contrast, selected-frame, and wavelength-window changes
do not invalidate the scientific data revision. Consumers should capture once
for a coherent series of observations and treat a stale-handle error as a
request to recapture and retry.

## Future assistant architecture

The intended call path is:

```text
User request
    -> assistant planner (future work)
    -> policy/controller (authorization, timeouts, budgets)
    -> typed observation API
    -> local NEAT dataset snapshot
```

Only the typed observation API is implemented here. Future model, planner,
workflow graph, popup, action execution, and mapping integration are out of
scope. A future controller should enforce policy outside model prompts:

- Data stay local by default. Any external image or spectrum transmission must
  require a clear user action and show what data will be sent.
- Read-only inspection can be available after the user starts the assistant;
  data-changing, fitting, mapping, and file-output actions need separate
  authorization and should be introduced in later issues.
- Enforce per-request time and resource budgets, cancellation, and a small
  bounded number of retries. Silence is not approval.
- Keep tool output structured, include dataset identity/revision, and reject
  stale results before presenting them as current.
- Keep the assistant out of raw GUI state and filesystem paths. Use explicit
  user-selected image previews where visual context is needed.

## Out of scope

This issue does not add an LLM, LangChain/LangGraph, an autonomous workflow,
assistant popup, fitting or mapping controls, provider configuration, upload,
or persistence. It also does not change legacy fitting, CSV, map, or d-fit
calculations and output conventions.
