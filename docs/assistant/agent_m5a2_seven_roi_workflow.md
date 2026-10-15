# M5A-2 autonomous seven-ROI fitting

## Feasibility and reuse

M5A-1 captures at most three representative, bounded image previews to choose
a reference ROI. Its measured spectrum is used to compare trusted phase overlays
and propose provisional phase/HKL choices. Only after that comparison does the
visual workflow choose six additional fitting locations. Uncertain or invalid
locations trigger up to two bounded replacement attempts; previously accepted
locations stay fixed, and a replacement must use different coordinates. If a
safe set of seven still cannot be found, fitting pauses. A local fitting worker
can then call the
read-only observation API for each ROI and reuse
`FittingEngine.fit_individual_edge()` without accessing Qt widgets or changing
the main fitting tab.

Fit configuration must preserve NEAT's default adjacent-edge midpoint rules,
the existing higher-wavelength Region 1 and lower-wavelength Region 2 bounds, and typed region order
`(Region 2, Region 1, Region 3)`. `s`, `t`, and `eta` remain free for each fit.
The existing worker inventory provides cooperative cancellation, revision
checks, and retirement tracking.

## User journey and layout

One click starts dataset preflight, bounded M5A-1 visual inspection, seven-ROI
validation, seven spectrum extractions, and every selected edge fit. The
planning, phase-comparison, location-retry, and per-cell fitting stages are
bounded LangGraph workflows that preserve cancellation and dataset-revision
checks. No routine
confirmation is inserted between stages. The default screen shows four short
conclusions; evidence, the matrix and parameters are disclosed on request.
Start/Run again, Stop and progress remain outside the content pages.

```text
NEAT / AI analysis
Analysis complete
Here is what each step found.

1 Dataset check       Dataset is ready for exploration
2 Phase comparison    Provisional match: Fe_bcc
3 Sample locations    7 sample locations selected
4 Local fitting       21 fits completed · 1 needs review

                        [View all results] [Review 1 flagged fit]
Exploratory analysis · physical interpretation needs separate validation.
Model/transfer disclosure     [Agent settings] [Data sent]
                                                      [Run again]
```

Clicking a step expands a short evidence explanation. Inspect evidence opens
only that step: dataset checks and warnings, the sample-location preview and
ROI controls, phase-comparison overlays, or the local fit matrix and curves.
The Step conclusions button returns to the summary. View all results and Show
fit details open the same fit-only matrix and plot view. Review flagged fits
first opens a concise reason and ROI/HKL identity. Multiple flagged fits can be
traversed without marking them accepted or changing fitting settings.

The sample-location preview uses original-pixel mapping and text ROI labels.
Selecting a fit-matrix cell changes its linked fit evidence. The matrix uses
fixed row heights so all seven rows fit; evidence from other steps is hidden in
the fit-only view. At 820×520 the summary scrolls as a single page and the
shared action controls stay visible.
Matrix cells show fit status; numerical values and the legacy `d_fit` convention
are shown in the selected result and its tooltip rather than as unlabelled cell
numbers. Column headers name the HKL and theoretical wavelength on separate lines.

Conclusions are derived from captured result objects, not guessed from progress
percentages or model prose. Limited data is labelled as limited, cancelled runs
as stopped, and blocked runs as needing attention. Withdrawn proposals clear
their result/review presentation. Dataset changes invalidate all four conclusions.

## Run states

- **Ready:** Show dataset/model readiness and one primary Start action. No
  proposed fit is implied.
- **Running:** Show the actual stage and worker-reported fit count. Keep Stop
  visible and prevent overlapping runs.
- **Completed:** Show four step conclusions. Offer results and flagged-fit
  review on request. Label phase provisional and fitted `d_fit` as a legacy
  numerical convention in the detailed view.
- **Completed with issues:** Preserve independent successes; failed or
  unavailable combinations remain individually marked with actionable detail.
- **Needs input / blocked:** Show actual proposed/rejected locations and a short
  reason; do not fit unless exactly seven valid distinct ROIs and supported
  edges pass deterministic checks.
- **Stopping:** Request cooperative cancellation, ignore later work, and retain
  worker ownership until thread exit is confirmed.
- **Stopped:** Show only completed results from that run and accurate cancelled
  or skipped counts.
- **Stale:** Clear current evidence immediately after dataset revision changes;
  require a fresh run.

## Privacy and scientific scope

The OpenAI disclosure permits at most six visual requests and sixteen bounded
PNG transfers, including repeated previews for location selection and retries.
Per-request byte limits and run authorization still apply. The measured
reference-spectrum plot is sent for phase comparison; fitted curves, parameters,
and results are not sent to a model. Input images, fitting-tab controls, saved
configurations, and source files are not modified. This is seven-location
exploratory fitting; it does not create a map or validated phase, strain, stress,
or physical d-spacing result. The existing `d_fit` convention caveats remain
visible beside results.

LangSmith tracing is separately opt-in in Agent settings and off by default.
When enabled, graph stages are traced and each direct model request is added as
an LLM run under its graph node where applicable. LLM runs include the request
and response; visual calls include the bounded PNG previews sent for review.
The traces go to the selected LangSmith agent and environment. The API key is stored in the OS keyring. The
agent is created by LangSmith on its first trace. Select the region that matches
the API key's workspace; a workspace ID is optional for keys linked to multiple
workspaces. The settings connection test sends one synthetic trace with a fixed
payload and records its separate model request as an LLM run when tracing is
enabled. When tracing is enabled, graph state, model prompts and responses,
metadata, errors, and runtime information are uploaded without client-side
redaction. Trace data may include dataset-derived values, ROI coordinates,
spectra, or rendered previews. Turning tracing off explicitly suppresses inherited LangSmith
environment settings.
