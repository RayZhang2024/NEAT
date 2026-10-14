# M5A-2 autonomous seven-ROI fitting

## Feasibility and reuse

M5A-1 already captures at most three representative, bounded image previews in
its `sample_roi` request. This request can return the reference ROI and six
additional spatial ROIs without another upload or request. The same visual
workflow extracts the reference spectrum, compares trusted phase overlays, and
returns provisional phase/HKL choices. A local fitting worker can then call the
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
validation, seven spectrum extractions, and every selected edge fit. No routine
confirmation is inserted between stages. The top action/status strip remains
visible while the resizable horizontal overview/evidence splitter is used for
the results matrix and selected scientific evidence.

```text
NEAT AI Agent — seven-location fitting
[ Start Autonomous Analysis ] [ Stop ] [progress]    Fitting ROI C, edge (200) · 11/21
Dataset/model status · Provisional phase · 7 locations · 3 edges

OVERVIEW / RESULTS                         VISUAL EVIDENCE
Agent settings · Preview model data        One actual frame with Reference and A–F labels
Dataset summary · fit counts               Selected ROI/HKL fit curves and residuals
ROI × HKL status matrix (7 × 1–5)          Existing image, spectrum and phase evidence
Reference | queued | queued | ...
A         | done   | done   | ...
...                                        [Analysis details, collapsed by default]
```

The sample image is a single actual representative preview with original-pixel
ROI mapping. Each outline has a text label; colour alone does not encode ROI
identity. The compact matrix is
keyboard-selectable and selecting a cell updates the one detail view on demand.
The default selection is the first failed or needs-review cell, otherwise the
Reference ROI and first selected edge. Long covariance and error details stay
collapsed until requested.

At 1280×820 the horizontal splitter gives useful room to both matrix and
evidence. At 820×520 the top controls and status stay fixed; the overview can
scroll vertically, the evidence area uses a vertically resizable image/plot
splitter, and technical detail remains collapsed until requested. Critical
stage, Stop, and selected-cell status do not depend on colour alone.

## Run states

- **Ready:** Show dataset/model readiness and one primary Start action. No
  proposed fit is implied.
- **Running:** Show the actual stage and worker-reported fit count. Keep Stop
  visible and prevent overlapping runs.
- **Completed:** Show the seven-ROI matrix and selected fit evidence. Label phase
  provisional and fitted `d_fit` as a legacy numerical convention.
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

The existing M5A-1 preview request count, image count, byte limits, consent and
disclosure remain unchanged. No local spectra, fitted curves, parameters or
results are sent to a model. Input images, fitting-tab controls, saved
configurations, and source files are not modified. This is seven-location
exploratory fitting; it does not create a map or validated phase, strain, stress,
or physical d-spacing result. The existing `d_fit` convention caveats remain
visible beside results.
