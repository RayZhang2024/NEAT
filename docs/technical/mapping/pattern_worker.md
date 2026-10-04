---
title: Pattern batch mapping worker
doc_id: neat-tech-mapping-pattern-worker
doc_type: technical_reference
functional_area: mapping
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [known-phase image stacks]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py, NEAT/workers/batch.py]
source_symbols: [FittingMixin.batch_fit, BatchFitWorker]
test_paths: [tests/test_pattern_batch_worker.py, tests/test_batch_mapping_outputs.py]
---

# Pattern batch mapping worker

The UI requires an initial full-pattern fit before mapping. For each box, the
worker sums pixels at every wavelength and calls its explicitly injected
`FittingEngine.fit_full_pattern` instance. The worker has no MainWindow parent
or GUI callback dependency. Its nested plain-Python fitting configuration is
deep-copied at construction, while wavelength and image arrays remain shared
inputs (the worker does not copy the full image stack). Calls use
`max_nfev=300`, `curve_fit_maxfev=300`, and do not update global lattice state.

On the first successful box, it allocates full detector arrays for fitted
lattice parameters and uncertainties plus per-edge `s`, `t`, `eta`, their
uncertainties, height and FWHM. Values are stored at box centers. Failed boxes
remain NaN; only the first five failure explanations are shown.

If no box succeeds, no files are saved. Otherwise an ungridded snapshot is
written, optional interpolation runs, then a gridded snapshot is written.
Stop is cooperative between boxes and is checked again before result writing,
so cancellation discards partial unsaved results even when requested during
the final fit.

Height/FWHM continue to be recalculated by the worker on a 14,000-point local
model grid using fitted lattice/edge values; the engine's own height/width
outputs are not substituted. This calculation occurs for every successful box
and edge. This decoupling applies only to full-pattern `BatchFitWorker`;
`BatchFitEdgesWorker` retains its GUI dependency pending Issue #7.

## Retrieval questions

- Why must I run an initial pattern fit before mapping?
- Are mapped lattice parameters shared within each box?
- How are failed boxes represented?
- Does batch pattern fitting alter the application’s phase lattice?
