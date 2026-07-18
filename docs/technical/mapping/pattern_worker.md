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
test_paths: []
---

# Pattern batch mapping worker

The UI requires an initial full-pattern fit before mapping. For each box, the
worker sums pixels at every wavelength and calls `fit_full_pattern_core` with
`max_nfev=300`, baseline `curve_fit` limit 300, and no update to global lattice
state.

On the first successful box, it allocates full detector arrays for fitted
lattice parameters and uncertainties plus per-edge `s`, `t`, `eta`, their
uncertainties, height and FWHM. Values are stored at box centers. Failed boxes
remain NaN; only the first five failure explanations are shown.

If no box succeeds, no files are saved. Otherwise an ungridded snapshot is
written, optional interpolation runs, then a gridded snapshot is written.
Stop is cooperative between boxes and is checked again before result writing,
so cancellation discards partial unsaved results even when requested during
the final fit.

Height/FWHM are recalculated on a 14,000-point local model grid using fitted
lattice/edge values. This expensive calculation occurs for every successful
box and edge.

## Retrieval questions

- Why must I run an initial pattern fit before mapping?
- Are mapped lattice parameters shared within each box?
- How are failed boxes represented?
- Does batch pattern fitting alter the application’s phase lattice?
