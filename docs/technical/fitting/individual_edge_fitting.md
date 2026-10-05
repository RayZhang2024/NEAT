---
title: Individual-edge fitting
doc_id: neat-tech-fitting-individual-edge
doc_type: technical_reference
functional_area: fitting
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/domain/individual_edge.py, NEAT/services/individual_edge.py, NEAT/services/fitting_engine.py, NEAT/ui/mixins/fitting.py]
source_symbols: [IndividualEdgeFitConfig, IndividualEdgeFitAttempt, IndividualEdgeFitResult, FittingEngine.fit_individual_edge, FittingMixin.fit_region, FittingMixin.fit_all_regions]
test_paths: [tests/test_individual_edge_service.py, tests/test_individual_edge_adapter.py, tests/test_fitting_headless.py]
---

# Individual-edge fitting

`FittingMixin.fit_region()` parses table or explicit row data and presents the
fit; the Qt-free `FittingEngine.fit_individual_edge()` computes the staged
fit from arrays and `IndividualEdgeFitConfig`. Its typed attempt retains
completed Region 1/2 curves even if Region 3 fails, so the UI can keep those
plots. The mixin converts success to the established, asymmetric legacy
dictionary shapes. `skip_ui_updates=True` suppresses plots, messages and
table updates. Region 3 refines four baseline parameters, one lattice-like
edge parameter and any unfixed `s`, `t`, `eta`, using bounded `curve_fit`
with `maxfev=300`.

The legacy row order is Region 2, Region 1, Region 3. Region 1 uses bounded
`curve_fit` and Region 2 unbounded `curve_fit`, both with SciPy's default
iteration limit. Known-phase rows use the current lattice `a` as the Region-3
initial value; unknown-phase rows use `d / 2` and the model's empty-HKL path.
The Region-3 model remains cubic-like even when a non-cubic phase is selected;
no new rejection or scientific support is implied. Its lattice-like initial
value is bounded ±5%; each baseline estimate uses a symmetric interval of
`max(abs(value), 1)`.

Unfixed bounds are:

- `s`: 0.0001 to 0.01, initial 0.01
- `t`: 0.01 to 0.1, initial 0.1
- `eta`: 0 to 1, initial 0.5

Fit covariance supplies standard errors. Fixed parameters receive NaN
uncertainty. The legacy `d_fit`, `fit_params[0]` and CSV `d_*` values retain
their existing factor-of-two and HKL-denominator meanings; names were not
reinterpreted. RMS is the square root of mean Region-3 residual squared.
Edge height is fitted curve max-minus-min over 4000 points; width is FWHM
of the numerical derivative over a clipped local 4000-point grid.

For known phases the height/width location still uses the supplied/current
phase structure and lattice state, not necessarily the newly fitted Region-3
parameter. Unknown phases use the fitted value. This known concern remains
uncorrected; domain review is required before interpreting output physically.

## Retrieval questions

- In what order are the three regions fitted?
- Which parameters are refined in individual-edge mode?
- What bounds are applied to s, t and eta?
- Does a converged fit guarantee a valid physical edge?

