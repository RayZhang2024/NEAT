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
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin.fit_region, FittingMixin.fit_all_regions]
test_paths: [tests/test_fitting_headless.py]
---

# Individual-edge fitting

The staged fit first estimates two exponential baselines with `curve_fit`.
Region 3 then refines the four baseline parameters, one lattice-like edge
parameter, and any unfixed `s`, `t`, `eta`, using bounded `curve_fit` with
`maxfev=300`.

Known-phase rows parse hkl and use the selected structure. Unknown-phase rows
use a synthetic edge label and the model’s empty-hkl path. Lattice-like
initial value is bounded ±5%; each baseline estimate uses a symmetric interval
of `max(abs(value),1)`.

Unfixed bounds are:

- `s`: 0.0001 to 0.01, initial 0.01
- `t`: 0.01 to 0.1, initial 0.1
- `eta`: 0 to 1, initial 0.5

Fit covariance supplies standard errors. Fixed parameters receive NaN
uncertainty in the individual result. RMS is the square root of mean Region 3
residual squared. Edge height is fitted curve max-minus-min over 4000 points;
width is FWHM of the numerical derivative over a local 4000-point grid.

Known concerns include mixed lattice/edge variable naming and later height/
width calculation using phase state that may not consistently reflect the
newly fitted value. Domain review and targeted numerical tests are required
before interpreting output physically.

## Retrieval questions

- In what order are the three regions fitted?
- Which parameters are refined in individual-edge mode?
- What bounds are applied to s, t and eta?
- Does a converged fit guarantee a valid physical edge?

