---
title: Multi-edge pattern fitting
doc_id: neat-tech-fitting-pattern
doc_type: technical_reference
functional_area: fitting
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [known phases]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin.fit_full_pattern_core, FittingMixin.fit_full_pattern]
test_paths: [tests/test_fitting_headless.py]
---

# Multi-edge pattern fitting

Pattern fitting requires a supported known structure and its required lattice
parameters. Valid Edge Table rows with Region 3 data are collected. Each edge’s
two baselines is fitted first; then all Region 3 arrays are concatenated.

`least_squares` jointly refines shared lattice parameters and four baseline
parameters per edge. Each unfixed shape parameter is still independent per
edge. Lattice values are bounded ±5%. Baseline bounds use
`value ± max(abs(value),1)`. Unfixed pattern bounds are:

- `s`: 0.0005–0.01, initial 0.01
- `t`: 0.01–0.1, initial 0.02
- `eta`: 0–1, initial 0.5

Default `max_nfev` is 300. The residual returned to the solver is
`observed-model`. A failed solver returns no result. Successful fitting can
update the application’s lattice parameters.

Pattern and individual modes do not use identical `s` lower bounds or initial
`t`; those differences still require policy review. In both modes, fixed shape
parameters have NaN uncertainty because their uncertainty is not estimated by
the optimizer.

## Retrieval questions

- Which parameters are shared across edges in pattern fitting?
- How does pattern fitting differ from individual-edge fitting?
- What happens to the selected phase lattice after a successful fit?
- Why do fixed-parameter uncertainties appear as NaN?
