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
source_paths: [NEAT/domain/fitting.py, NEAT/services/fitting_engine.py, NEAT/ui/mixins/fitting.py]
source_symbols: [FullPatternFitConfig, FullPatternFitResult, FullPatternEdgeFit, FittingEngine.fit_full_pattern, FittingMixin.fit_full_pattern_core, FittingMixin.fit_full_pattern]
test_paths: [tests/test_fitting_domain.py, tests/test_fitting_engine.py, tests/test_fitting_headless.py]
---

# Multi-edge pattern fitting

`FittingEngine.fit_full_pattern` performs the numerical pattern fit from
explicit wavelength/intensity arrays and `FullPatternFitConfig`. Its primary
result is `FullPatternFitResult`, including ordered typed `FullPatternEdgeFit`
entries. Fixed/free flags and solver limits remain explicit call arguments.
The UI-facing `FittingMixin.fit_full_pattern_core` remains a compatibility
adapter: it converts the legacy table/context dictionary to the typed config,
delegates to the engine, converts a successful result back to the existing
dictionary shape, and applies the lattice update when requested.

`_build_batch_fit_context()` still returns the plain shared table snapshot
used by both batch modes. Only the full-pattern path converts its scientific
fields to typed models; `BatchFitEdgesWorker` continues using the legacy
dictionary until Issue #7. Output provenance and mapping geometry are not
part of the scientific config.

Pattern fitting requires a supported known structure and its required lattice
parameters. Valid Edge Table rows with Region 3 data are collected. Each edge’s
two baselines is fitted first; then all Region 3 arrays are concatenated.

`least_squares` jointly refines shared lattice parameters and four baseline
parameters per edge. Each unfixed shape parameter is still independent per
edge. Lattice values are bounded ±5%. Baseline bounds use
`value ± max(abs(value),1)`. Unfixed pattern bounds are:

- `s`: 0.0001–0.01
- `t`: 0.01–0.1
- `eta`: 0–1

Initial shape values come from the valid Edge Table rows, subject to the
configured bounds when those values are free.

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
