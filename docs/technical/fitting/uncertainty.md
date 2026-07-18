---
title: Fitting uncertainty and planning estimator
doc_id: neat-tech-fitting-uncertainty
doc_type: technical_reference
functional_area: fitting
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/core/fitting.py, NEAT/ui/mixins/fitting.py]
source_symbols: [calculate_uncertainty_estimator_constant, estimate_uncertainty_parameter, FittingMixin.fit_region, FittingMixin.fit_full_pattern_core]
test_paths: [tests/test_core_bragg_edges.py, tests/test_fitting_headless.py]
---

# Fitting uncertainty and planning estimator

Individual fits use the square root of diagonal entries from SciPy
`curve_fit` covariance. Pattern fitting estimates:

```text
variance = RSS / max(1, N-M)
covariance = inverse(JᵀJ) × variance
standard error = sqrt(diagonal(covariance))
```

If pattern `JᵀJ` is singular, all parameter standard errors become infinity.
The calculation assumes locally linear, unweighted, independent residuals.
Fixed shape parameters are not estimated by the optimizer and therefore have
no fitted standard error. They are consistently reported as NaN in both
individual and pattern modes; NaN means “not estimated”, not zero uncertainty.

The separate uncertainty-planning estimator uses:

```text
K = macro_pixel_size × Uamp × fitting_uncertainty²
```

and algebraically solves for any one of those three positive finite values.
This empirical estimator is independent of the optimizer covariance and must
not be described as the same uncertainty calculation.

The planning equation has been confirmed for use by NEAT. Scientific review
must still define its units and provenance, as well as the confidence
interpretation and validity of the optimizer covariance assumptions.

## Retrieval questions

- How does NEAT estimate fit parameter uncertainty?
- What does infinite uncertainty mean in pattern fitting?
- Why do fixed parameters show NaN uncertainty?
- Is the macro-pixel uncertainty estimator the optimizer covariance?
