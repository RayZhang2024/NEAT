---
title: Fit bounds, fixed parameters and diagnostics
doc_id: neat-tech-fitting-bounds-diagnostics
doc_type: technical_reference
functional_area: fitting
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin.fix_s_enabled, FittingMixin.fix_t_enabled, FittingMixin.fix_eta_enabled, FittingMixin._plot_residual_line]
test_paths: [tests/test_fitting_headless.py]
---

# Fit bounds, fixed parameters and diagnostics

The checkbox in each `s`, `t`, `eta` column header controls whether that
parameter is fixed for all rows: checked means fixed. Fixed values come from
the Edge Table; unchecked parameters are included in optimization.

Diagnostics include observed and fitted curves, a residual panel showing
`data-fit`, Region 3 RMS, covariance-derived parameter uncertainties, fitted
edge height and derivative FWHM. Large wavelength gaps can split plotted
residual lines to avoid visually connecting disjoint edges.

The code does not calculate reduced chi-square from measurement uncertainties,
does not weight residuals by per-point variance, and does not perform an
automatic goodness-of-fit acceptance test. Optimization success only means the
numerical solver met its stopping condition.

Bounds and initial values differ between individual and pattern modes; see
their respective pages. Scientific review must define acceptable residuals,
uncertainty thresholds, boundary-hit handling and fit rejection criteria.

## Retrieval questions

- Does a checked s/t/eta header mean fixed or fitted?
- What residual does NEAT display?
- Is fitting weighted by counting uncertainty?
- How should I tell whether a converged fit is trustworthy?

