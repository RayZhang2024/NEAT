---
title: Bragg-edge pseudo-Voigt/exponential-tail model
doc_id: neat-tech-fitting-edge-model
doc_type: technical_reference
functional_area: fitting
audience: [scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/core/fitting.py]
source_symbols: [fitting_function_1, fitting_function_2, fitting_function_3]
test_paths: [tests/test_core_bragg_edges.py, tests/test_fitting_headless.py]
---

# Bragg-edge pseudo-Voigt/exponential-tail model

Baseline models are:

```text
Region 1: exp(-(a0 + b0*x))
Region 2: exp(-(a0 + b0*x)) * exp(-(a_hkl + b_hkl*x))
```

Region 3 evaluates a Gaussian integrated step with an exponential tail
`g_term`, a Lorentzian integrated step with an exponential tail `l_term`, and:

```text
step = (1-eta)*g_term + eta*l_term
pre  = exp(-(a_hkl + b_hkl*x))
edge = exp(-(a0+b0*x)) * (pre + (1-pre)*step)
```

The edge location is `2*d_hkl` for a known phase. With no hkl list, it is
`2*a` from the supplied pseudo-lattice dictionary. Edges outside Region 3 are
ignored. Multiple hkl contributions are added, not multiplied.

The fitting parameters have the following domain interpretation:

- `s`: edge broadening associated with the sample microstructure
- `t`: edge broadening associated with the instrument
- `eta`: instrument neutron-pulse edge shape; numerically it mixes the
  Gaussian-like (`eta=0`) and Lorentzian-like (`eta=1`) terms

The present implementation treats `s` and `t` as wavelength-like quantities
in ångströms and `eta` as dimensionless.

Overflow/value errors for one contribution are silently skipped. No explicit
guard prevents zero/negative `s` or `t` inside the function; fitting bounds are
responsible for that. The physical definitions and known-phase edge position
have been reviewed; the normalization and additive multi-edge combination
still require scientific review.

Focused tests verify finite/distinct `eta=0` and `eta=1` model endpoints and
that an edge outside the fit window contributes zero.

## Retrieval questions

- What functions form the Region 1, 2 and 3 models?
- How does eta mix Gaussian and Lorentzian contributions?
- How is the modeled edge position obtained?
- What happens when a theoretical edge lies outside Region 3?
