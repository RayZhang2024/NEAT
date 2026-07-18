---
title: Edge Table and fitting windows
doc_id: neat-tech-fitting-edge-table
doc_type: technical_reference
functional_area: fitting
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin._default_bragg_edge_windows, FittingMixin._sync_derived_region3_bounds, FittingMixin._active_region_bounds]
test_paths: [tests/test_fitting_headless.py]
---

# Edge Table and fitting windows

Each row contains hkl, theoretical `d`, two visible baseline windows, hidden
derived Region 3 bounds, and initial/fixed `s`, `t`, `eta`.

For theoretical edge `x`, defaults are:

```text
lower min = max(0.90x, x-0.3)
lower max = 0.98x
upper min = 1.04x
upper max = min(1.12x, x+0.4)
```

Outer limits are clamped by midpoints to adjacent theoretical edges. A window
is valid only when each min is less than its max. Hidden Region 3 always spans
visible column “1 Min” through “2 Max”; independent Region 3 editing is no
longer exposed.

Implementation naming is confusing: regional plotting and fitting sometimes
map the visible first/second baseline columns in reversed variable order.
Documents describe the executed column mapping, not assumed pre/post-edge
semantics. Domain review must confirm which side represents Region 1 and 2.

## Retrieval questions

- How are default Edge Table windows calculated?
- How are nearby theoretical edges used to clamp windows?
- Where do Region 3 limits come from?
- Why is an edge row invalid?

