---
title: Post-processing coordinates and display units
doc_id: neat-tech-postprocessing-coordinates
doc_type: technical_reference
functional_area: postprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [fixed 0.055 mm-per-pixel display assumption]
scientific_review: pending
source_paths: [NEAT/ui/dialogs.py]
source_symbols: [ParameterPlotDialog.calculate_edges, ParameterPlotDialog.toggle_units, ParameterPlotDialog.plot_parameter, ParameterPlotDialog.update_coordinates]
test_paths: [tests/test_postprocessing_helpers.py]
---

# Post-processing coordinates and display units

CSV `x/y` values are treated as pixel-center coordinates. Display cell edges
are midpoints between adjacent centers, with the first/last interval
extrapolated symmetrically. This helper requires at least two centers.

The **mm** switch applies a hard-coded factor to both axes:

```text
millimetres = pixel coordinate × 0.055
```

Mouse coordinates and ROI inputs are divided by the same factor before data
lookup. The nearest coordinate center is used for the cursor value.

The factor is not read from metadata, instrument, detector mode or binning.
It also assumes identical X/Y pitch. Scientific/instrument review must decide
when 0.055 mm/pixel is valid and whether macro-pixel or detector geometry
changes the physical scale.

A focused test verifies midpoint cell edges and extrapolated end edges.

## Retrieval questions

- How does NEAT convert post-processing pixels to millimetres?
- Is 0.055 mm/pixel read from the CSV?
- How are plotted cell edges derived from coordinates?
- Which map value is shown for the mouse position?
