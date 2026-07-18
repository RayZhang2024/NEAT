---
title: ROI statistics and line profiles
doc_id: neat-tech-postprocessing-roi-lines
doc_type: technical_reference
functional_area: postprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/dialogs.py]
source_symbols: [ParameterPlotDialog.calculate_mean, ParameterPlotDialog.extract_line_profile, ParameterPlotDialog.interpolate_z_values, LineProfileDialog]
test_paths: [tests/test_postprocessing_helpers.py]
---

# ROI statistics and line profiles

## Mean and standard deviation

Entered min/max coordinates must be strictly increasing. Millimetres are
converted with 0.055 mm/pixel, values are clamped to the map’s coordinate
range, and `searchsorted(left/right)` makes both requested bounds effectively
inclusive of matching coordinate centers. NEAT reports `nanmean` and population
`nanstd` (`ddof=0`).

The displayed unit is hard-coded as ångström for every non-Strain parameter,
even for `eta`, height or uncertainty columns where that may be incorrect.

## Line profiles

Two clicked points define 500 evenly spaced samples. `RegularGridInterpolator`
performs linear interpolation on `(Y_unique, X_unique)` with out-of-bounds
values set to NaN. Plot distance is Euclidean distance from the first point in
the currently displayed coordinate unit (pixels or mm).

Saved line data always use tab delimiters and header `Distance\tValue`,
including when the selected filename filter says CSV. Units and parameter name
are not written into the data file.

A focused test verifies that interpolation uses `(y,x)` grid order and produces
the expected bilinear center value.

## Retrieval questions

- Are mean-ROI bounds inclusive?
- Does the reported standard deviation ignore NaNs?
- How many samples are used in a line profile?
- What interpolation and distance units does the line profile use?
