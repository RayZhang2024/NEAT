---
title: Parameter maps, masks and FITS export
doc_id: neat-tech-postprocessing-maps-export
doc_type: technical_reference
functional_area: postprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/postprocessing.py, NEAT/ui/dialogs.py]
source_symbols: [PostProcessingMixin.plot_parameter, ParameterPlotDialog.plot_parameter, ParameterPlotDialog.apply_filter, ParameterPlotDialog.save_as_fits, ParameterPlotDialog.resize_parameter_map]
test_paths: [tests/test_postprocessing_helpers.py]
---

# Parameter maps, masks and FITS export

Maps use `pcolormesh` with cell edges inferred halfway between coordinate
centers. Aspect ratio is equal and the y-axis is inverted so row zero appears
at the top. Default color limits are finite-map mean ± two standard deviations.

**Filter** loads a FITS mask through NEAT’s orientation-aware reader. Shape
must exactly match `Z`. Filtering multiplies map and mask, then changes every
zero result to NaN. Therefore a genuine parameter value of zero is also lost,
and nonbinary masks scale values rather than merely keep/discard them. NEAT
warns when a mask contains values other than binary 0 and 1 before applying it.

**Save image** preserves the parameter map’s original two-dimensional grid,
casts a copy to `float32`, and writes FITS with `PARAM`, COMMENT and HISTORY
header entries. It does not resize or interpolate the imported CSV result.
Original grid coordinates, physical scale and complete source metadata are
not written.

Direct image-result loading accepts FITS/TIFF; arrays with more than two
dimensions use only `arr[0]`.

Focused tests verify unchanged export shape/data and binary-mask detection.

## Retrieval questions

- How are parameter-map cell edges and color limits calculated?
- Does a mask only hide pixels or can it scale values?
- Why did valid zero values disappear after filtering?
- What size and metadata does exported FITS contain?
