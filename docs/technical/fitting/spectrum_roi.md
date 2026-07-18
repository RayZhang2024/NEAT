---
title: Spectrum extraction, ROI and macro-pixel averaging
doc_id: neat-tech-fitting-spectrum-roi
doc_type: technical_reference
functional_area: spectrum
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin.apply_selected_area, FittingMixin.update_plots]
test_paths: [tests/test_fitting_headless.py]
---

# Spectrum extraction, ROI and macro-pixel averaging

For image data, the selected ROI uses half-open array bounds
`image[ymin:ymax, xmin:xmax]`. NEAT sums that slice for each wavelength frame
and divides by `(xmax-xmin)*(ymax-ymin)`, producing the arithmetic mean pixel
intensity. The UI and messages sometimes call this “summed intensity,” but the
stored fitting spectrum is the mean.

Bounds must lie inside the first image. Wavelength and image counts must match.
The global wavelength range then filters points inclusively at both ends.
Changing the ROI recomputes the spectrum and clears earlier regional fit
parameters.

Imported profiles bypass spatial extraction and use their supplied intensity
array directly. Scientific review should define the intended ROI statistic,
partial-volume implications and whether masking/invalid pixels should affect
the denominator.

## Retrieval questions

- Does NEAT sum or average pixels in the yellow ROI?
- Are ROI maximum coordinates inclusive?
- Why does changing ROI clear the previous fit?
- How does imported-profile fitting differ from image ROI fitting?
