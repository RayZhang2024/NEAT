---
title: Manual wavelength and ToF anchors
doc_id: neat-tech-wavelength-manual-anchors
doc_type: technical_reference
functional_area: wavelength
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [classic image folders without spectra]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin.set_manual_wavelength_mode, FittingMixin.update_manual_wavelengths, FittingMixin.open_manual_spectra_settings_dialog]
test_paths: [tests/test_fitting_headless.py]
---

# Manual wavelength and ToF anchors

Manual mode is enabled when no spectra file supplies an axis. Anchors use
one-based image numbers; image 1 displays suffix `_00000`. Values can be direct
wavelengths or ToF in milliseconds. ToF anchors use current flight path and
delay.

Invalid entries and nonpositive indices are ignored. Indices beyond the number
of images are clamped to the last image. Repeated indices overwrite earlier
entries. With no anchors, no wavelengths are produced. One anchor assigns a
constant wavelength to all frames.

With at least two anchors, wavelengths are linearly interpolated by image
index. If the first/last images are outside the supplied anchor range, the
nearest anchor value is extended constantly to the corresponding stack edge.
No monotonicity or positive-wavelength check is enforced.

The dialog starts with at least ten anchor rows and permits displayed values
between -10 and 100. Scientific review should define required calibration
points, valid ranges and whether endpoint extrapolation is acceptable.

Focused tests verify wavelength interpolation with constant endpoint extension
and the millisecond ToF conversion including delay.

## Retrieval questions

- Are manual anchor image numbers zero- or one-based?
- What happens outside the first and last anchor?
- Can I use ToF rather than wavelength anchors?
- Why did one anchor give every frame the same wavelength?
