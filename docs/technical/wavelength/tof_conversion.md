---
title: Time-of-flight to wavelength conversion
doc_id: neat-tech-wavelength-tof-conversion
doc_type: technical_reference
functional_area: wavelength
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/workers/batch.py, NEAT/ui/mixins/fitting.py]
source_symbols: [_axis_to_wavelengths, _tof_us_to_wavelengths, FittingMixin.update_wavelengths, FittingMixin._recalculate_wavelengths_from_tof_axis]
test_paths: [tests/test_fitting_headless.py]
---

# Time-of-flight to wavelength conversion

NEAT implements the neutron conversion with constant `3.956`, but has two unit
entry paths.

For NeXus/RADEN axis centers stored internally in microseconds:

```text
wavelength [Å] = ToF [µs] × 3.956 / flight_path [m] / 1000
```

For classic `_Spectra.txt` and manual ToF anchors interpreted as milliseconds:

```text
wavelength [Å] = (ToF [ms] + delay [ms]) × 3.956 / flight_path [m] × 1000
```

For stack axes, delay is converted from milliseconds to microseconds before
addition. A positive finite flight path is required by low-level stack
conversion. The classic update path does not explicitly validate positivity
before division.

NeXus axes identified as wavelength are used directly and do not depend on
flight path. Ambiguous NeXus units use a magnitude heuristic documented in the
NeXus page.

Scientific review must confirm the constant, units, delay sign and whether all
supported export formats follow these assumptions.

## Retrieval questions

- What equation converts ToF to wavelength?
- Why do classic spectra and RADEN axes use different scale factors?
- When is a NeXus axis used directly as wavelength?
- What happens with a zero flight path?

