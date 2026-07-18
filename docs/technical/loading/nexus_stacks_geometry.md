---
title: NeXus image stacks and geometry
doc_id: neat-tech-loading-nexus
doc_type: technical_reference
functional_area: loading
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [Mantid-style NeXus workspaces]
scientific_review: pending
source_paths: [NEAT/workers/batch.py, NEAT/ui/mixins/fitting.py]
source_symbols: [get_nexus_image_stack_info, load_nexus_image_stack, NexusImageStackLoadWorker, FittingMixin._choose_nexus_flight_path]
test_paths: [tests/test_fitting_headless.py]
---

# NeXus image stacks and geometry

## Dataset discovery

NEAT searches HDF5 for numeric two-dimensional datasets. Signal attributes,
common dataset names and size contribute to candidate selection. The selected
signal is interpreted as `(detector_spectra, bins)`. The number of detector
spectra must be a perfect square; only square grids are supported.

An `axis1` dataset must exist beside the signal. If it contains `n_bins+1`
values, adjacent edge values are averaged to bin centers. If it contains
`n_bins`, values are already centers. Other axes with more than three values
are also converted from edges heuristically.

## Geometry and flight path

Source-to-sample distance is searched in Mantid parameter-map or instrument XML
paths. Sample-to-detector distance is taken from physical-detector distances or
the mean norm of detector positions. When both exist:

```text
file flight path = source-sample + sample-detector
```

The UI asks whether to use file geometry or the current app setting. If file
geometry is unavailable, the app setting is used.

## Frames and axes

Signal data are converted to `float32`; NaN and infinity become zero. Every bin
column is reshaped to the square detector grid. Unlike classic FITS/TIFF
loading, this reshape is not vertically flipped.

Axis units containing ToF/microsecond terms are treated as ToF; wavelength or
angstrom terms are treated as wavelength. Ambiguous axes are treated as ToF
when their maximum center exceeds 1000. ToF axes depend on flight path;
wavelength axes do not.

## Limitations

- Dataset and unit selection contain heuristics and can choose incorrectly in
  unusual NeXus layouts.
- Only square detector grids are accepted.
- Mean detector distance may hide geometric variation.
- The difference in orientation handling versus FITS/TIFF needs instrument
  verification.

## Retrieval questions

- How does NEAT find the image signal and axis in NeXus?
- Why does NEAT reject a non-square detector workspace?
- How is the NeXus flight path obtained?
- When does changing flight path recalculate NeXus wavelengths?

