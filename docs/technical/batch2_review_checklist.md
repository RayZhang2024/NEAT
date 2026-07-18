# Batch 2 domain-review checklist

Add corrections beneath each item. Batch 2 remains outside production RAG
until the applicable scientific and instrument items are resolved.

## Image loading and orientation

- [ ] Confirm classic FITS/TIFF vertical flipping matches detector coordinates.
  Not sure, but the latest one is the correct one.
- [ ] Confirm whether NeXus reshaped images require the same flip.
  Not sure
- [ ] Confirm that filename suffix order is spectral order for classic exports.
  yes
- [ ] Identify source FITS headers that must be preserved.
  Not sure

## NeXus

- [ ] Confirm supported Mantid/NeXus signal and axis layouts.
  Not sure
- [ ] Confirm source-sample and sample-detector geometry paths.
  Not sure
- [ ] Confirm whether mean detector distance is an acceptable flight path.
  Not sure
- [ ] Confirm the ambiguous-axis rule `maximum > 1000 => ToF`.
  Not sure

## RADEN

- [ ] Confirm largest-TIFF selection when a folder contains multiple stacks.
  Not sure
- [ ] Confirm `.stat` and JSON axis meanings and JSON millisecond assumption.
  Not sure
- [ ] Confirm uniform bin-center reconstruction from min/max/bins.
  Not sure
- [ ] Confirm TIFF vertical orientation and pulse metadata meanings.
  Not sure

## Wavelength calibration

- [ ] Confirm `3.956` and both implemented unit equations.
  Yes
- [ ] Confirm delay is added, not subtracted, and is expressed in milliseconds.
  yes
- [ ] Define valid flight-path and delay ranges.
  no need ranges limit
- [ ] Confirm whether wavelength-valued NeXus axes should ignore delay.
  Not sure
- [ ] Confirm manual-anchor interpolation and constant endpoint extension.
  Yes
- [ ] Decide whether one anchor assigning a constant wavelength is acceptable.
  Not sure

## Profile import

- [ ] Confirm first two columns should always mean wavelength and intensity.
  Yes
- [ ] Define expected wavelength and intensity units.
  Wavelength: Angstrom, intensity unit: counts.
- [ ] Decide how duplicate wavelengths should be handled.
  Not sure
- [ ] Confirm imported profiles may be fitted without additional normalization.
  Yes

## Phase library

- [x] Check every built-in structure, lattice value and hkl list.
  Reviewed. Cu and beta titanium were corrected on 2026-07-16.
- [x] Correct `Cu_fcc` to FCC copper with `a = 3.615 Å`.
  Completed.
- [x] Correct `Ti_Beta` to BCC beta titanium with `a = 3.32 Å`.
  Completed. The value is a reference value and can vary with temperature and
  composition.
- [x] Confirm reflection lists should remain explicit rather than generated
  from selection rules.
  Yes
- [ ] Define whether custom phases need provenance, temperature or version
  metadata.
  Not sure
