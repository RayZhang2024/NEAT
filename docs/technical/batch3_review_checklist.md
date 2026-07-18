# Batch 3 domain-review checklist

## Spectrum and windows

- [ ] Confirm ROI should be the arithmetic mean of all pixels.
  Not sure
- [ ] Define handling of masked, zero and nonfinite pixels in ROI means.
  Not sure
- [x] Confirm visible Edge Table baseline columns correspond to the intended
  pre-edge and post-edge regions.
  Yes
- [ ] Confirm default percentages, absolute caps and midpoint clamping.
  Not sure
- [ ] Define minimum points required in each region.
  Not sure

## Model

- [x] Verify the Gaussian and Lorentzian exponential-tail equations.
  Yes
- [x] Define physical meaning and units of `s`, `t` and `eta`.
  's' is edge broadening associated with sample (microstructure), 't' is edge broadening associated with instrument, 'eta' is edge shape associated with instrument (neutron pulse)
- [ ] Confirm multiple modeled edge contributions should be added.
  Not sure
- [x] Confirm known-phase edge position is `2*d_hkl`.
  Yes
- [ ] Define the scientifically meaningful unknown-phase parameterization.
  Not sure

## Individual-edge fitting

- [x] Confirm staged Region 1 → Region 2 → Region 3 procedure.
  Yes
- [x] Verify all initial values and bounds.
  Yes
- [x] Review the lattice/d-spacing conversions and variable naming.
  Yes
- [ ] Check whether height/width should use the fitted lattice value rather than
  stored phase state.
  Not sure
- [ ] Define minimum fit-quality acceptance criteria.
  Not sure

## Pattern fitting

- [x] Confirm lattice parameters are shared while shape parameters remain
  per-edge.
  Yes
- [x] Verify ±5% lattice bounds.
  Yes
- [ ] Explain or harmonize bounds/initial values that differ from individual
  fitting.
  Not sure
- [ ] Decide whether a successful pattern fit should update the phase lattice
  in application state.
  Not sure

## Diagnostics and uncertainty

- [ ] Confirm residual sign `data-fit`.
  Not sure
- [ ] Decide whether fitting should be weighted by counting/statistical errors.
  Not sure
- [ ] Define interpretation of RMS, edge height and derivative FWHM.
  Not sure, derive from the code
- [ ] Confirm covariance calculation and confidence meaning.
  Not sure, derive from the code
- [x] Harmonize fixed-parameter uncertainty: NaN versus zero.
  Completed: fixed parameters now report NaN in individual and pattern fits.
- [x] Confirm the empirical planning equation
  `K = macro_pixel_size × Uamp × uncertainty²`, including units and provenance.
  Yes
