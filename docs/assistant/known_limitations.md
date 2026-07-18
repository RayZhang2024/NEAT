# NEAT Assistant — Known Limitations and Escalation Boundaries

Knowledge-base version: NEAT 4.8.0  
Approval: user-safe limitations derived from reviewed technical documentation

## A successful fit is not scientific validation

Numerical convergence only means the optimiser met its stopping condition.
NEAT does not automatically establish the correct phase, reference spacing,
window choice, physical model or stress interpretation.

Escalate dataset-specific scientific conclusions to an experienced
Bragg-edge scientist. The assistant may explain controls and diagnostics but
must not guarantee that a result is scientifically valid.

## No fitting settings can guarantee a correct strain map

There are no universal fitting settings that guarantee a strain map is
scientifically correct. Correctness depends on wavelength calibration,
preprocessing, phase/reflection choice, fitting windows, model suitability,
fit diagnostics and the physical validity of `d0`.

The assistant may provide the documented checks for those controls, but the
final strain-map interpretation requires dataset-specific review by an
experienced scientist.

## Unresolved interpolation policy

Mapping interpolation fills locations that were not independently fitted.
Exact policy for fallback interpolation, extrapolation outside the convex hull
and interpolation of uncertainties remains under review.

The assistant should explain that interpolated pixels are estimates and refer
the user to the mapping metadata. It should not claim that every output pixel
is a direct measurement.

## Coordinate scale is not dataset-aware

The post-processing mm switch assumes 0.055 mm/pixel on both axes. This value
is not read from the instrument, detector, binning or CSV metadata.

If that scale has not been confirmed for the dataset, use pixel coordinates
and ask the instrument scientist for the correct physical calibration.

## Export metadata is incomplete

Parameter-map FITS export preserves the map grid but does not yet include full
coordinate calibration, units, source CSV identity or complete analysis
provenance. Line-profile export similarly omits metric, units and source.

Keep the original CSV and its metadata with exported images and profiles.

## Post-processing zero-value masking

Post-processing filtering multiplies the map by the mask and converts every
zero result to `NaN`. This can remove a genuine parameter value of zero.

The assistant should warn about this behaviour rather than describe the
filtered output as lossless.

## Direct multi-plane image results

When a directly loaded FITS or TIFF result has more than two dimensions,
post-processing currently displays only the first plane. Confirm the intended
plane or export it explicitly as a two-dimensional image before analysis.

## Questions requiring human review

Mandatory human review is required for:

- guaranteed fitting settings or scientific correctness;
- phase identification from an uncertain dataset;
- selection of a physically valid `d0`;
- conversion of strain to stress;
- unusual instrument geometry or wavelength calibration;
- confidential or facility-sensitive experiment information;
- repeatable crashes, corrupted files or unsupported formats.
