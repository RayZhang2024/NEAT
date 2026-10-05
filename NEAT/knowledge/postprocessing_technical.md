# NEAT Assistant — Reviewed Post-Processing Details

Knowledge-base version: NEAT 4.8.2
Approval: reviewed technical guidance for user support

## Result CSV loading

A NEAT result CSV begins with:

```text
Metadata Name,Metadata Value
```

Metadata continues until a blank line, followed by the data table. Plotting
requires exact lowercase `x` and `y` coordinate columns. Every other column is
offered as a map metric.

If duplicate `(x,y)` rows exist, the final row encountered supplies the map
value. Sparse or unavailable coordinates remain `NaN`.

## Default map colour range

The initial map colour range is based on the finite-value mean plus or minus
two population standard deviations. Manual limits affect display contrast; they
do not change the underlying parameter data.

## Post-processing masks

The mask must have the same two-dimensional shape as the parameter map. A
binary mask uses 1 to keep and 0 to exclude.

If a selected mask contains other values, NEAT warns that those values will
scale the map rather than simply keep or exclude pixels. The current filtering
implementation also converts every zero result to `NaN`, so a genuine zero
parameter value is not preserved. Review a filtered map before exporting it.

## FITS parameter-map export

Save image exports the map with the same rows and columns as the imported CSV
result. It does not resize the data to 512×512 or interpolate it. Values are
written as `float32`.

Current headers include `PARAM`, COMMENT and HISTORY. Coordinate scale,
parameter units and complete source provenance are not yet stored in the FITS
header, so retain the source CSV and metadata.

## Strain calculation restrictions

Calculate Strain is enabled only for fitted d-spacing maps named `d_*`.
Uncertainty maps such as `d_unc_*`, other fit metrics and arbitrary images are
not valid strain sources.

With a positive reference spacing `d0` in the same unit and for the same
phase/reflection, NEAT calculates:

```text
microstrain = (d / d0 - 1) × 1,000,000
```

Positive microstrain means measured spacing is larger than `d0`; negative
microstrain means it is smaller. This is lattice strain, not stress. Stress
must not be inferred without appropriate elastic constants and measurement
geometry.

## ROI statistics

ROI minimum and maximum coordinate centers are both included when they match
map coordinates. NEAT ignores `NaN` values and reports the arithmetic mean and
population standard deviation (`ddof=0`).

The interface currently labels every non-Strain metric as ångströms, which is
not correct for dimensionless `eta`, signal height or every uncertainty type.
Use the metric-specific units documented with the result schema.

## Line profiles

A line profile uses 500 evenly spaced samples between the two selected points.
Values are linearly interpolated on the `(y,x)` map grid. Distance is measured
from the first point in the active display unit, pixels or millimetres.

The current exported line file contains only `Distance` and `Value`; it does
not include the metric name, source filename or units. Record those details
separately when exporting a profile.

## Millimetre display limitation

The mm display currently applies a hard-coded `0.055 mm/pixel` to both axes.
It is not read from result metadata and does not adjust for detector mode,
binning or macro-pixel mapping.

Use millimetres only when 0.055 mm/pixel is valid for the dataset. Otherwise
use pixel coordinates or perform an externally validated coordinate conversion.
