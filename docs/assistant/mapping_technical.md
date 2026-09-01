# NEAT Assistant — Reviewed Mapping and Result Details

Knowledge-base version: NEAT 4.8.2
Approval: reviewed technical guidance for user support

## Mapping boxes and stored coordinates

Mapping ROI maxima are half-open: the maximum x and y bounds are excluded from
the array slice. Each wavelength value for a mapping box is the sum of all
pixels in that box.

The fit result is stored at:

```text
row = top + box_height // 2
column = left + box_width // 2
```

For an even-sized box this is the lower/right one of the two central pixel
positions. There is no universal recommended box size, step size or overlap;
choose them based on the successful representative fit and required spatial
resolution.

## Number of edges in individual mapping

Individual-edge fitting and mapping process every valid Edge Table row. There
is no five-edge limit. An invalid or incomplete row is skipped.

One edge may fail in a box while other edges continue. Failed results remain
`NaN`.

## Speeding up batch fitting

Increase `Step x` and/or `Step y` to fit fewer spatial positions and reduce
batch-fitting time. Locations skipped by the sampling grid may be filled when
interpolation is enabled.

A larger step reduces the density of directly fitted measurements and can miss
small spatial features. First establish a stable representative fit, then
choose the box and step sizes based on the required spatial detail rather than
speed alone.

## Mapping cancellation

Stop is cooperative between fitting operations. When cancellation is detected,
NEAT exits before writing mapping CSV files. Partial mapping results are not
saved, including when Stop is requested while the final box is being fitted.

## Mapping interpolation

When interpolation is enabled, NEAT interpolates in detector-pixel coordinates.
Directly fitted box centers are the source samples. Interpolated values are not
additional measurements and should not be interpreted as independently fitted
pixels.

Detailed policy for nearest-neighbour fallback, convex-hull extrapolation and
uncertainty interpolation remains under review. Inspect the sampling step and
interpolation setting in the CSV metadata.

## Individual-edge CSV columns

Individual-edge results use the complete three-index HKL suffix. For `(1,1,0)`,
both ungridded and gridded CSV files use:

- `d_110`, `s_110`, `t_110`, `eta_110`;
- `fwhm_110`, `height_110`;
- `d_unc_110`, `s_unc_110`, `t_unc_110`, `eta_unc_110`.

`d`, `s`, `t` and FWHM are in ångströms; `eta` is dimensionless; height has the
same signal units as the fitted spectrum. Each `_unc` column has the same unit
as its parameter.

## Pattern CSV columns

Pattern results first include the shared structure-dependent lattice parameters
and their `_unc` columns. Per-edge columns then contain `s`, `t`, `eta`, FWHM,
height and the corresponding shape-parameter uncertainties.

Every mapping CSV has `x` and `y` detector-pixel coordinates and one row for
every detector pixel. Pixels outside the mapping ROI or without a value are
normally `NaN`.

## Mapping output files

Successful individual and pattern mapping produce an ungridded CSV followed by
a gridded CSV. The completion message contains the actual absolute paths of
both files.

The metadata block records box and step sizes, ROI bounds, interpolation and
fixed-parameter settings, wavelength information, phase and serialized Edge
Table rows. Keep the metadata block with the data table when sharing results.
