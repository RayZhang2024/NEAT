# NEAT Assistant — Reviewed Wavelength and Fitting Details

Knowledge-base version: NEAT 4.8.0  
Approval: reviewed technical guidance for user support

## Time-of-flight to wavelength conversion

For a time of flight in microseconds and flight path in metres, NEAT uses:

```text
wavelength (Å) = ToF (µs) × 3.956 / flight_path (m) / 1000
```

For manual time-of-flight anchors entered in milliseconds, the configured time
delay is added before conversion:

```text
wavelength (Å) =
    (ToF (ms) + delay (ms)) × 3.956 / flight_path (m) × 1000
```

The flight path must be positive and must describe the experiment. An incorrect
flight path or delay shifts the wavelength axis and therefore every fitted edge.

## Manual wavelength anchors

With two or more valid anchors, NEAT linearly interpolates wavelength against
image index. Before the first and after the final anchor, it uses the nearest
endpoint wavelength.

One anchor produces a constant wavelength for the whole stack and is generally
not a meaningful wavelength calibration. Use at least two anchors for a
wavelength-resolved image stack.

## Imported intensity profiles

An imported profile uses its first two numeric columns as wavelength and
intensity. Wavelength is interpreted in ångströms and intensity in counts or
the units of the imported signal. Rows are sorted by wavelength.

NEAT does not apply an additional normalisation to the imported profile.
Duplicate-wavelength policy remains under review, so remove unintended
duplicates before relying on a profile fit.

## Corrected built-in copper and beta-titanium phases

The reviewed built-in definitions are:

- `Cu_fcc`: face-centred cubic, `a = 3.615 Å`;
- `Ti_Beta`: body-centred cubic, reference `a = 3.32 Å`.

The beta-titanium value is a reference value. Its appropriate lattice parameter
depends on temperature and composition; create a custom phase when a different
reference is required.

NEAT uses explicit reflection lists. A phase label and successful numerical fit
do not prove that the selected phase is correct for the sample.

## Edge-model parameter meanings

- `s`: wavelength-like edge broadening associated with sample microstructure;
- `t`: wavelength-like edge broadening associated with the instrument;
- `eta`: dimensionless instrument neutron-pulse edge-shape parameter.

Numerically, `eta=0` selects the Gaussian-like term and `eta=1` selects the
Lorentzian-like term. Parameter correlations mean that fitted values still
require scientific interpretation.

## Individual-edge fitting sequence and bounds

Individual fitting proceeds in three stages:

1. fit the pre-edge baseline;
2. fit the post-edge baseline; and
3. fit the full edge interval using the baseline estimates.

The full interval runs from visible `1 Min` to visible `2 Max`. For a known
phase, the modeled edge position is `2 × d_hkl`.

Reviewed individual-fit bounds are:

- `s`: 0.0001–0.01;
- `t`: 0.01–0.1;
- `eta`: 0–1;
- lattice-like starting value: ±5%.

## Pattern fitting behaviour

Pattern fitting refines the selected edges jointly. Lattice parameters are
shared across the pattern, while each edge retains its own unfixed `s`, `t` and
`eta`. Lattice parameters are bounded to ±5% of their starting values.

Individual and pattern modes have some different shape-parameter starting
values and bounds. Do not assume that both modes solve an identical
optimisation problem.

## Fixed parameters and uncertainty

When `s`, `t` or `eta` is fixed, the optimiser does not estimate its uncertainty.
NEAT reports that uncertainty as `NaN`, meaning “not estimated.” It does not
mean zero uncertainty.

Unfixed parameter uncertainties come from the fitting covariance estimate.
They depend on local model assumptions and are not confidence guarantees.

## Fit diagnostics

The residual displayed by NEAT is:

```text
observed data - fitted model
```

RMS is the square root of the mean squared Region-3 residual. Edge height is
the fitted maximum-minus-minimum over the edge interval. FWHM is calculated
from the numerical derivative of the fitted edge.

Fitting is currently unweighted by per-point counting uncertainty, and NEAT
does not automatically reject every physically poor fit. Always inspect the
curve, residual, uncertainty, parameter bounds and physical plausibility.

## Uncertainty planning equation

The separate empirical planning estimator uses:

```text
K = macro_pixel_size × Uamp × fitting_uncertainty²
```

It can solve algebraically for any one of the three positive inputs. This
planning relation is separate from the optimiser covariance and must not be
described as the fitted-parameter uncertainty calculation.
