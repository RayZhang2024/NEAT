# NEAT Assistant Parameter Reference

Knowledge-base version: NEAT 4.8.2
Approved sources: `User manual.md` and documented NEAT GUI behaviour

## Preprocessing parameters

### Base name

Filename stem used for generated outputs. It must be non-empty and should not contain path separators.

Relevant modules: Summation, Clean, Overlap Correction, Normalisation, Filtering and Full Process.

### Window half (`n`)

Spatial half-width used during normalisation. The full moving window is `(2n+1) × (2n+1)` pixels.

- `n=0`: no spatial neighbourhood averaging.
- Supported range: `0` to `100`.
- Larger `n`: smoother results and improved statistical stability, with reduced spatial detail.

Source: NEAT User Manual §3.4.

### Adjacent (`m`)

Half-width of the temporal or frame neighbourhood used during normalisation. The full window contains `(2m+1)` frames.

- `m=0`: only the current frame.
- `m=1`: current frame plus one neighbouring frame on each side.
- Supported range: `0` to `10`; the reviewed default is `m=0`.
- Larger `m`: more frame-to-frame smoothing, with greater risk of smoothing wavelength-dependent structure.

Source: NEAT User Manual §3.4.

### Binary mask

A FITS image used to keep or exclude spatial pixels. Its dimensions must exactly match the data images or post-processing map to which it is applied.

Source: NEAT User Manual §§3.5 and 5.2.

## Wavelength and experiment parameters

### Flight path

Source-to-detector flight path used when converting time of flight to wavelength. NEAT can process data from instruments other than IMAT, so this must be the correct value for the instrument and experiment. The documented `56.4 m` value is only the default commonly used at IMAT; it must not be assumed for another instrument. Users should obtain the experiment-specific value from the responsible scientist when uncertain.

Source: NEAT User Manual §4.1.

### Time delay

Timing offset used with flight path when manual anchors are specified in time-of-flight mode. An incorrect value shifts the calculated wavelength axis.

Source: NEAT User Manual §4.1, “Manual anchors in Config”.

### Manual anchor: Image number

One-based image index associated with a known wavelength or time-of-flight value. **Unused** rows are ignored. Indices beyond the loaded image count are clamped to the last image.

### Manual anchor: Wavelength or ToF value

Known value assigned to an anchor image. Wavelength values are used directly. Time-of-flight values are converted using flight path and time delay. NEAT linearly interpolates between anchors.

Source: NEAT User Manual §4.1, “Manual anchors in Config”.

### Global minimum and maximum wavelength

Defines the overall wavelength interval shown and considered when listing/fitting Bragg edges. The interval must contain the target edge or edges.

Source: NEAT User Manual §§4.2 and 4.3.

### Phase

Material/crystal-phase selection used to calculate theoretical Bragg-edge positions and populate the Edge Table. Selecting a phase is optional when the material is not known, but an incorrect phase can produce irrelevant theoretical edges.

Source: NEAT User Manual §§4.1 and 4.2.

## Spectrum and fitting parameters

### Initial macro-pixel coordinates

The minimum and maximum x/y coordinates used to extract an averaged test spectrum. A larger region generally improves counting statistics but reduces spatial localisation and can mix regions with different physical behaviour.

Source: NEAT User Manual §4.2.

### Window 1 (`1 Min`, `1 Max`)

Lower-wavelength, pre-edge baseline region used in the staged fit.

### Window 2 (`2 Min`, `2 Max`)

Higher-wavelength, post-edge baseline region used in the staged fit.

### Full Bragg-edge fitting interval

The full interval fitted by the Bragg-edge model runs from visible `1 Min` to visible `2 Max`, including the transition between the two baseline windows. In NEAT v4.8.1, the former `3 Min` and `3 Max` inputs are hidden and are derived automatically from `1 Min` and `2 Max`; users do not configure a separate third window.

Source: NEAT User Manual §4.3 and Appendix A.2.

### `s` (`sigma`)

Wavelength-like edge-broadening parameter associated with sample
microstructure. Treat it as a fitted model parameter; physical interpretation
requires awareness of the instrument response and fitting model.

Source: NEAT User Manual §4.3 and Appendix A.1.

### `t` (`tau`)

Wavelength-like edge-broadening parameter associated with the instrument.
Physical interpretation should consider the instrument model and parameter
correlations.

Source: NEAT User Manual §4.3 and Appendix A.1.

### `eta`

Dimensionless instrument neutron-pulse edge-shape parameter. Numerically it is
the Lorentzian fraction in the pseudo-Voigt combination:

`B_PV = (1-eta) B_G + eta B_L`

The documented range is `0 ≤ eta ≤ 1`. `eta=0` selects the Gaussian component and `eta=1` selects the Lorentzian component.

Source: NEAT User Manual Appendix A.1.

### Fixed versus refined parameters

A fixed parameter remains at its supplied value during the relevant optimisation. A refined parameter is allowed to vary within its configured bounds. Refining more correlated parameters may make weak or noisy fits less stable; fixing a parameter also risks bias if the supplied value is inappropriate. Test the decision on representative spectra.

Source: NEAT User Manual §§4.3 and 4.4.

### Fit edges

Runs the individual three-stage fitting procedure for selected edges.

### Fit pattern

Runs a multi-edge fit using a shared lattice parameter `a` for the selected pattern.

Source: NEAT User Manual §§2.2 and 4.4.

## Mapping parameters

### Box width and box height

Spatial dimensions of the mapping macro-pixel. Keep them consistent with the macro-pixel dimensions used for the successful representative test fit.

### Step x and step y

Spatial step between directly fitted mapping positions. A larger step reduces the number of direct fits and speeds processing; NEAT can interpolate skipped positions.

### Mapping ROI

Minimum and maximum x/y bounds defining the region processed during batch fitting. The ROI must lie within the image and be large enough for the selected mapping box.

Source: NEAT User Manual §§4.5 and 4.6.

## Post-processing parameters

### Result CSV parameter columns

Data Post-Processing turns every CSV data column except spatial coordinates
`x` and `y` into a selectable map button. Individual-edge results commonly use
`d_...` for fitted lattice spacing. Pattern results contain the refined
structure-dependent lattice parameter names, such as `a` or `c`. Edge-model
columns can include `s_...`, `t_...`, `eta_...`, `fwhm_...` and `height_...`.
Columns containing `_unc` are the estimated uncertainty for the corresponding
fitted parameter. The suffix identifies the associated Bragg edge.

### Colour Bar Min and Max

Manual display limits for the map. Minimum must be smaller than maximum. When initially blank, NEAT can populate them from the map data.

### Display in mm

Changes axes and coordinate inputs from pixels to millimetres using `0.055 mm/pixel`. In NEAT v4.8.1 this conversion is fixed in the software and is not editable in the interface.

### `d0`

Positive reference spacing used to calculate strain. It must use the same units and represent the same quantity as the displayed spacing map.

### Strain

Calculated in microstrain using:

`((d-d0)/d0) × 1,000,000`

### Post-processing ROI bounds

Rectangle used for mean-value calculation. Coordinates follow the currently selected display units.

Source: NEAT User Manual §5.2.
