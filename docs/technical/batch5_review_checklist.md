# Batch 5 domain-review checklist

## CSV and parameter maps

- [x] Define a versioned result schema and required numeric columns.
  Code-derived: there is currently no schema version or numeric validation.
  Plotting requires exact lowercase `x` and `y`; all other columns are offered
  as parameters.
- [x] Decide how duplicate x/y rows should be handled.
  Code-derived: the last row encountered for a duplicate coordinate wins.
- [x] Confirm default colour range mean ± 2 standard deviations.
  yes
- [x] Decide whether direct image stacks should use only the first plane.
  Code-derived: direct FITS/TIFF result loading currently displays `arr[0]`
  when the loaded array has more than two dimensions.

## Masks and export

- [x] Decide whether masks must be binary.
  NEAT now warns before applying a non-binary mask.
- [x] Preserve genuine zero parameter values when filtering.
  Code-derived current behaviour: multiplication results equal to zero become
  NaN, so genuine zero parameter values are not preserved. This remains a
  known limitation.
- [x] Confirm whether every export should be resized to 512×512.
  No. FITS export now preserves the imported CSV result dimensions and values.
- [x] Define required FITS coordinate, unit and provenance headers.
  Code-derived current headers are `PARAM`, COMMENT and HISTORY only.
  Coordinate scale, parameter unit and complete provenance are not exported.

## Coordinates

- [ ] Identify instruments/configurations where 0.055 mm/pixel is valid.
  Instrument applicability remains unresolved.
- [x] Decide whether X and Y pixel pitch can differ.
  Code-derived: the current display assumes the same 0.055 mm pitch for both.
- [x] Determine scale changes after detector binning or macro-pixel mapping.
  Code-derived: the current display does not adjust the scale.
- [x] Store/read physical scale from metadata instead of hard-coding it.
  Code-derived: scale is not read from metadata; 0.055 mm/pixel is hard-coded.

## Strain

- [x] Restrict strain calculation to appropriate d-spacing columns.
  Yes. Only `d_*` values are eligible; `d_unc_*` and other metrics are blocked.
- [x] Define `d0` selection, provenance and units.
  Code-derived: the user enters any positive value. It must use the same
  length unit as the d-spacing map, but its value and provenance are not saved.
- [x] Confirm microstrain formula and sign convention.
  yes
- [ ] Add phase/hkl, temperature and composition guidance.
  not sure, derive from the code
- [x] Confirm stress must not be inferred without elastic constants and
  measurement geometry.
  yes

## ROI and line profiles

- [x] Confirm inclusive ROI-coordinate behavior.
  Code-derived: matching coordinate centers at both bounds are included.
- [x] Confirm population standard deviation (`ddof=0`).
  Code-derived: `numpy.nanstd` uses `ddof=0`.
- [x] Define correct units per metric instead of ångström for all non-Strain
  values.
  Code-derived current behaviour incorrectly labels every non-Strain metric
  as ångströms; metric-specific units remain to be implemented.
- [x] Confirm 500-point linear line interpolation.
  Code-derived: profiles use 500 evenly spaced samples and linear interpolation.
- [x] Add units, metric name and source information to exported line data.
  Code-derived current export contains only `Distance` and `Value`; these
  additions remain to be implemented.
