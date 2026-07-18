# Batch 4 domain-review checklist

## Mapping geometry

- [x] Confirm half-open ROI bounds and center-pixel storage convention.
  Code-derived: ROI maxima are excluded and results are stored at box centers.
- [x] Confirm even-sized boxes should map to `top + size//2`.
  Code-derived: this is the current storage convention.
- [x] Decide whether mapping spectra should be sums or means.
  Code-derived: mapping uses the sum of all pixels in each box.
- [x] Define recommended box/step sizes and overlap limits.
  There are no scientific recommendations or imposed overlap limits.

## Fit failures and cancellation

- [ ] Define when a failed box may be interpolated.
  Not sure
- [x] Decide whether fit-failure and unsampled-grid NaNs need separate masks.
  No
- [x] Decide whether Stop should save clearly marked partial results.
  No. Cancellation returns before the result-writing stage.
- [x] Confirm a maximum of five individual Edge Table rows is intended.
  No. All identified valid edges are now processed.

## Interpolation

- [x] Confirm linear interpolation in detector-pixel coordinates.
  Yes
- [ ] Decide whether nearest-neighbor fallback is acceptable.
  Not sure
- [ ] Define behavior outside the convex hull.
  Not sure
- [ ] Decide whether parameter uncertainties should be interpolated.
  Not sure
- [ ] Harmonize pattern interpolation safeguards with individual-edge mode.
  Not sure

## Results

- [x] Confirm exact meanings and units of every CSV column.
  Code-derived and documented in `mapping/result_csv_schema.md`.
- [x] Confirm full-detector rows are preferable to sparse sampled rows.
  Code-derived: both writers retain one row per full-detector pixel, using NaN
  where no value is available.
- [ ] Define provenance/status columns for measured, failed and interpolated
  pixels.
  Not sure
- [ ] Confirm HKL concatenation is unambiguous for all supported indices.
  Not sure
- [x] Resolve individual ungridded `d_11` versus gridded `d_110` naming for
  hkl `(1,1,0)`.
  Corrected: both ungridded and gridded outputs now use `d_110` and the full
  three-index suffix for every related column.
- [x] Decide whether completion signals should return actual output paths.
  Yes. Successful workers now emit the absolute ungridded and gridded paths.
