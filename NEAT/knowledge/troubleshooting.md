# NEAT Assistant Troubleshooting Guide

Knowledge-base version: NEAT 4.8.3
Approved source: `User manual.md`

Use the checks below in order. Do not claim that a numerical result is scientifically valid solely because the software completed or the optimiser converged.

## Summation reports that only one subfolder was found

**Likely cause:** The selected parent contains only one run, or the wrong folder level was selected.

**Actions:**

1. Confirm that the selected parent contains at least two run folders.
2. If the experiment has only one run, skip Summation.
3. Otherwise select the parent folder that directly contains all runs.

Source: NEAT User Manual §3.1.

## Summation rejects a mixed folder structure

**Likely cause:** Some immediate child folders contain images directly while others contain another level of run folders.

**Actions:** Reorganise the data so every child under the selected parent has the same depth. Use either the two-level run layout or the three-level sample/run layout, not both together.

Source: NEAT User Manual §3.1.

## Summation aborts because image counts or suffixes differ

**Likely cause:** At least one run has missing, additional or differently named frames.

**Actions:** Compare the FITS frame sets in every run. Each run must provide the same suffix keys and frame count before pixel-wise summation is valid.

Source: NEAT User Manual §3.1.

## A preprocessing stage cannot create its output folder

**Checks:**

1. Use **Set output** to choose an existing directory.
2. Confirm that the directory is writable.
3. Confirm that sufficient free disk space is available.
4. Provide a non-empty base name without path separators.

Source: NEAT User Manual §§3.1–3.6.

## Clean says that no dataset is selected or fails to load a dataset

**Actions:**

1. Select the dataset with **Add data**.
2. Confirm that the selected folder contains readable FITS images, or contains dataset subfolders in batch mode.
3. Review the message pane to identify the folder that was skipped.

Source: NEAT User Manual §3.2.

## Overlap Correction skips a dataset

Overlap Correction selects `.fits`, `.fit`, `.tiff` and `.tif` files whose final
underscore-separated suffix contains 1–10 ASCII digits. Frames are ordered by
the numeric suffix. Other supported FITS/TIFF images, including
`*_SummedImg.fits` summaries, are excluded from the correction stack and
reported in a warning; those files are left untouched. Duplicate numeric IDs (including
1 and 01), gaps in the sequence, or a mismatch between selected numeric frames
and Spectra rows are rejected before corrected images are written. The Spectra
rows are paired positionally with frames in numeric order; the filename does
not encode a ToF value. A folder with no numeric-suffix frames cannot be
processed.

**Likely causes:** a missing or malformed sidecar, ambiguous multiple sidecars,
a noncontiguous numeric frame sequence, or a mismatch between the selected
numeric image count and Spectra ToF row count.

**Actions:**

1. Confirm that exactly one `*_Spectra.txt` and one `*_ShutterCount.txt` file
   are present inside each dataset folder and that both belong to the images.
2. Confirm that frame suffixes are numeric, unique, contiguous, and have one
   corresponding Spectra row each.
3. Read the message pane for the stage, selected folder, counts, sidecar name,
   and any extra, missing, or invalid frame identifier.
4. In Full Process, unmanifested FITS/TIFF files in reused intermediate folders
   are ignored and reported; the current stage's output manifest defines the
   selected frames.

Source: NEAT User Manual §3.3.

## Normalisation cannot start

**Checks:**

1. Load the sample images with **Add Data**.
2. Load the open-beam images with **Add Open Beam**.
3. Select an existing, writable output directory.
4. Confirm that the sample and open-beam data correspond to compatible frame sequences.
5. Review the message pane for dataset-specific loading failures.

Source: NEAT User Manual §3.4.

## The normalised images are too noisy or too smooth

**If too noisy:** Consider increasing **Window half (`n`)** for more spatial averaging or **Adjacent (`m`)** for more averaging across neighbouring frames.

**If too smooth:** Reduce `n` to preserve spatial detail and reduce `m` to preserve wavelength-dependent detail.

Both parameters trade resolution against statistical stability. Recheck the resulting transmission spectrum rather than selecting values from appearance alone.

Source: NEAT User Manual §3.4.

## Filtering fails after selecting a mask

**Checks:**

1. Confirm that a sample dataset is loaded.
2. Confirm that the mask is a readable FITS image.
3. Confirm that the mask is binary and has exactly the same rows and columns as every data frame.
4. Confirm that an existing output folder and a non-empty base name are set.

Source: NEAT User Manual §3.5.

## Full Process aborts before running

**Checks:**

1. Select an existing, writable output folder.
2. Select a sample folder when prompted.
3. Select an open-beam folder when prompted.
4. Ensure overlap-correction metadata files are available when that stage is required.

Full Process currently performs its normalisation with `Adjacent = 0`.

Source: NEAT User Manual §3.6.

## NEAT cannot find a spectra file or has no wavelength axis

Use **Manual Spectra Setting**:

1. Select wavelength or time-of-flight anchor mode.
2. Enter at least two valid image indices and their corresponding values.
3. If using time of flight, verify the flight path and time delay.
4. Ensure anchor indices fall within the loaded image range.

NEAT linearly interpolates the wavelength axis between anchors. One anchor assigns a single wavelength to all images and is therefore generally not a meaningful wavelength curve.

Source: NEAT User Manual §4.1, “Manual anchors in Config”.

## No theoretical Bragg edges appear in the Edge Table

**Checks:**

1. Select the appropriate material phase, if it is known.
2. Confirm that the global wavelength interval contains the expected theoretical edge.
3. If the phase is unknown, use the supported unknown/custom-phase workflow rather than assuming a material.
4. Confirm that the wavelength axis is valid.

Source: NEAT User Manual §§2.2, 4.1 and 4.2.

## The extracted spectrum is weak or noisy

**Actions:**

1. Choose a stable region inside the sample and away from strong gradients or boundaries.
2. Increase the coarse macro-pixel area to improve counting statistics.
3. If background pixels are being included, consider applying a valid binary mask first.
4. Confirm that normalisation produced a plausible transmission spectrum.

Increasing the macro-pixel size reduces spatial resolution and may mix different material regions.

Source: NEAT User Manual §§3.5, 4.2 and 4.5.

## A test Bragg-edge fit fails, is unstable or appears biased

Check in this order:

1. Confirm that the target edge is clearly visible and lies inside the global wavelength range.
2. Confirm that Window 1 (`1 Min`, `1 Max`) covers a suitable lower-wavelength, pre-edge baseline region.
3. Confirm that Window 2 (`2 Min`, `2 Max`) covers a suitable higher-wavelength, post-edge baseline region.
4. Confirm that the complete interval from `1 Min` to `2 Max` contains the Bragg-edge transition. In v4.8.1, this full fitting interval is derived automatically; there are no separate visible `3 Min` and `3 Max` inputs.
5. Review the fitted curve, residuals, convergence messages, parameter values and uncertainties.
6. Adjust the window bounds or the bounds/fixed state of `s`, `t` and `eta`, then repeat the macro-pixel test fit.
7. If the spectrum remains inadequate, improve preprocessing or counting statistics before full-field fitting.

Do not proceed to mapping simply because the optimiser returned a result.

Source: NEAT User Manual §§4.2–4.4 and Appendix A.2.

## Batch fitting produces many failures

**Checks:**

1. Verify that the same macro-pixel size worked in the representative test fit.
2. Confirm that the mapping ROI lies within the loaded image and is large enough for the selected box.
3. Inspect whether failures cluster at sample boundaries, masked regions or low-transmission areas.
4. Review the message pane for retries, bounds hits and other diagnostics.
5. Reassess fitting windows and parameter bounds using representative spectra from the failing region.

Source: NEAT User Manual §§4.4–4.6.

## A Data Post-Processing mask is rejected

The FITS mask must have exactly the same 2D shape as the displayed map. Select
a matching mask or regenerate it using the map dimensions. If a mask has the
correct shape but is not binary, NEAT warns that its values will scale the map
rather than simply keep or exclude pixels.

Source: NEAT User Manual §5.2, “Mask filter”.

## The post-processing colour scale does not update

The minimum must be numerically smaller than the maximum. If the fields are initially empty, NEAT derives limits from the loaded map. Invalid limits leave the existing display unchanged.

Source: NEAT User Manual §5.2, “Colour scale”.

## ROI mean reports no data

Confirm that the ROI intersects valid map pixels. NEAT clips bounds to the valid range, but an empty region or a region containing only masked/invalid values cannot produce a meaningful mean.

Source: NEAT User Manual §5.2, “ROI mean” and “Tips & edge cases”.

## Strain calculation fails or gives an implausible result

**Checks:**

1. Enter a positive `d0` value.
2. Ensure the displayed metric is a fitted `d_*` spacing map; uncertainty and
   other parameter maps are not valid strain sources.
3. Ensure `d0` uses the same units and represents the same phase/reflection as
   the displayed map.
4. Confirm that the chosen reference is scientifically appropriate for the material and experiment.
5. Independently review the fitted spacing map and uncertainties.

NEAT applies `((d-d0)/d0) × 10^6`; it cannot determine whether the selected
reference state is scientifically valid. The result is lattice strain, not
stress.

Source: NEAT User Manual §5.2, “Strain from d0”.

## When should the assistant escalate instead of answering?

Escalate to the NEAT developer or an experienced beamline scientist when:

- the documented controls do not match the installed NEAT version;
- a repeatable crash or traceback occurs;
- files appear corrupted or unsupported;
- a result requires dataset-specific scientific validation;
- the question asks for a guaranteed fitting configuration;
- the available manual, FAQ and troubleshooting material do not support the proposed answer.
