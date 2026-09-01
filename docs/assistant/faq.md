# NEAT Assistant FAQ

Knowledge-base version: NEAT 4.8.2
Approved source: `User manual.md`

This file contains concise answers for common NEAT usage questions. Answers describe documented NEAT behaviour only. Dataset-specific scientific interpretation should be reviewed by an experienced Bragg-edge imaging scientist.

## What is the normal NEAT workflow?

If you have just received a Bragg-edge dataset and are unsure where to start,
use the following normal NEAT workflow:

1. Preprocess the wavelength-resolved images.
2. Load the normalised or filtered dataset in **Bragg Edge Fitting**.
3. Set the flight path and, when appropriate, select the material phase.
4. Extract a representative macro-pixel spectrum.
5. Configure the global wavelength range and the visible Window 1 and Window 2 bounds in the Edge Table.
6. Test an individual-edge or pattern fit.
7. Run batch fitting over the required region of interest.
8. Load the resulting CSV file in **Data Post-Processing** to create maps, strain calculations and line profiles.

Source: NEAT User Manual §§1.4, 4 and 5.

## Which data should I load for Bragg-edge fitting?

Load a normalised or filtered wavelength-resolved dataset. Normalisation divides the sample images by an open-beam reference. Filtering can additionally apply a binary mask to exclude invalid or out-of-sample pixels.

Source: NEAT User Manual §§3.4, 3.5 and 4.1.

## When should I use Summation?

Use **Summation** when the same sample has two or more compatible runs that should be combined to improve counting statistics. If there is only one run, skip this stage. Every run being summed must contain the same set of image frames.

Source: NEAT User Manual §3.1.

## What folder structures does Summation accept?

NEAT accepts either:

- a parent folder containing multiple run folders; or
- a parent folder containing multiple sample folders, where every sample contains multiple run folders.

Do not mix these two depths under the same parent. NEAT detects mixed structures and stops rather than combining ambiguous data.

Source: NEAT User Manual §3.1.

## What does Clean do?

**Clean** detects and replaces outlier pixels in FITS image stacks and writes cleaned frames to a new output folder. It can process one dataset folder or a parent folder containing several datasets.

Source: NEAT User Manual §§2.1 and 3.2.

## What files are required for Overlap Correction?

Each dataset requires its FITS images plus readable `*_Spectra.txt` and `*_ShutterCount.txt` files. If either text file is missing or cannot be read, NEAT skips that dataset and reports the reason in the message pane.

Source: NEAT User Manual §3.3.

## Why would I use Overlap Correction?

Overlap Correction compensates for detector count overlap or pile-up effects
that can bias measured image intensities. NEAT uses the time-of-flight segments
from `*_Spectra.txt` together with the corresponding shutter counts from
`*_ShutterCount.txt` to correct and interval-scale the image intensities.

Use this stage when the acquisition requires pile-up/overlap correction and the
matching metadata files are available. It is not a generic image-cleaning step:
whether it is required depends on the detector and acquisition. Ask the
instrument scientist if this is uncertain.

Source: NEAT User Manual §3.3 and documented Overlap Correction behaviour.

## What does Normalisation do?

Normalisation divides each sample image by an open-beam reference to reduce source and detector flux variations. NEAT can process a single sample dataset or multiple dataset folders sequentially, using a separately selected open-beam folder.

Source: NEAT User Manual §3.4.

## What do Window half and Adjacent mean during normalisation?

**Window half (`n`)** controls spatial averaging. The spatial kernel is `(2n+1) × (2n+1)` pixels. Larger values generally produce smoother images but reduce spatial detail.

**Adjacent (`m`)** controls averaging across neighbouring wavelength or frame indices. The temporal window contains `(2m+1)` frames. `m=0` means that only the current frame is used.

Source: NEAT User Manual §3.4.

## What does Filtering do?

Filtering applies one binary FITS mask to a loaded dataset. The mask must contain the intended keep/drop regions and must have exactly the same image dimensions as the data frames. It is useful for excluding background or invalid regions before fitting.

Source: NEAT User Manual §3.5.

## What does Full Process run?

**Full Process** runs summation when multiple sample runs are present, followed by outlier removal, overlap correction and normalisation. It requires a sample folder, an open-beam folder and a writable output folder. Its normalisation stage currently uses `Adjacent = 0`.

Source: NEAT User Manual §3.6.

## How does NEAT obtain the wavelength axis?

NEAT normally uses the wavelength or time-of-flight information associated with the loaded dataset. If a valid spectra text file is unavailable, it enters manual wavelength mode. In that mode, define at least two image-index anchors in **Manual Spectra Setting** and choose whether their values represent wavelength or time of flight. NEAT linearly interpolates the wavelength axis between the anchors.

Source: NEAT User Manual §4.1, “Manual anchors in Config”.

## What is a macro-pixel?

A macro-pixel is a spatial region whose pixels are combined to obtain an averaged transmission spectrum. A larger region can improve counting statistics and fitting stability, but it reduces spatial resolution and may mix different sample regions. Use a stable, representative area for the initial test fit and avoid strong spatial gradients.

Source: NEAT User Manual §§2.2, 4.2 and 4.5.

## How should I select the wavelength range?

Set the global minimum and maximum wavelength so that the target Bragg edge or edges are included. After extracting a representative spectrum, confirm visually that the required edge lies inside the displayed range before configuring the visible Window 1 (`1 Min`, `1 Max`) and Window 2 (`2 Min`, `2 Max`) bounds in the Edge Table.

Source: NEAT User Manual §§4.2 and 4.3.

## What are the Window 1 and Window 2 bounds?

For each selected Bragg edge:

- **Window 1** (`1 Min`, `1 Max`) selects the lower-wavelength, pre-edge baseline region.
- **Window 2** (`2 Min`, `2 Max`) selects the higher-wavelength, post-edge baseline region.
- The full Bragg-edge fitting interval runs from **`1 Min` to `2 Max`**, so it includes both baseline regions and the transition between them.

In NEAT v4.8.1, the former independent `3 Min` and `3 Max` inputs are not shown. NEAT derives the full fitting interval automatically from `1 Min` and `2 Max`. If a fit is biased or unstable, inspect the two visible windows and the complete interval against the displayed spectrum before changing model parameters.

Source: NEAT User Manual §4.3 and Appendix A.2.

## How do I configure the Edge Table?

After loading a wavelength-resolved dataset:

1. Set the global wavelength minimum and maximum so they include the expected edge or edges.
2. Select the material phase, when known, and choose **Pick** to populate the Edge Table.
3. Select the edge row required for fitting.
4. Adjust its four visible wavelength values against a representative spectrum:
   - `1 Min` and `1 Max` define Window 1, the lower-wavelength pre-edge baseline.
   - `2 Min` and `2 Max` define Window 2, the higher-wavelength post-edge baseline.
   - the full Bragg-edge fit runs from `1 Min` to `2 Max` and must contain the transition.
5. Review the initial `s`, `t` and `eta` values and use their header checkboxes to choose whether each parameter is fixed or refined.
6. Test the selected edge or pattern on a representative macro-pixel before batch fitting.

In v4.8.1, `3 Min` and `3 Max` are not visible inputs; NEAT derives the full
fitting interval automatically. There are no universal Edge Table values that
work for every dataset.

Source: NEAT User Manual §§4.2–4.4 and Appendix A.2.

## What is the difference between Fit edges and Fit pattern?

**Fit edges** fits individual Bragg edges. **Fit pattern** performs a multi-edge fit in which the selected edges share a lattice parameter `a`. Test the chosen fitting mode on a representative macro-pixel before running its corresponding batch mode.

Source: NEAT User Manual §§2.2 and 4.4.

## What should I inspect after a test fit?

Inspect the fitted curves and residual behaviour on the right-hand canvas. Review the message pane for convergence information, fitted parameters and uncertainties. A converged numerical fit should still be checked for physical plausibility and sensitivity to the selected windows.

Source: NEAT User Manual §4.4.

## How can I make full-field fitting faster?

Use a positive pixel-skip or step size so NEAT fits a subset of spatial positions and interpolates the skipped locations. Larger steps can reduce computation substantially, but they also reduce the density of directly fitted points. Keep the mapping macro-pixel size consistent with the successful test fit.

Source: NEAT User Manual §4.5.

## What files does batch fitting produce?

After **Batch edges** or **Batch patterns** completes, NEAT saves the results in CSV files. Load an appropriate result CSV in **Data Post-Processing** to visualise fitted parameter maps.

Source: NEAT User Manual §§4.6 and 5.

## What do the result CSV parameter buttons mean?

After loading a batch-result CSV, Data Post-Processing creates one button for
each data column other than the `x` and `y` spatial coordinates. The exact
buttons depend on whether the CSV came from individual-edge or pattern fitting.

Common columns are:

- `d_...`: fitted lattice spacing for an individual Bragg edge.
- `a`, `c` or other lattice-parameter names: lattice parameters refined by pattern fitting; the available names depend on the selected crystal structure.
- `s_...`: wavelength-like edge broadening associated with sample microstructure.
- `t_...`: wavelength-like edge broadening associated with the instrument.
- `eta_...`: dimensionless instrument neutron-pulse edge-shape parameter.
- `fwhm_...`: fitted Bragg-edge full width at half maximum, in wavelength units.
- `height_...`: fitted change in signal across the Bragg edge.
- names containing `_unc`: the estimated uncertainty associated with the corresponding fitted parameter.

The suffix identifies the relevant edge or Miller indices. Selecting a button
creates a two-dimensional map from that CSV column. A map being available does
not by itself establish that its fitted values are scientifically reliable;
review fit quality, uncertainty and physical plausibility.

Source: NEAT User Manual §§4.6 and 5 and documented batch-result CSV behaviour.

## How is strain calculated in Data Post-Processing?

Enter a positive reference value `d0` in the same units as the displayed spacing parameter, then select **Calculate Strain**. NEAT calculates

`strain = ((d - d0) / d0) × 1,000,000`

and displays the result in microstrain. Calculate Strain is available only for
fitted `d_*` spacing maps, not uncertainty or other parameter maps. This is
lattice strain, not stress; stress additionally requires appropriate elastic
constants and measurement geometry. The scientific suitability of the
selected reference value is the user’s responsibility.

Source: NEAT User Manual §5.2, “Strain from d0”.

## How do I obtain a line profile?

Load a batch-result CSV, display the required map and enable **Select Points**. Click the start and end points of the desired path. NEAT draws the path and opens a line-profile window. Additional click pairs create additional profiles.

Source: NEAT User Manual §5.2, “Point selection & line profile”.

## Can NEAT display coordinates in millimetres?

Yes. Enable **Display in mm** in Data Post-Processing. NEAT changes the axes and coordinate inputs from pixels to millimetres using `0.055 mm/pixel`. In v4.8.1 this value is fixed in the software and is not editable in the interface.

Source: NEAT User Manual §5.2, “Units”.
