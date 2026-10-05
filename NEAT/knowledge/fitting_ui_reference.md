# Bragg Edge Fitting Controls and Dialogs

This reference explains the controls in the **Bragg Edge Fitting** workflow and
the dialogs opened from its File, Setting and View menus.

## When fitting and mapping controls are enabled

The fitting actions require usable spectrum data. Load an image stack or line
profile, define a valid wavelength axis and select a phase and Bragg-edge
information before fitting.

The ROI-fit play button can operate on an extracted image ROI or on an imported
line profile. The mapping gear and batch-map buttons require spatial image
data, so they remain disabled for an imported line profile. They are also
disabled while image loading or a conflicting fit operation is active.

If a button is disabled, first check the **Messages** area for the loading or
validation result. Confirm that images are loaded, the wavelength axis exists,
the phase and wavelength range produce usable edge rows, and no fitting job is
still running.

## Phase selector and wavelength range

**Select Phase** chooses the crystal phase used to generate theoretical Bragg
edges. Choose `Unknown_Phase` when using the unknown-phase workflow; otherwise,
select a built-in or user-defined phase that represents the material being
analysed.

**Min WL (Å)** and **Max WL (Å)** restrict the displayed and fitted wavelength
range. Changing either value rebuilds the available edge table and refreshes
the fitting plots. The minimum must be lower than the maximum and the interval
must overlap the loaded wavelength data. The correct interval is
dataset-dependent and should include the selected edge and suitable baseline
regions.

Selecting a phase creates theoretical candidates; it does not prove that every
listed edge is present in the measured spectrum.

## Yellow ROI coordinate fields

**X Min**, **X Max**, **Y Min** and **Y Max** define the yellow ROI used to
extract the spectrum for an ROI fit. The bounds are integer pixel indices and
are half-open: the maximum values are excluded.

For example, `X Min=10`, `X Max=12`, `Y Min=20`, `Y Max=23` includes two
columns and three rows. The ROI must lie within the loaded image and have a
positive width and height. Editing the fields and finishing the edit updates
the image rectangle and extracted spectrum.

This yellow ROI is independent of the orange mapping area configured in
**Mapping Setting**.

## Bragg-edge table columns and row selection

The table contains:

- **hkl**: Miller indices for the reflection.
- **d**: theoretical or fitted d-spacing associated with that reflection.
- **1 Min** and **1 Max**: Region 1 pre-edge baseline bounds.
- **2 Min** and **2 Max**: Region 2 post-edge baseline bounds.
- **3 Min** and **3 Max**: derived full-edge fitting bounds. These columns are
  normally hidden in the main table.
- **s**, **t** and **eta**: initial values or fixed values for the edge-shape
  parameters.

Select one row to make it the active edge. Editable cells can be changed by
double-clicking, selecting and typing, or using the normal table edit key.
An incomplete or invalid row is skipped rather than treated as a usable edge.

The Region 1 and Region 2 bounds must preserve their intended order around the
edge. Inspect the plot after editing; a numerically accepted window is not
automatically a scientifically suitable one.

### Four wavelength values in an Edge Table row

The four normally visible wavelength fields are **1 Min**, **1 Max**, **2 Min**
and **2 Max**. They define the lower and upper bounds of the pre-edge Region 1
and post-edge Region 2. NEAT derives the full Region 3 fitting interval from
those entries; its **3 Min** and **3 Max** columns are normally hidden. These
row-specific bounds are different from the global Min WL and Max WL controls.

## Fixed parameter header checkboxes

The checkboxes in the `s`, `t` and `eta` column headers apply to those
parameters:

- checked means the table value is held fixed during fitting;
- unchecked means the parameter is fitted, using the table value as its
  starting value.

The fixed state applies across the relevant fitting operation and is remembered
between normal NEAT sessions. A fixed parameter has no fitted uncertainty from
that optimisation. Fixing weakly constrained parameters can stabilise a fit,
but the choice must be justified for the instrument, phase and dataset.

## Individual Edges and Edge Pattern controls

The first two icon buttons select the fitting mode:

- **Individual Edges** fits the selected valid edge rows independently.
- **Edge Pattern** fits multiple edges together with shared lattice
  information.

The play icon runs the selected fitting mode for the yellow ROI. The batch-play
icon applies the selected mode across the orange mapping area. The gear icon
opens **Mapping Setting**, and the stop icon requests cancellation of an active
batch fit.

Hover over an icon to see its tooltip or accessible action name. During a fit,
actions that would start a conflicting operation are temporarily disabled.
Use the messages and residual plots to judge the outcome; completion or
convergence alone is not scientific validation.

## Mapping Setting dialog

Open the dialog with the gear icon beside the fit controls.

**Macro pixel size** sets the width and height, in detector pixels, of the box
fitted at each mapping position. A larger box combines more spatial pixels and
usually improves counting statistics at the cost of spatial resolution.

**Skipping step** sets how far the box moves in X and Y between fits. Smaller
steps give denser, more overlapping samples and require more fits. **Enable
interpolation** fills positions between sampled locations in the exported map
when the step skips positions; it does not create new measured information.

**Mapping area** defines the orange ROI using half-open X/Y bounds. All entries
must be integers. **Apply** updates the settings without closing the dialog,
**OK** applies and closes it, and **Cancel** discards the un-applied edits. The
mapping settings do not replace the yellow ROI used for a single-ROI fit.

### What macro pixel size, skipping step and interpolation mean in Mapping Setting

In **Mapping Setting**, macro pixel size means the fitted box dimensions,
skipping step means the movement between fits, and interpolation means filling
output positions between fitted locations.

Macro-pixel **Width** and **Height** set the fitted detector-pixel box at every
mapping location. **Step X** and **Step Y** set the movement between successive
fits; a smaller step increases sampling density, overlap and running time.
**Enable interpolation** fills output positions between the locations that were
actually fitted when steps skip positions. Interpolation creates display/output
values, not additional neutron measurements.

## Instrument Setting dialog

Open **Setting > Instrument Setting...** to set:

- **Flight Path (m)**: accepted UI range 0 to 1000 m, displayed to three
  decimal places.
- **Time Delay (ms)**: accepted UI range -1 to 1 ms, displayed to four decimal
  places, with a 0.001 ms step.

Choose **OK** to apply and remember the values, or **Cancel** to leave the
current values unchanged. When the loaded wavelength axis depends on flight
path or ToF, accepting the dialog recalculates the loaded wavelength values.
For data that already has a wavelength axis independent of flight path, NEAT
reports that there is no flight-path-dependent loaded data to update.

A value being inside the UI range does not make it correct. Use calibrated
instrument geometry and the correct timing convention.

## NeXus and RADEN flight-path prompts

When a NeXus file contains sufficient geometry, NEAT presents the flight path
calculated from the file alongside the current app setting. Choose **Use NeXus
File** to use the file geometry, **Use App Setting** to use the value in
Instrument Setting, or **Cancel** to stop loading. If the file lacks sufficient
geometry, NEAT informs the user and uses the current app setting.

A RADEN ToF stack requires a positive flight path in **Instrument Setting**.
Before loading, NEAT shows the frame count, image dimensions, ToF range, chosen
app flight path and estimated image memory. **OK** continues and **Cancel**
leaves the current fitting input unchanged.

The chosen flight-path source is reported in the fitting messages. Confirm it
before interpreting fitted positions.

## Manual Spectra Setting dialog

Manual spectra anchors are used when MCP image loading cannot obtain a valid
spectra/ToF file and NEAT enables manual wavelength mode.

At the top, choose whether anchor values represent **Wavelength (Å)** or **Time
of Flight (ms)**. Each active row contains:

- **Image #**: the one-based image position in the stack;
- **Suffix**: the corresponding displayed image suffix; and
- the wavelength or ToF value assigned to that image.

An index displayed as **Unused** is ignored. **Add more lines** appends another
anchor row. Choose **OK** to store the active rows and rebuild the manual axis,
or **Cancel** to retain the previous anchors.

Anchor index and value fields are enabled only while manual wavelength mode is
active. When a valid automatic spectra axis is in use, saved anchors may still
be visible but are not applied. Use multiple correctly ordered anchors spanning
the stack; NEAT interpolates between them and extends the end slopes outside
the anchor range. One valid anchor produces a constant wavelength for the whole
stack and is generally not a meaningful wavelength calibration; use at least
two for a wavelength-resolved image stack.

## Phase Management dialog

The left panel lists available phases. Select one to inspect its definition,
choose **Add New** to clear the form for a new phase, or choose **Delete** to
remove the selected phase from the user's available list.

The form contains:

- **Name**: a non-empty unique phase name;
- **Structure**: `fcc`, `bcc`, `tetragonal`, `hexagonal` or `orthorhombic`;
- lattice parameters **a**, **b** and **c** in ångström; and
- an **hkl list**, with one integer `h,k,l` triplet per line.

Required lattice parameters are `a` for fcc/bcc, `a` and `c` for
tetragonal/hexagonal, and `a`, `b` and `c` for orthorhombic. Choose **Save** to
validate and store the phase for future sessions. A custom phase cannot reuse a
built-in phase name. Placeholder and unknown-phase entries cannot be deleted.

NEAT validates formatting and required fields, not the scientific correctness
of a phase definition. Check the structure, lattice constants and reflection
list against an authoritative source.

## Auto Adjust and Manual Adjust dialog

**View > Auto Adjust** sets the image display range to the 5th and 95th
percentiles of the current image, resets contrast to 1.0 and brightness to
zero, then redraws the image. It requires a loaded image.

**View > Manual Adjust...** opens **Adjust Image** with:

- **Contrast** from 0.01 to 2.00;
- **Brightness**, limited by the current display range;
- **Minimum** display intensity; and
- **Maximum** display intensity.

Slider and numeric controls stay synchronized. Minimum must remain below
maximum; otherwise NEAT warns and restores the previous valid limit. Reopening
the command raises the existing dialog instead of creating a duplicate.

Auto and manual adjustment change display scaling only. They do not modify the
loaded detector array, extracted spectrum or saved source images.

## Uncertainty Estimator dialog

Open **View > Uncertainty Estimator...** to explore the empirical relationship

`K = macro pixel size × Uamp × fitting uncertainty²`.

Under **Reference measurement**, enter a known macro-pixel size, Uamp and
fitting uncertainty. NEAT calculates the constant `K`. Under **Prediction**,
choose which one of those three quantities to estimate and enter the other two;
the estimated field is disabled and updated automatically. All three
quantities must be positive.

The curve panel fixes one selected parameter and plots the relationship between
the other two. Pan or zoom with the plot toolbar, and move the pointer near the
curve to read approximate values.

This estimator transfers an empirical constant from a reference measurement.
It is a planning aid, not a fitted uncertainty calculation for the currently
loaded dataset and not a guarantee that a chosen macro-pixel will produce a
valid fit.

## Batch Progress and Stop

During batch edge or pattern fitting, the non-modal **Batch Progress** window
shows the progress percentage and an estimated remaining time. Its **Stop**
button and the stop icon in the fitting controls request the same cooperative
cancellation.

A stop request may take time to reach a safe cancellation check. NEAT avoids
writing a final partial mapping result when cancellation is detected. Wait for
the message area and progress window to confirm whether the operation
completed, failed or was cancelled before starting another mapping job.

## Fitting configuration load and export

**File > Export fitting configuration** writes a metadata-style CSV containing
the wavelength range, selected phase, instrument values, fixed-parameter
states, edge-table rows and mapping settings.

**File > Load fitting configuration** requires a CSV beginning with the
`Metadata Name,Metadata Value` header. It applies recognised values when
available; absent or unrecognised values leave the corresponding current
setting unchanged. If mapping keys are present, the orange mapping settings are
applied.

After loading a configuration, verify the phase, wavelength range, edge rows,
flight path, delay, fixed states and mapping ROI against the current dataset.
The configuration is reusable settings, not a complete project and not proof
that settings from one experiment are appropriate for another.
