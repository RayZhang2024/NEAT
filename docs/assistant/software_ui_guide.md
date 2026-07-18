# NEAT Software and User Interface Guide

This guide describes the visible NEAT application, its main work areas and the
ways a user can manage the interface. It is user guidance, not a description of
the source-code architecture.

## Main workflow tabs and context-sensitive menu bar

NEAT is organised into three top-level workflow tabs shown vertically on the
left:

1. **Data Preprocessing** prepares detector-image stacks before fitting.
2. **Bragg Edge Fitting** loads images or a line profile, extracts spectra,
   fits Bragg edges and creates maps.
3. **Data Post-Processing** opens fitting results and displays or exports
   parameter maps.

The menu bar changes when the active tab changes. This is intentional: each tab
shows only the File, Setting and View actions that apply to that workflow.
When a user switches from fitting to postprocessing, the menus therefore change
from fitting commands to postprocessing commands.
Consequently, `Ctrl+O` loads MCP images while **Bragg Edge Fitting** is active,
but loads a post-processing CSV while **Data Post-Processing** is active.

The active top-level tab is remembered when NEAT closes and is restored at the
next launch.

## Data Preprocessing screen main areas and controls

The preprocessing tab contains six functional panels:

- **Summation**: add data, select an output folder and base name, then sum
  matching image runs.
- **Clean**: load images and remove invalid pixels and positive spike outliers.
- **Overlap Correction**: correct an image stack using the required overlap
  inputs.
- **Normalisation**: load sample and open-beam images, set the output and the
  `n` and `m` windows, then normalise.
- **Filtering**: load data and either load an existing mask or create one with
  **Generate Mask**.
- **Full Process**: run the combined preprocessing workflow from sample and
  open-beam inputs.

Each panel has separate image-loading and processing progress bars. Its **Stop**
button is enabled while that operation is running. The message area reports
loading, validation, processing and output information. **Clear message**
empties only the visible message area; **Save message** writes its current text
to a user-selected text file.

For the algorithm, input structure and output details of an individual panel,
consult the corresponding preprocessing technical section.

## Main areas of the Bragg Edge Fitting screen

The fitting workspace is split into resizable areas:

- The upper-left image viewer displays the current detector image. The vertical
  slider moves through the loaded image stack.
- The lower-left controls select the phase, initial yellow ROI, wavelength
  range, Bragg-edge rows, fitting mode and fitting or mapping action.
- The upper-right area displays the extracted spectrum and fitting results.
- The lower-right **Messages** area reports loading, fitting and export status.

The **Individual Edges** mode fits selected Bragg edges independently. The
**Edge Pattern** mode fits multiple edges together with shared lattice
parameters. The play button fits the current yellow ROI. The batch-play button
maps the orange mapping area. The gear button opens **Mapping Setting**, where
the macro-pixel width and height, X/Y step, interpolation and mapping bounds are
configured. A line profile can be fitted, but image mapping is disabled when
the current input is a profile rather than an image stack.

In the edge table, select a row to work with that edge. The header checkboxes
over `s`, `t` and `eta` control whether each parameter is fixed or fitted.
Double-click or select an editable cell to change a fitting value.

## Fitting image viewer and ROI controls

The yellow rectangle is the ROI used to extract the spectrum for an ROI fit.
The orange rectangle is the area traversed during batch mapping.

- Drag inside either rectangle to move it.
- Hold `Ctrl` and drag a rectangle corner to resize it.
- Drag outside the rectangles to pan the image when the Matplotlib toolbar is
  not already in pan or zoom mode.
- Use the mouse wheel over the image to zoom around the pointer.
- Edit the X Min, X Max, Y Min and Y Max fields to position the yellow ROI
  numerically.

The ROI bounds are half-open: X Max and Y Max are excluded. For example,
`x[10, 12), y[20, 23)` includes X pixels 10 and 11 and Y pixels 20, 21 and 22.
Changing the mapping area does not overwrite the yellow ROI used for the
single-ROI spectrum.

The Matplotlib toolbar can also provide standard navigation controls. Direct
drag and wheel gestures are disabled while its pan or zoom mode is active, to
avoid conflicting actions.

## Data Post-Processing screen main areas and controls

Use **Load CSV File** for a NEAT mapping-result CSV. The left side displays the
file metadata. The right side creates one button for each result column other
than `x` and `y`; selecting a parameter button opens that parameter map in its
own plot window.

Use **Load Image Result** to open one or more two-dimensional FITS or TIFF
parameter maps directly. Each selected image opens in a separate plot window.
An image with more than two dimensions uses its first plane when possible; an
input that cannot be reduced to a two-dimensional image is skipped with a
warning.

**Clear Results** removes the loaded CSV, metadata and parameter buttons from
the post-processing workspace. Plot windows that were already opened are
separate windows and can be closed with their own window controls.

## Resizing the NEAT work areas

The fitting tab uses splitter boundaries between the image, controls, plots and
message area. Drag a splitter boundary to give more space to the part currently
being inspected. The main application can also be resized normally, down to its
minimum supported window size.

The tab names are placed on the left edge. If the available height is too
small, the tab control provides scrolling buttons rather than removing a
workflow.

## Managing the AI Assistant panel

The **NEAT AI Assistant** is a dockable panel. It starts on the right and can be
docked on either the left or right side, resized by dragging its boundary,
floated using the normal dock controls, or hidden.

Choose **AI Assistant > AI Assistant** or press `Ctrl+Shift+A` to show or hide
it. Open **AI Assistant > AI Assistant Settings...** to select access and model
settings. Type a question and choose **Ask**, or press `Ctrl+Enter`. **Clear**
clears the visible conversation and its in-memory chat history; it does not
delete the knowledge documents or the locally stored feedback log.

Choose **Explain current screen** for an overview of the active NEAT workflow,
its main controls, prerequisites and cautions. The request includes the active
workflow name and the same small allow-listed screen context used by an ordinary
assistant question. It does not attach images or file paths. If text is already
typed in the question box, the draft is preserved.

Only one assistant request can run at a time. While it is working, **Ask** and
**Clear**, together with **Explain current screen**, are temporarily disabled.
If NEAT is closed while an assistant request is still finishing, NEAT asks the
user to try closing again after the answer or error appears.

The assistant uses approved NEAT guidance and a small allow-listed set of
on-screen settings. Raw images and application file paths are not sent. The
conversation is kept in memory for the current NEAT session only. Choosing
**Helpful** or **Not helpful** stores an optional local rating without storing
the generated answer or raw data.

## What NEAT remembers between sessions

When NEAT closes normally, it stores selected interface and fitting preferences
for the current Windows user. These include:

- UI and fitting-canvas font sizes;
- the last active workflow tab and AI Assistant visibility;
- phase, instrument flight path and delay;
- the initial yellow ROI and wavelength range;
- whether `s`, `t` and `eta` are fixed;
- edge-line visibility, live-fit preview, one/four-canvas layout and symbol
  size;
- manual wavelength or ToF anchors;
- mapping box size, step, interpolation and orange mapping bounds; and
- update-check preferences.

These remembered values are starting settings, not saved analysis results.
Loaded image arrays, extracted spectra, fit results, post-processing CSV data
and assistant conversation history must be loaded or generated again after
restarting NEAT.

User-defined phase records are also stored locally for reuse. If a settings
file is missing or cannot be read, NEAT retains its built-in defaults.

## What the Clear commands do

Clear commands release data from the running NEAT session; they do not delete
the user's source images, CSV files or previously exported results from disk.

- **Data Preprocessing > File > Clear Loaded Images** releases images and masks
  loaded by all preprocessing panels and resets their progress bars. NEAT asks
  the user to stop an active preprocessing operation before clearing.
- **Data Preprocessing > File > Clear Messages** empties only the preprocessing
  message area.
- **Bragg Edge Fitting > File > Clear images** stops an active image loader,
  clears the loaded stack, spectrum, edge table, fitting plots and fitting
  messages, and returns the fitting workspace to an unloaded state.
- **Data Post-Processing > File > Clear Results** clears the loaded result CSV,
  displayed metadata and generated parameter buttons.
- **AI Assistant > Clear** clears only the current in-memory conversation.

Use the relevant **Stop** button before clearing when a preprocessing or fitting
calculation is running.
