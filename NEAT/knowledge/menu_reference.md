# NEAT Menu and Command Reference

NEAT rebuilds the menu bar for the active workflow tab. The entries below use
the exact visible menu names.

## Data Preprocessing File menu

- **Clear Loaded Images** releases all image data and the loaded mask held by
  the preprocessing panels. It does not delete input or output files.
- **Clear Messages** empties the preprocessing message area only.
- **Exit** closes NEAT. Shortcut: `Ctrl+Q`.

The preprocessing controls for adding data, selecting an output, running a
stage, stopping it and saving messages are located directly inside the
preprocessing panels rather than in the File menu.

## Bragg Edge Fitting File menu

- **Load MCP images** loads an MCP image stack for fitting. Shortcut: `Ctrl+O`.
- **Load Nexus images** loads a NeXus image stack and establishes its ToF or
  wavelength axis from the supported NeXus information and instrument setting.
- **Load Raden images** loads a RADEN multi-page TIFF stack for fitting and
  collects the required wavelength or ToF information.
- **Load line profile** imports a one-dimensional wavelength/intensity profile
  from a supported CSV, TXT, XLSX or NeXus file. It enables ROI fitting but
  disables spatial mapping because no two-dimensional images are loaded.
- **Export line profile** saves the currently extracted wavelength/intensity
  spectrum so it can be inspected or reused.
- **Load fitting configuration** reads fitting and mapping settings stored in a
  compatible NEAT configuration CSV.
- **Export fitting configuration** writes the current fitting and mapping
  settings to a CSV for reuse. This is a settings exchange, not an export of the
  loaded images or a complete project.
- **Clear images** clears the current fitting input, extracted spectrum, edge
  table, plots and messages without deleting source files.
- **Exit** closes NEAT. Shortcut: `Ctrl+Q`.

Select the loader that matches the actual input format. Renaming a file does
not convert MCP, NeXus and RADEN data into one another.

### Export fitting configuration versus saving a project

**Export fitting configuration** saves reusable fitting and mapping settings to
a compatible CSV. It does not save the loaded image stack, extracted spectra,
fit-result arrays or open plot windows, and NEAT does not currently use it as a
complete project-save command. Keep the original images and exported result
files separately.

## Bragg Edge Fitting Setting menu

- **Instrument Setting...** sets the flight path in metres and time delay in
  milliseconds. When the loaded wavelength axis depends on those instrument
  values, accepting the dialog updates that loaded wavelength data.
- **Manual Spectra Setting...** defines image-index anchors as wavelength or
  time of flight. This supports datasets for which NEAT cannot obtain a
  satisfactory spectra axis automatically. An anchor row with index
  **Unused** is ignored.
- **Phase Management...** adds, edits or deletes user-defined material phases
  and their crystal information. Built-in and saved custom phases appear in the
  phase selector.

Instrument settings, manual spectrum anchors and phase definitions serve
different purposes. A fitting configuration CSV may restore compatible
settings, but it does not replace correct instrument metadata or scientific
phase selection.

## Bragg Edge Fitting View menu

- **Show Edge Line** shows or hides theoretical Bragg-edge guide lines on the
  fitting plots.
- **Live Fit Preview** enables or disables automatic fit previews as the active
  ROI or edge selection changes. It affects preview behaviour, not the stored
  source images.
- **Four Fit Canvases** switches between one combined fitting canvas and four
  canvases for the fit and its regions.
- **Auto Adjust** chooses image display limits automatically from the current
  image data.
- **Manual Adjust...** opens controls for manually changing the image display
  adjustment. Display adjustment changes how an image looks; it does not alter
  the underlying detector values.
- **Uncertainty Estimator...** opens the standalone uncertainty-estimation
  utility.
- **Symbol Size...** changes fitting-plot marker size from 1 to 20.
- **Increase UI Font** and **Decrease UI Font** change the application-control
  font.
- **Increase Canvas Font** and **Decrease Canvas Font** change the font used on
  fitting plots.

The checked states for edge lines, live preview and four-canvas layout, along
with symbol and font sizes, are remembered between normal NEAT sessions.

## Data Post-Processing File menu

- **Load Post-processing CSV...** loads a NEAT result CSV containing its
  metadata block and gridded parameter columns. Shortcut: `Ctrl+O`.
- **Load Image Result...** opens one or more two-dimensional FITS or TIFF
  parameter maps directly.
- **Clear Results** removes the loaded CSV, metadata and parameter buttons from
  the current session without deleting the result files.
- **Exit** closes NEAT. Shortcut: `Ctrl+Q`.

Use the CSV command when parameter names, X/Y coordinates and NEAT metadata are
needed. Use the image-result command to view an already exported two-dimensional
map.

## Common View menu and keyboard shortcuts

The preprocessing and post-processing View menus provide:

- **Increase UI Font** — `Shift+Up`.
- **Decrease UI Font** — `Shift+Down`.

The fitting View menu additionally provides:

- **Increase Canvas Font** — `Ctrl+Up`.
- **Decrease Canvas Font** — `Ctrl+Down`.

Other application-wide shortcuts are:

- **Exit** — `Ctrl+Q`.
- **Show or hide AI Assistant** — `Ctrl+Shift+A`.
- **Ask the AI Assistant** while its question box is active — `Ctrl+Enter`.
- **Open** — `Ctrl+O`, with behavior determined by the active fitting or
  post-processing tab.

UI font changes affect controls and messages. Canvas font changes affect the
fitting plot labels, ticks and legends. They are separate settings.

### UI font versus canvas font

To make NEAT controls and messages larger without changing fitting-plot labels,
choose **Increase UI Font** or press `Shift+Up`. Use **Decrease UI Font** or
`Shift+Down` to reverse it. The separate canvas-font commands, `Ctrl+Up` and
`Ctrl+Down`, change fitting plot labels, ticks and legends instead.

## AI Assistant menu

The AI Assistant menu is available for every workflow tab:

- **AI Assistant** shows or hides the dockable assistant panel.
- **AI Assistant Settings...** selects shared or personal access, model
  supplier, model and API credentials or a local model server.

## About menu

The About menu is available for every workflow tab:

- **About NEAT** shows the installed NEAT version, development group and main
  author information.
- **Check for Updates on Startup** enables or disables the automatic GitHub
  release check.
- **Check for Updates Now** checks immediately for a newer NEAT release. This
  requires a network connection but does not install an update automatically.
- **User Manual** opens the NEAT user manual in the default web browser.
- **GitHub Repository** opens the NEAT source repository.
- **Video Tutorials** opens the NEAT tutorial channel.

If an online resource or update check does not open, check the network
connection and default web-browser configuration. The local analysis functions
can still be used without opening those links.
