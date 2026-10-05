---
title: Application startup, window composition and persistent state
doc_id: neat-tech-architecture-startup-state
doc_type: technical_reference
functional_area: architecture
audience: [user, developer, support]
neat_version: 4.8.0
verified_commit: 3a22254e1ffdfd12148929968263a0e52a545fe6
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/app.py, NEAT/package_resources.py, NEAT/ui/main_window.py]
source_symbols: [create_app, create_launch_splash, main, FitsViewer.__init__, FitsViewer.load_user_settings, FitsViewer.save_user_settings, FitsViewer.closeEvent]
test_paths: [tests/test_assistant_panel.py, tests/test_package_resources.py]
---

# Application startup, window composition and persistent state

## Launch sequence

Both `python -m NEAT.app` and the installed `neat` command call
`NEAT.app.main`. Startup:

1. enables Qt high-DPI scaling and high-DPI pixmaps;
2. creates one `QApplication` and applies a maximum line-edit height;
3. resolves the packaged `NEAT/assets/launch_splash.png` resource, or displays
   a generated fallback when it is missing or unreadable;
4. imports `FitsViewer` after the splash is visible;
5. builds and shows the main window; and
6. enters the Qt event loop.

The splash is capped at approximately 1100 by 720 logical pixels and uses the
screen device-pixel ratio when the source image has enough resolution.
Source/package installs resolve it through `importlib.resources`; the
PyInstaller build bundles the same package-owned asset.
`KeyboardInterrupt` at the event loop produces process exit code 130.

## Main-window composition

`FitsViewer` combines a `QMainWindow` with the preprocessing, fitting and
post-processing mixins. Its central west-tabbed `QTabWidget` contains:

- Data Preprocessing;
- Bragg Edge Fitting; and
- Data Post-Processing.

All three tab interfaces are constructed before saved settings are restored.
The AI assistant is created as a right-side dock, and the native menu bar is
rebuilt whenever the top-level tab changes.

Important in-memory state includes loaded image arrays, wavelength/ToF axes,
ROI and display values, fitting parameters, mapping workers and post-processing
CSV data. Analysis arrays and chat history are not persisted across launches.

## User settings

GUI settings are JSON at:

```text
~/.neat_gui_settings.json
```

Saved values include GUI and plot font sizes, flight path, delay, phase,
initial ROI, wavelength window, fixed `s`/`t`/`eta` switches, symbol size,
edge-line and live-preview visibility, fitting plot layout, last top-level tab,
assistant visibility, manual anchors, mapping box/step/ROI settings and update
preferences.

Loading a missing, unreadable or invalid file silently keeps application
defaults. Saving also suppresses write errors. Text fields are stored as text;
most numeric validation therefore occurs later when a workflow uses them.

## Custom phases

User-defined phases and built-in phase removals are separate from GUI settings:

```text
~/.neat_custom_phases.json
```

At startup, valid custom definitions are combined with built-in `PHASE_DATA`
after applying recorded removals. The phase file is per operating-system user,
not per dataset or NEAT project.

## Close behavior

Closing first asks the assistant dock to stop. If its request thread does not
finish within 1.5 seconds, NEAT ignores the close event and asks the user to
close again after the answer or error appears. Otherwise it saves settings,
requests known workers to stop, waits up to one second per running worker,
clears large fitting arrays and lets Qt process the close event.

The window uses `WA_DeleteOnClose=False`, so object destruction is not the
primary cleanup mechanism.

## Failure and recovery notes

- A startup import error occurs before the main window appears if a required
  core dependency is missing.
- The splash has no staged exception dialog; console launches show the Python
  traceback.
- Invalid settings are ignored as a whole only when JSON loading fails.
  Individual well-formed but unsuitable values can fail later in a workflow.
- User settings do not record loaded paths, image data, fitting results or
  assistant conversation history.

## Retrieval questions

- What happens when NEAT starts?
- Which tabs are created in the main window?
- Where does NEAT save GUI settings and custom phases?
- Which settings persist after closing NEAT?
- Why can NEAT refuse to close while the AI assistant is busy?

