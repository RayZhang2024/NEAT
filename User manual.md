# NEAT User Manual

1. Introduction
1.1 Overview

NEAT (Neutron Bragg Edge Analysis Toolkit) is an open-source Python-based graphical user interface (GUI) designed for Bragg-edge neutron imaging data analysis. It enables users to visualise, fit, and interpret wavelength-resolved transmission data collected from neutron imaging beamlines such as IMAT (ISIS Neutron and Muon Source).

The software provides a streamlined workflow that integrates data loading, edge fitting, mapping, and batch processing — allowing both scientific and industrial users to efficiently extract quantitative information such as lattice spacing and residual strain from transmission spectra.

1.2 Purpose and Motivation

Bragg-edge imaging is a powerful neutron technique that reveals material structure and strain information through energy-resolved transmission. However, the analysis of such data is often time-consuming and requires specialised knowledge of fitting functions and image handling.

NEAT was developed to:

Simplify the analysis process for both new and experienced neutron users.

Provide a consistent and reproducible data analysis workflow.

Offer interactive visualisation tools to inspect transmission spectra, edge positions, and strain maps.

Facilitate collaboration and reproducibility through open-source code and transparent fitting algorithms.

1.3 Key Features

User-friendly GUI: intuitive layout for spectrum viewing, ROI selection, and batch analysis.

Flexible fitting functions: pseudo-Voigt model used, Jørgensen Bragg-edge models is planned to included in the future.

Result visualisation: generate maps of lattice spacing, strain, and fit quality.

Export tools: save fitted parameters and spectra as CSV.

1.4 Typical Workflow

Load Data: Import wavelength-resolved transmission files (.fits, .tiff, .npy, etc.).

Select ROI: Choose single points or regions for analysis.

Fit Bragg Edges: Apply fitting models to extract lattice spacing.

Visualise Results: Display fitted edges and 2D maps.

Export Output: Save results for further analysis or publication.

1.5 Intended Users

NEAT is intended for:

Researchers and engineers performing Bragg-edge imaging experiments.

Beamline scientists conducting in-situ studies at neutron imaging facilities.

Industrial users analysing strain, phase evolution, or microstructural variations in materials.

2. GUI Overview

NEAT’s graphical user interface (GUI) integrates the entire Bragg-edge imaging workflow — from loading and preprocessing data to edge fitting, mapping, and post-processing — within a single interactive environment. It consists of three major tabs: data preprocessing, Bragg edge fitting, and data post processing.

2.1 Data Preprocessing Tab

The Data Preprocessing tab is the first step in the NEAT workflow. It consolidates all image-level corrections and normalisation steps needed before Bragg-edge fitting.
Each panel corresponds to a specific operation in the preprocessing chain, and users can execute them individually or as a complete automated batch using the Full Process module.

| Panel                   | Purpose                                                                                                               | Key Operations                                                                                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Summation**           | Combines multiple raw run files into one dataset to improve signal-to-noise ratio (SNR).                              | - Add multiple runs of the same measurement.<br> - Specify output directory and file name.<br> - Click **Sum** to merge all runs.                                                                     |
| **Clean**               | Detects and replaces dead or hot pixels in the detector images.                                                       | - Load the dataset to be cleaned.<br> - Apply **z-score thresholding** to identify abnormal pixels.<br> - Replaces flagged pixels with a local 5×5 mean filter.                                       |
| **Overlap Correction**  | Corrects pixel pile-up effects that occur in Timepix detectors when multiple neutrons arrive in one shutter interval. | - Load sample images.<br> - Apply the **Tremsin multiplicity-recovery algorithm** to restore accurate short-wavelength intensities.                                                                   |
| **Normalisation**       | Removes detector and beam profile effects by dividing the corrected sample stack by the open-beam reference.          | - Load sample and corresponding open-beam datasets.<br> - Set spatial (`n`) and wavelength (`m`) smoothing window sizes.<br> - Click **Normalise** to generate transmission images.                   |
| **Filtering (Masking)** | Optionally filters data to exclude invalid regions or apply binary masks for macro-pixel averaging.                   | - Load dataset and optional binary mask.<br> - Exclude non-sample areas (e.g. apertures or gaps).                                                                                                     |
| **Full Process**        | Automates all preprocessing steps sequentially for multiple datasets organised in a standard directory structure.     | - Choose parent folder and output location.<br> - Define base name and macro-pixel size.<br> - Click **Full Process** to run Summation → Cleaning → Overlap Correction → Normalisation automatically. |

⚙️ Progress and Output Controls

Progress Bars: Each operation includes live progress indicators for image loading and processing.

Messages Panel: Displays real-time status messages, warnings, and completion confirmations.

Buttons:

Stop — interrupts an ongoing process.

Clear message / Save message — clears or exports the session log for record-keeping.

💡 Typical Workflow

Load Raw Runs → Add Data in Summation panel → click Sum.

Clean Detector Noise → Add Data in Clean panel → click Clean.

Apply Overlap Correction → Add Data and click Correct.

Normalise sample stack using Add Open Beam reference → click Normalise.

(Optional) Filter/Mask invalid pixels.

Run Full Process to execute all steps automatically for multiple datasets.

After these steps, the resulting normalised transmission dataset is saved to the selected output directory and ready to be loaded in the Bragg Edge Fitting tab.


2.1 Main Window Layout

The main GUI window is divided into the following key areas (see Figure 13 of the NEAT paper):

Viewport (top-left)

Displays the loaded transmission image or map.

Allows region-of-interest (ROI) definition using rectangle or polygon selection tools.

Supports zooming, panning, and pixel-value inspection.

Spectrum-Extraction Panel (bottom-left)

Used to define the macro-pixel size (e.g. 10 × 10 pixels) and wavelength window for spectral analysis.

The “Pick” button extracts the averaged transmission spectrum from the selected region.

The extracted spectrum is displayed immediately on the right-hand canvas for verification.

Bragg-Edge Table (centre-left)

Lists all theoretical Bragg edges of the selected material phase that fall within the chosen wavelength range.

Each edge entry includes editable left, right, and edge window boundaries, as well as fit parameters (s, t, η).

Users can choose to fix or free each parameter during fitting.

Fitting Control Panel (bottom-centre)

Contains the main fitting and batch-processing buttons:

Fit edges – performs separate three-stage fits for individual edges.

Fit pattern – performs simultaneous multi-edge (pattern) fitting with a shared lattice parameter a.

Batch edge / Batch pattern – applies the validated fit automatically across a selected ROI using the specified macro-pixel and pixel-skip settings.

Fit parameters and processing options are stored automatically to ensure reproducibility.

Right-Hand Canvas (top-right)

Displays the transmission spectrum and fitted curves for visual comparison.

Overlays residuals and key parameters for quality assessment.

Message Pane / Log Window (bottom-right)

Outputs numerical fitting results, χ² values, and diagnostic information in real time.

Provides progress messages during batch operations.

4.2 Toolbar and General Settings

The upper toolbar provides quick access to essential functions and preferences:

Icon / Menu	Function
📂 Load Data	Opens normalised or raw datasets (.fits, .tiff, .npy, etc.) into the viewport.
⚙️ Pre-processing	Launches automated run summation, pixel cleaning, overlap correction, and normalisation.
🧩 Phase / Edge Table	Opens the phase-selection dialog and populates the Bragg-edge list.
▶ Fit Controls	Executes test fits or batch runs using the parameters defined in the fitting panel.
🗺️ Post-Processing	Loads CSV result files to visualise fitted parameters as 2D maps (lattice spacing, strain, width, height).
💾 Export	Saves spectra, fit parameters, or maps to CSV/TIFF for further analysis.
🔧 Settings	Opens the configuration dialog where users can set: flight path length, time-zero correction, default macro-pixel size, pixel-skip factor, and plotting preferences.
❓ Help	Displays version information and a link to the online manual or GitHub repository.
4.3 Tabs and Their Functions
Tab	Description	Typical Use
Create Input / Pre-Process	Handles input selection, run summation, pixel cleaning, and normalisation.	Combine and prepare datasets before analysis.
Build Model / Fitting	Defines macro-pixel, wavelength window, fitting model, and executes fits (single-edge or pattern).	Extract lattice spacing, edge width, and height.
Batch Mode	Applies validated fits over ROIs with pixel-skip acceleration.	Generate full-field maps rapidly.
Post-Processing	Loads and visualises CSV results as colour-mapped 2D distributions; supports line-profile extraction and strain calculation using a user-specified d₀.	Inspect spatial variations and quantify strain.
Settings / Preferences	Stores global paths, plotting options, and default parameters.	Customise behaviour and display style.
4.4 Real-Time Visualisation

Fitted curves update instantly on the canvas after each run, enabling on-the-fly inspection.

Colour-mapped results (lattice parameter, edge width, edge height, strain) can be rendered in separate pop-up windows for interactive exploration.

Line-profiles between two selected points provide quick quantitative checks of strain gradients.

4.5 Workflow Summary through the GUI

Load and pre-process data via the “Create Input” tab.

Extract spectrum from ROI → verify signal quality.

Perform edge or pattern fit → check residuals and fitting statistics.

Launch batch processing → generate parameter maps.

Visualise and export results in the “Post-Processing” tab.