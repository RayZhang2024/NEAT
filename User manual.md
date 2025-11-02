# NEAT User Manual

## A video tutorial is available on Youtube.
[![Watch the video](https://img.youtube.com/vi/fbucLB5Bypc/hqdefault.jpg)](https://www.youtube.com/watch?v=fbucLB5Bypc)

# 1. Introduction

## 1.1 Overview

NEAT (Neutron Bragg Edge Analysis Toolkit) is an open-source Python-based graphical user interface (GUI) designed for Bragg-edge neutron imaging data analysis. It enables users to visualise, fit, and interpret wavelength-resolved transmission data collected from neutron imaging beamlines such as IMAT (ISIS Neutron and Muon Source).

The software provides a streamlined workflow that integrates data loading, edge fitting, mapping, and batch processing — allowing both scientific and industrial users to efficiently extract quantitative information such as lattice spacing and residual strain from transmission spectra.

## 1.2 Purpose and Motivation

Bragg-edge imaging is a powerful neutron technique that reveals material structure and strain information through energy-resolved transmission. However, the analysis of such data is often time-consuming and requires specialised knowledge of fitting functions and image handling.

NEAT was developed to:

Simplify the analysis process for both new and experienced neutron users.

Provide a consistent and reproducible data analysis workflow.

Offer interactive visualisation tools to inspect transmission spectra, edge positions, and strain maps.

Facilitate collaboration and reproducibility through open-source code and transparent fitting algorithms.

## 1.3 Key Features

User-friendly GUI: intuitive layout for spectrum viewing, ROI selection, and batch analysis.

Flexible fitting functions: pseudo-Voigt model used, Jørgensen Bragg-edge models is planned to included in the future.

Result visualisation: generate maps of lattice spacing, strain, and fit quality.

Export tools: save fitted parameters and spectra as CSV.

## 1.4 Typical Workflow

Load Data: Import wavelength-resolved transmission files (.fits).

Select ROI: Choose single points or regions for analysis.

Fit Bragg Edges: Apply fitting models to extract lattice spacing.

Visualise Results: Display fitted edges and 2D maps.

Export Output: Save results for further analysis or publication.

## 1.5 Intended Users

NEAT is intended for:

Researchers and engineers performing Bragg-edge imaging experiments.

Beamline scientists conducting in-situ studies at neutron imaging facilities.

Industrial users analysing strain, phase evolution, or microstructural variations in materials.

# 2. GUI Overview

NEAT’s graphical user interface (GUI) integrates the entire Bragg-edge imaging workflow — from loading and preprocessing data to edge fitting, mapping, and post-processing — within a single interactive environment. It consists of three major tabs: data preprocessing, Bragg edge fitting, and data post processing.

## 2.1 Data Preprocessing Tab

The Data Preprocessing tab is the first step in the NEAT workflow. It consolidates all image-level corrections and normalisation steps needed before Bragg-edge fitting.
Each panel corresponds to a specific operation in the preprocessing chain, and users can execute them individually or as a complete automated batch using the Full Process module.


| **Panel**               | **Purpose**                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Summation**           | Combines multiple raw run files into one dataset to improve signal-to-noise ratio (SNR).                         |
| **Clean**               | Detects and replaces dead or hot pixels in the images using thresholding and spatial averaging.                  |
| **Overlap Correction**  | Corrects pixel pile-up effects in Timepix detectors, restoring accurate short-wavelength intensities.            |
| **Normalisation**       | Normalisation by dividing the corrected sample stack by the corrected open-beam reference.                       |
| **Filtering (Masking)** | Optionally filters data to exclude invalid regions by a binary mask.                                             |
| **Full Process**        | Automates all preprocessing steps sequentially for datasets, from summation to normalisation.                    |

![Data Preprocessing Tab](docs/images/Data_Preprocessing_Tab.png)

## 2.2 Bragg Edge Fitting Tab

| **Section**                       | **Purpose**                                                                                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **(a) Viewport**                  | Displays the loaded image and allows users to define a **Region of Interest (ROI)** interactively. The ROI determines which pixels or macro-pixels are used for spectral extraction.                                                 |
| **(b) Spectrum-Extraction Panel** | Sets the **macro-pixel size** (e.g. 10 × 10 pixels) and **wavelength range** for fitting. Pressing **Pick** extracts and displays the averaged transmission spectrum from the selected ROI.                                          |
| **(c) Bragg-Edge Table**          | Lists all theoretical Bragg edges within the chosen wavelength range for the selected phase. Each row includes editable left/right/edge windows and model parameters (*s*, *t*, *η*), which can be fixed or refined.                 |
| **(d) Fitting Controls**          | Provides main fitting commands — **Fit edges** (individual three-stage fits), **Fit pattern** (multi-edge fitting with shared lattice parameter *a*), and **Batch Edges / Batch Pattern** for automated high-throughput ROI mapping. |
| **(e) Right-Hand Canvas**         | Displays extracted transmission spectra, fitted curves, and residuals. Updates occur in real time after each fit to enable rapid assessment of fit quality.                                                                          |
| **(f) Message Pane**              | Shows numerical results, fitting statistics, and diagnostic information. Also reports batch-processing progress and status messages during fitting operations.                                                                       |


![Bragg Edge Fitting Tab](docs/images/Bragg_Edge_Fitting_Tab.png)


## 2.3 Data Post Processing Tab



| **Section**                     | **Purpose**                                                                                                                                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Batch-Fitting Data Import**   | Allows the user to load previously saved batch-fitting results (CSV file) containing fitted parameters such as lattice parameter *a*, Bragg-edge position, or strain.                                                          |
| **Metric Selection**            | Enables the user to choose which fitted parameter to visualise — for example, lattice parameter *a*, edge shift, or intensity. The selected metric determines the data displayed in the 2D map.                                |
| **2D Map Rendering**            | Displays a spatially resolved map of the chosen metric over the defined ROI, providing an intuitive visualisation of parameter distribution across the sample. A dynamic colour scale bar indicates quantitative values.       |
| **Line Profile Extraction**     | Lets the user select two points on the 2D map to define a line path. The corresponding variation of the metric along this line is plotted in a **pop-up window**, allowing detailed analysis of local gradients or interfaces. |
| **Export and Analysis Options** | Offers options to save the generated 2D maps and line-profile plots for further documentation, reporting, or comparison with simulation results.                                                                               |


![Data Post Processing Tab](docs/images/Data_Post-Processing_Tab.png)

# 3. Functions
## 3.1 Data Preprocessing
### 3.1.1 Summation

* Combines multiple **runs of FITS images** by **pixel-wise addition** to boost SNR.
* Supports batch processing for multiple samples.
* Writes the **summed images** into a chosen output folder with your **base name**.

## 1) Quick strat (2-level example)

1. **Add data** → choose the containing `/Run_01`, `/Run_02`, `/Run_03`.
2. **Set output** → choose a folder that will store the summed images.
3. **Base name** → i.e. `Fe_summed`.
4. **Sum** → wait for both progress bars to finish.
5. Find results at:

   ```
   /Results/Summed_TopFolder/Fe_summed_*.fits
   ```

---

## 2) Prepare your data (folder layouts NEAT accepts)

NEAT detects the layout automatically and **refuses mixed depths**.

Case 1: if you have a measurement of a sample consists of multiple runs and you want to sum the multiple rusn into one.

**Two-level (folder → runs) – “2-level”**

```
/TopFolder/
  /Run_01/
    /image_00001.fits
    /image_00002.fits
    /...
    /_ShutterCount.txt
    /_Spectra.txt
  /Run_02/
  /Run_03/
```
Case 2: if you have a measurement that consists of multiple samples, and each sample consists of multiple runs, you would like to sum the runs for each of the sample.

**Three-level (folder → sample → run) – “3-level”**

```
/TopFolder/
  /Sample_A/
    /Run_01/
      /image_00001.fits
      /image_00002.fits
      /...
      /_ShutterCount.txt
      /_Spectra.txt
    /Run_02/
  /Sample_B/
    /Run_01/
    /Run_02/
```

> ❗ **Not allowed:** mixing some children with subfolders and others without (e.g., a mix of 2- and 3-level under the same parent). NEAT will stop and ask you to reorganize.

---

## 3) Load runs for summation

1. On **Summation** panel.
2. Click **Add data** and select the **TopFolder** (for 2- or 3-level).

   * NEAT scans immediate children to decide:

     * **“Detected three-level structure (folder → sample → run).”**
     * **“Detected two-level structure (folder → runs).”**
     * Or reports an error if structure is invalid (e.g., only one child or mixed).


**What NEAT checks when loading:**

* For **3-level**: each sample must have **≥ 2 run subfolders** or it is skipped with an error message.
* For **2-level**: the selected folder must have **≥ 2 subfolders (runs)**.

---

## 4) Choose where and how to save

* In **Set output**: select a **writable output folder**.
* In **Base name**: enter a short prefix (e.g., `Summed`).

  * For **3-level**: NEAT creates `<output>/Summed_<SampleName>/`.
  * For **2-level**: NEAT creates `<output>/Summed_<ParentFolder>/`.
  * Summed files use your **base name** as the stem.

---

## 5) Run the summation

1. Click **Sum**.
2. NEAT performs **lazy loading** of runs one by one:

   * **Image Loading Progress** bar reflects per-run loading.
   * **Summation Progress** shows merge & write progress.

**Strict consistency check (automatic):**

* For each run, NEAT compares the **set of image suffixes (keys)** loaded in the first run.
* If any run has a **different set/count** (e.g., missing frames), NEAT aborts and reports:

  * *“Run/Sub-folder ‘…’ has N images – expected M. Aborting summation.”*

**Merging logic (per image key):**

* For each matching suffix, NEAT **adds arrays**: `combined[suffix] += run[suffix]`.

---

## 6) Completion (and what gets written)

* On **2-level** completion: message *“Summation process (2-level) completed successfully!”*
* On **3-level** completion: for each sample, NEAT writes to:

  ```
  <output>/Summed_<SampleName>/<base_name>_*.fits
  ```

  and finally reports *“Batch summation (3-level) completed successfully!”*
* The **message pane** logs:

  * Runs loaded per sample
  * Any errors
  * Output folders created
* **Sum** is re-enabled; **Stop** is disabled.

---

## 7) Stopping a run

* Click **Stop** at any time:

  * NEAT sets a global cancel flag, asks workers to stop, and terminates loader threads.
  * **Sum** re-enabled; **Stop** disabled.
  * Message: *“Stop signal sent – aborting all processes.”*

---

## 8) Progress & messages you’ll see

* **“Detected three-level structure…” / “Detected two-level structure…”**
* **“Processing sample: … with X run(s).”**
* **“Loading run i of sample …: …”** (3-level) or **“Loading subfolder i: …”** (2-level)
* **“All runs for sample ‘…’ loaded and merged.”**
* **“All 2-level subfolders loaded and merged => SummationWorker.”**
* **“Finished summation for sample ‘…’.”**
* **Error cases:** mixed structure, single child, mismatched image counts, cannot create output folder.

---

## 9) Troubleshooting (common pitfalls)

* **Only one subfolder detected:** Add at least a second run folder or choose a different parent.
* **Mixed 2- and 3-level structure:** Move run folders so every child has the **same depth**.
* **Mismatched frame sets:** Ensure every run contains the **same set of FITS frames** (same suffix keys).
* **Output path invalid:** Use **Set output** to pick an existing, writable directory.
* **Base name empty:** Provide a short, valid string (no path separators).

---
