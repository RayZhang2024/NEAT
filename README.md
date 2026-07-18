
# 🧠 NEAT – Neutron Bragg Edge Analysis Toolkit

**Author:** Ruiyao Zhang, Ranggi Ramadhan  
**Affiliation:** ISIS Neutron and Muon Source, STFC Rutherford Appleton Laboratory  
**Email:** ruiyao.zhang@stfc.ac.uk  

---

## 📘 Overview

**NEAT (Neutron Bragg Edge Analysis Toolkit)** is an integrated graphical user interface (GUI) for quantitative Bragg-edge imaging analysis.  
It provides a streamlined workflow for:
- ✅ Image preprocessing (summation, scaling, normalisation)  
- ✅ Bragg-edge fitting (pattern and single-edge fitting)  
- ✅ Data post-processing and visualisation  

Developed at ISIS Neutron and Muon Source, NEAT is designed for use with IMAT and other neutron imaging instruments.

---

## ⚙️ Requirements

- **Python** ≥ 3.9  
- Supported platforms: **Windows**

All required packages (PyQt5, matplotlib, numpy, scipy, astropy, pandas, psutil, etc.) will be installed automatically.

---

## 🚀 Run the GUI

### Standalone executable
You can download the Windows standalone executable from the [NEAT v4.8.1 Release](https://github.com/RayZhang2024/NEAT/releases/download/v4.8.1/NEATv4.8.1.zip).
No installation is needed; extract the ZIP and double-click `NEAT.exe`.

### 🚀 Example data
An example dataset is available for Bragg edge fitting tutorial, click to download [Example_dataset](https://github.com/RayZhang2024/NEAT/releases/download/v4.6/5_Ubend_normalised.zip). The dataset has been pre-processed and is ready for Bragg edge fitting, go and have a try!

### If you would like to run the script:

### 1️⃣ Clone the repository
Open a terminal (Git Bash, PowerShell) and run:
```bash
git clone https://github.com/RayZhang2024/NEAT.git
cd ~/NEAT
````

---

### 2️⃣ Install NEAT via `pip`

From the repository root (the folder containing `pyproject.toml`), install into a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install ".[assistant]"
```

This installs NEAT together with the optional AI-assistant dependencies. Use
`python -m pip install .` only when the assistant is not required.


---

### 🧠 Running NEAT

With the virtual environment activated, launch the GUI with the installed console script:

```bash
NEAT
```

If you prefer to run straight from the source checkout, execute:

```bash
python -m NEAT.app
```

✅ The main window titled
**“NEAT Neutron Bragg Edge Analysis Toolkit v4.8.1”**
will appear, with tabs for:

* **Data Preprocessing**
* **Bragg Edge Fitting**
* **Data Post-Processing**

### AI Assistant

NEAT 4.8 adds a dockable, retrieval-grounded assistant for questions about the
software, its controls and documented technical behaviour. Open **AI Assistant
> AI Assistant** to show or hide the panel and **AI Assistant > AI Assistant
Settings...** to select access and model settings.

The assistant supports personal API keys for OpenAI, Anthropic, Google Gemini,
DeepSeek and Kimi/Moonshot; manually configured OpenAI-compatible endpoints;
and local models through Ollama or LM Studio. API keys are stored in the
operating-system credential manager and are not saved in the repository or
ordinary NEAT settings. Official standalone downloads include automatic shared
NEAT access with a limited daily request allowance.

Questions, approved NEAT documentation excerpts, recent chat turns and a small
allow-listed set of screen settings may be sent to the selected model. Raw
images and automatically collected file paths are not sent. When the semantic
search runtime is unavailable, NEAT falls back to its bundled lexical search so
the approved knowledge base remains usable.

---

## 🧑‍💻 Citation / Acknowledgment

Please cite the following reference if you use NEAT for your data analysis:
RayZhang2024. (2025). RayZhang2024/NEAT: NEAT v4.6 (v4.6). Zenodo. https://doi.org/10.5281/zenodo.17512269

---

## 📜 License

MIT License © 2025 **Ruiyao Zhang**
You are free to use, modify, and redistribute with attribution.

---

## 📧 Contact

For questions, bug reports, or collaborations:

* **Email:** [ruiyao.zhang@stfc.ac.uk](mailto:ruiyao.zhang@stfc.ac.uk)
* **GitHub:** [RayZhang2024](https://github.com/RayZhang2024)

```
