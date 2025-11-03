
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
You can download the Windows standalone executable from the [NEAT v4.5 Release](https://github.com/RayZhang2024/NEAT/releases/download/v4.5/NEAT.exe).
No installation is needed, just run the executable by doule clicking.

### Pull from the repository

### 1️⃣ Clone the repository
Open a terminal (Git Bash, PowerShell) and run:
```bash
git clone https://github.com/RayZhang2024/NEAT.git
cd ~/NEAT
````

---

### 2️⃣ Install NEAT via `pip`

Because this project includes a `pyproject.toml`, you can install it directly:

```bash
pip install .
```

or

```bash
python -m pip install .
```

This automatically installs NEAT **and all dependencies** specified in `pyproject.toml`.


---

### 🧠 Running NEAT

After installation, simply launch the GUI with:

```bash
cd ~/NEAT/NEAT
python NEAT.py
```

✅ The main window titled
**“NEAT Neutron Bragg Edge Analysis Toolkit v4_beta”**
will appear, with tabs for:

* **Data Preprocessing**
* **Bragg Edge Fitting**
* **Data Post-Processing**
* **About**

---

## 🧑‍💻 Citation / Acknowledgment

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
