---
title: "NEAT: A Python Toolkit for Neutron Bragg-Edge Imaging Data Analysis"
tags:
  - Python
  - neutron imaging
  - Bragg edge
  - materials characterization
  - strain mapping
  - open-source software
authors:
  - name: Ruiyao Zhang
    orcid: 0000-0002-5557-5210
    affiliation: 1
  - name: Ranggi Ramadhan
    orcid: 0000-0003-1256-7015
    affiliation: 1
  - name: Manuel Morgano
    orcid: 0000-0001-6195-5538
    affiliation: 1
  - name: Scott Young
    orcid: 0000-0002-4478-7656
    affiliation: 1
  - name: Winfried Kockelmann
    orcid: 0000-0003-2325-5076
    affiliation: 1
  - name: Sylvia Britto
    orcid: 0000-0003-1515-307X
    affiliation: 1
affiliations:
  - name: ISIS Neutron and Muon Source, Rutherford Appleton Laboratory, Harwell, United Kingdom
    index: 1
date: 2025-12-10
bibliography: paper.bib
---

# Summary

**NEAT (Neutron Bragg-Edge Analysis Toolkit)** is an open-source Python graphical user interface (GUI) [@Zhang2025] that enables scientists to analyse neutron imaging data without advanced programming experience.  
Neutron imaging uses beams of neutrons to look inside solid objects in a non-destructive way, revealing the inner structure of samples through the neutrons’ interaction with the nuclei contained in the traversed object [@Santisteban2001]. When a neutron beam passes through a powder-like multi-crystalline solid, sharp changes—called *Bragg edges*—appear in the transmitted intensity as a function of neutron wavelength. The position and shape of these edges provide information about lattice spacing, elastic strain, phase composition, and microstructure [@Ramadhan2022].

**NEAT** offers a complete and user-friendly environment to transform raw neutron time-of-flight images acquired through spatially and temporally resolving detector such as the Berkeley-developed MCP-based detector [@Tremsin2020] in use at ISIS neutron and muon source [@Kockelmann2018] into quantitative maps of material properties. It combines data cleaning, correction, fitting, and visualisation steps into a single tool, reducing the need for multiple scripts or external software. NEAT adopts key aspects of the data-analysis workflow presented in Ramadhan’s doctoral thesis [@Ramadhan2019]. 
The program is particularly suited for experiments performed at energy-resolved neutron imaging instruments such as **IMAT (ISIS, UK)**, but can also be used with data from similar facilities worldwide like **RADEN (J-PARC, Japan)** [@Shinohara2016], and **POLDI (PSI, Swiss)** [@Polatidis2020], where the data pipeline is broadly similar given the use of temperolly resolving detector.

# Statement of Need

Bragg-edge neutron imaging (NBEI) is increasingly used to study residual stress, deformation, and phase transformation in engineering components, cultural artefacts, and advanced manufactured parts. Despite its growing importance, analysis tools remain fragmented—often requiring manual scripting for each stage of data processing.  

Existing packages such as **RITS** [@Sato2011], **TPX\_EdgeFit** [@Tremsin2016], **BEATRIX** [@Minniti2019], **BEAn** [@Liptak2019], and **iBeatles** [@Bilheux2025] address parts of the workflow, but none integrate pre-processing, fitting, and map visualisation in one open-source, cross-platform environment.

**NEAT** was developed to fill this gap by providing:
- A consolidated pipeline from raw detector frames to strain or phase maps, all within a single GUI;
- A robust analytical model and a modified three-stage optimisation strategy for stable fitting. By using a **pseudo-Voigt Bragg-edge function**, a standard in the field, it also allows users to directly compare NEAT’s output to any other similar software;
- A *pattern-fitting* mode that refines multiple edges simultaneously to obtain a global lattice parameter, analogous to Pawley refinement in diffraction;
- High-throughput mapping with **pixel-skip** and **macro-pixel** options for rapid feedback during beamtime.

Validation on benchmark iron samples at the IMAT beamline demonstrated that NEAT accurately reproduced the expected tensile–compressive strain fields and microstructural variations, while reducing analysis time from hours to minutes.  
The tool has since been applied to projects on additively manufactured superalloys, residual-stress mapping, and cultural-heritage specimens.

# Functionality Overview

- **Automated preprocessing:** run summation, pixel cleaning, overlap (pile-up) correction, and normalisation with optional binary masking.  
- **Edge fitting:** individual or multi-edge pattern fitting using pseudo-Voigt profiles to extract lattice spacing, strain, edge width, and edge height.  
- **Post-processing and visualisation:** immediate 2-D maps and line-profiles of fitted parameters for interpretation and publication.

Detailed explanations and instructions for each function can be found in the **user manual** (https://github.com/RayZhang2024/NEAT/blob/main/User%20manual.md)

# Example of Use

To demonstrate NEAT’s workflow, we analysed Bragg-edge imaging data collected from a U-shaped iron specimen measured at the IMAT beamline (ISIS, UK). The sample was bent to U-shape from a straight bar, leaving residual stress / strain in the bend [@Haribabu2024]. 

The raw time-of-flight image stacks were loaded into NEAT, where the runs were automatically summed, cleaned, and normalised using the built-in preprocessing pipeline.  

After selecting the Fe-bcc phase, a macro-pixel region of 20 x 20 pixels was defined to extract the transmission spectrum, and the 110, 200, and 211 Bragg edges were fitted using the multi-edge (“pattern”) modes with a pseudo-Voigt profile.  

The automated batch-fitting routine then produced two-dimensional maps of lattice spacing, Bragg-edge width, and edge height across the sample within minutes.

The resulting lattice-parameter map revealed tensile regions (red) along the inner bend and compressive regions (blue) along the outer bend, consistent with the expected plastic bending behaviour (*Figure 1*). In addition to lattice spacing mapping, NEAT quantifies Bragg-edge width and height, which provide complementary information on microstructural evolution due to plastic deformation induced by bending. These results have been validated through the use of other similar software, FEA modelling and have been replicated at other facilities as part of a round-robin measurement campaign to compare the performances of various neutron instruments.

![Example](https://github.com/RayZhang2024/NEAT/blob/main/docs/paper/Example.png)
**Figure 1:** From left to right, fitted lattice parameter, 110 edge width and 110 edge height of the U-shape bent sample.

# Acknowledgements

This work was supported by the **Engineering and Imaging Group** at the **ISIS Neutron and Muon Source**, Science and Technology Facilities Council (STFC), United Kingdom.  
The authors thank **Computing Division** at ISIS Neutron and Muon Source and **Scientific Computing Department** of STFC for their technical support, and the **IMAT user community** for their valuable feedback during development and testing.

# References

