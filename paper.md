---

title: "NEAT: A Python Toolkit for Neutron Bragg-Edge Imaging Data Analysis"

tags:

&nbsp; - Python

&nbsp; - neutron imaging

&nbsp; - Bragg edge

&nbsp; - materials characterization

&nbsp; - strain mapping

&nbsp; - open-source software

authors:

  - name: Ruiyao Zhang

    orcid: 0000-0002-5557-5210

    affiliation: 1

  - name: Ranggi Ramadhan,

    affiliation: 1

  - name: Manuel Morgano,

    affiliation: 1

  - name: Scott Young,

    affiliation: 1

  - name: Winfried Kockelmann,

    affiliation: 1

  - name: Sylvia Britto,

    affiliation: 1

affiliations:

&nbsp; - name: ISIS Neutron and Muon Source, Rutherford Appleton Laboratory, Harwell, United Kingdom

&nbsp;   index: 1

date: 2025-11-03

bibliography: paper.bib

---



\# Summary



\*\*NEAT (Neutron Bragg-Edge Analysis Toolkit)\*\* is an open-source Python graphical user interface (GUI) that enables scientists to analyse neutron imaging data without advanced programming experience.  

Neutron imaging uses beams of neutrons to look inside solid objects in a non-destructive way, revealing how atoms are arranged and how materials deform under stress. When a neutron beam passes through a crystalline solid, sharp changes—called \*Bragg edges\*—appear in the transmitted intensity as a function of neutron wavelength. The position and shape of these edges provide information about lattice spacing, elastic strain, phase composition, and microstructure.



\*\*NEAT\*\* offers a complete and user-friendly environment to transform raw neutron time-of-flight images into quantitative maps of material properties. It combines data correction, normalisation, fitting, and visualisation steps into a single tool, reducing the need for multiple scripts or external software.  

The program is particularly suited for experiments performed at energy-resolved neutron imaging instruments such as \*\*IMAT (ISIS, UK)\*\* and \*\*ENGIN-X (ISIS, UK)\*\*, but can also be used with data from similar facilities worldwide like \*\*RADEN (J-PARC, Japan)\*\*, and \*\*POLDI (PSI, Swiss)\*\*.



\# Statement of Need



Bragg-edge neutron imaging (NBEI) is increasingly used to study residual stress, deformation, and phase transformation in engineering components and advanced manufactured parts. Despite its growing importance, analysis tools remain fragmented—often requiring manual scripting for each stage of data processing.  



Existing packages such as \*\*RITS\*\* (Sato et al., 2011), \*\*TPX\_EdgeFit\*\* (Tremsin 2016), \*\*BEATRIX\*\* (Minniti et al., 2019), and \*\*iBeatles\*\* (Bilheux et al., 2018) address parts of the workflow, but none integrate pre-processing, fitting, and map visualisation in one open-source, cross-platform environment.



\*\*NEAT\*\* was developed to fill this gap by providing:

\- A consolidated pipeline from raw detector frames to strain or phase maps, all within a single GUI;

\- A robust analytical model using a \*\*pseudo-Voigt Bragg-edge function\*\* and a modified three-stage optimisation strategy for stable fitting;

\- A \*pattern-fitting\* mode that refines multiple edges simultaneously to obtain a global lattice parameter, analogous to Pawley refinement in diffraction;

\- High-throughput mapping with \*\*pixel-skip\*\* and \*\*macro-pixel\*\* options for rapid feedback during beamtime.



Validation on benchmark iron samples at the IMAT beamline demonstrated that NEAT accurately reproduced the expected tensile–compressive strain fields and microstructural variations, while reducing analysis time from hours to minutes.  



\# Functionality Overview



\- \*\*Automated preprocessing:\*\* run summation, pixel cleaning, overlap (pile-up) correction, and normalisation with optional binary masking.  

\- \*\*Edge fitting:\*\* individual or multi-edge pattern fitting using pseudo-Voigt profiles to extract lattice spacing, strain, edge width, and edge height.  

\- \*\*Post-processing and visualisation:\*\* immediate 2-D maps and line-profiles of fitted parameters for interpretation and publication.  

\- \*\*Open-source and extensible:\*\* written in Python 3 (NumPy, SciPy, Matplotlib, PyQt5, Pandas) and released under the MIT License.  



\# Example of Use



To demonstrate NEAT’s workflow, we analysed Bragg-edge imaging data collected from U-shaped iron specimens measured at the IMAT beamline (ISIS, UK).  

The raw time-of-flight image stacks were loaded into NEAT, where the runs were automatically summed, cleaned, and normalised using the built-in preprocessing pipeline.  

After selecting the Fe-bcc phase, a macro-pixel region was defined to extract the transmission spectrum, and the 110, 200, and 211 Bragg edges were fitted using both single-edge and multi-edge (“pattern”) modes with a pseudo-Voigt profile.  

The automated batch-fitting routine then produced two-dimensional maps of lattice spacing, Bragg-edge width, and edge height across the sample within minutes.



The resulting strain map revealed tensile regions on the inner bend and compressive regions on the outer bend—matching the expected elastic bending behaviour.  

Maps of edge width and height highlighted microstructural changes associated with plastic deformation, illustrating NEAT’s ability to quantify both strain and microstructure directly from imaging data.





\# Acknowledgements



This work was supported by the \*\*Engineering and Imaging Group\*\* at the \*\*ISIS Neutron and Muon Source\*\*, Science and Technology Facilities Council (STFC), United Kingdom.  

The authors thank the IMAT user community for their valuable feedback during development and testing.



\# References







