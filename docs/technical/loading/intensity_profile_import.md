---
title: Intensity-profile import
doc_id: neat-tech-loading-profile
doc_type: technical_reference
functional_area: loading
audience: [user, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py]
source_symbols: [FittingMixin._read_intensity_profile_file, FittingMixin._read_nexus_intensity_profile_file, FittingMixin._clean_profile_arrays, FittingMixin.import_intensity_profile]
test_paths: [tests/test_fitting_headless.py]
---

# Intensity-profile import

## Supported inputs

The UI accepts CSV, TXT, XLSX and `.nxs`; the reader also recognizes `.h5` and
`.hdf5`. Legacy `.xls` is explicitly rejected. Tabular files use their first
two columns as wavelength and intensity. Headers and nonnumeric rows are
coerced away. At least three finite pairs must remain, and rows are sorted by
wavelength. Duplicate wavelengths are retained.

For NeXus/HDF5, NEAT gathers numeric one-dimensional datasets with at least
three entries. Labels, paths, units and monotonicity are scored to select a
wavelength-like array and same-shaped intensity-like array.

## State change

Importing a profile clears image-stack state, makes the profile the current ROI
spectrum, updates wavelength bounds and Edge Table, and disables mapping.
Profile import supports individual and pattern fitting of the one spectrum but
cannot produce a spatial map.

## Limitations

- The first two spreadsheet/text columns are assumed; units are not validated.
- HDF5 selection is heuristic.
- Duplicate or nonmonotonic input is sorted but not averaged.
- Imported values are assumed to already represent the desired intensity or
  transmission quantity.

## Retrieval questions

- Which profile formats can NEAT import?
- What columns and minimum number of rows are required?
- How does NEAT choose arrays from a NeXus profile?
- Why is mapping disabled after profile import?

