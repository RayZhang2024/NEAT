---
title: Strain calculation from reference spacing
doc_id: neat-tech-postprocessing-strain
doc_type: technical_reference
functional_area: postprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/dialogs.py]
source_symbols: [ParameterPlotDialog.calculate_strain]
test_paths: []
---

# Strain calculation from reference spacing

For the currently displayed map `Z` and a user-entered positive `d0`, NEAT
computes:

```text
microstrain = (Z / d0 - 1) × 1,000,000
```

NaNs propagate. A new dialog is opened with parameter name `Strain`; source
metadata is copied unchanged without adding `d0` or the formula.

The calculation is enabled only for fitted d-spacing metrics whose column name
starts with `d_`; uncertainty columns such as `d_unc_110` are excluded.
Filtered d-spacing maps remain eligible. A runtime guard prevents strain from
being calculated from `s`, `t`, height, FWHM, an already-strain map or an
arbitrary image.

This is lattice strain relative to the supplied reference spacing, not stress.
Positive values mean the measured spacing is larger than `d0`; negative values
mean it is smaller. Stress must not be inferred without the required elastic
constants and measurement geometry. Scientific review must still define
selection/provenance of `d0`, its units, phase/hkl matching, and
temperature/composition assumptions.

## Retrieval questions

- What strain equation does NEAT use?
- What units does the strain map have?
- Can Calculate Strain be used on any parameter button?
- How should I choose a valid d0?
