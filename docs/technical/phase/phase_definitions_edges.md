---
title: Phase definitions and theoretical Bragg edges
doc_id: neat-tech-phase-definitions
doc_type: technical_reference
functional_area: phase
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: completed 2026-07-16
source_paths: [NEAT/core/fitting.py, NEAT/ui/mixins/fitting.py, NEAT/ui/main_window.py]
source_symbols: [PHASE_DATA, calculate_d_spacing_general, calculate_theoretical_bragg_edges, calculate_x_hkl_general, FittingMixin.open_add_phase_dialog]
test_paths: [tests/test_core_bragg_edges.py]
---

# Phase definitions and theoretical Bragg edges

Each phase contains a structure name, lattice parameters in ångströms and an
integer `(h,k,l)` list. Supported structures and required values are:

- cubic/fcc/bcc: `a`
- tetragonal/hexagonal: `a`, `c`
- orthorhombic: `a`, `b`, `c`

The code calculates the structure-specific `d_hkl` and then:

```text
theoretical edge wavelength = 2 × d_hkl
```

Invalid/missing parameters yield NaN. Reflection-selection rules are not
generated from the structure; the supplied hkl list determines which edges
appear. FCC and BCC built-ins use fixed default lists.

Built-in phases include Unknown, Cu_fcc, Fe_bcc, Fe_fcc, Al, Ni_gamma, CeO2,
Ti_Beta and Ti_alpha_hex. Custom phases can be created, persisted in
`~/.neat_custom_phases.json`, and removed through Phase Management. A custom
phase cannot overwrite a built-in name.

## Corrected copper and beta-titanium definitions

`Cu_fcc` is face-centred cubic with `a = 3.615 Å`. `Ti_Beta` is
body-centred cubic with `a = 3.32 Å`; it therefore requires only the cubic
`a` parameter. Its explicit reflection list contains the BCC-allowed
reflections 110, 200, 211, 220, 310 and 222.

These values are reference values. In particular, the beta-titanium lattice
parameter is temperature- and composition-dependent, so a user analysing a
specific material should create a custom phase with the appropriate reference
lattice parameter when necessary.

`Unknown_Phase` has no structure, lattice parameters or hkl list and follows a
separate fitting behavior described in Batch 3.

## Retrieval questions

- How does NEAT calculate a theoretical Bragg-edge wavelength?
- Which crystal structures can a custom phase use?
- Does NEAT automatically generate allowed reflections?
- Where are custom phase definitions stored?
