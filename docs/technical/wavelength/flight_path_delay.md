---
title: Flight path and time delay
doc_id: neat-tech-wavelength-instrument-settings
doc_type: technical_reference
functional_area: wavelength
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py, NEAT/workers/batch.py]
source_symbols: [FittingMixin.change_flight_path, FittingMixin.set_delay, FittingMixin._apply_instrument_settings_to_loaded_data, FittingMixin._choose_nexus_flight_path]
test_paths: [tests/test_fitting_headless.py]
---

# Flight path and time delay

Flight path is stored in metres and delay in milliseconds. Instrument settings
allow flight path and delay to be changed and persisted. Recalculation occurs
for classic spectra, manual ToF anchors, RADEN and ToF-based NeXus axes. Imported
intensity profiles and wavelength-valued NeXus axes are not recalculated.

When NeXus geometry is available, the UI lets the user choose file-derived or
app flight path. The chosen source is recorded in state. RADEN uses the app
setting.

Delay is added to ToF before conversion:

```text
adjusted ToF = measured ToF + delay
```

The delay dialog allows `-1.0` to `1.0` ms with three decimals; the combined
instrument dialog uses its own controls. Changing delay through the standalone
dialog recalculates only when the classic `tof_array` exists, whereas applying
instrument settings uses the broader recalculation dispatcher.

Scientific review is required for path definition, delay sign and valid ranges.

## Retrieval questions

- Which loaded data change when flight path is edited?
- Can I choose NeXus geometry instead of the app flight path?
- What unit and sign does time delay use?
- Why did a wavelength-valued NeXus file not change?

