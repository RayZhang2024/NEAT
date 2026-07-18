---
title: Mapping result CSV schemas and metadata
doc_id: neat-tech-mapping-result-csv
doc_type: technical_reference
functional_area: output
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [image stacks]
scientific_review: pending
source_paths: [NEAT/workers/batch.py]
source_symbols: [BatchFitEdgesWorker.save_results_to_csv_ungrid, BatchFitEdgesWorker.save_results_to_csv, BatchFitWorker.save_results_to_csv_ungrid, BatchFitWorker.save_results_to_csv]
test_paths: [tests/test_batch_mapping_outputs.py]
---

# Mapping result CSV schemas and metadata

Files begin with:

```text
Metadata Name,Metadata Value
key,value
...

x,y,<result columns...>
```

Metadata includes box/step sizes, half-open ROI bounds, interpolation and fix
flags, directory, flight path/source, data source, input file, wavelength
range, selected phase, edge count and serialized Edge Table rows.

Timestamped names are:

- individual: `results_edges_ungridded_YYYYMMDD_HHMMSS.csv` and
  `results_edges_gridded_...csv`
- pattern: `results_ungridded_...csv` and `results_gridded_...csv`

Every file has `image_height × image_width` rows, including coordinates outside
the mapping ROI. Those values are normally NaN.

Individual columns per edge are `d`, `s`, `t`, `eta`, `fwhm`, `height` and
uncertainties for `d/s/t/eta`. Pattern output first includes each lattice
parameter and `<parameter>_unc`, then per-edge `s/t/eta/fwhm/height` and shape
uncertainties.

Column meanings and units are:

- `x`, `y`: zero-based detector-pixel column and row indices
- `d` and fitted lattice parameters: ångströms
- `s`, `t` and `fwhm`: ångströms
- `eta`: dimensionless neutron-pulse edge-shape parameter
- `height`: fitted intensity/transmission difference, in the same units as the
  fitted spectrum
- each `_unc` column: standard error in the same unit as its parameter; NaN
  means unavailable or not estimated

All individual and pattern result writers use all three HKL indices in column
suffixes. For example, `(1,1,0)` is consistently written as `d_110`, `s_110`,
`t_110`, `eta_110` and the corresponding result/uncertainty columns.

Each successful CSV writer returns its absolute saved pathname. On successful
completion, the worker emits the actual ungridded and gridded output paths as
a newline-separated string for display by the GUI.

## Retrieval questions

- Why does an ungridded CSV contain every detector pixel?
- What metadata are saved before the data table?
- How are per-edge columns named?
- What is the difference between gridded and ungridded files?
