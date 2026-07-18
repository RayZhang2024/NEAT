---
title: Post-processing CSV ingestion and metric discovery
doc_id: neat-tech-postprocessing-csv
doc_type: technical_reference
functional_area: postprocessing
audience: [user, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/mixins/postprocessing.py]
source_symbols: [PostProcessingMixin.load_csv_file, PostProcessingMixin.create_parameter_buttons, PostProcessingMixin.plot_parameter]
test_paths: []
---

# Post-processing CSV ingestion and metric discovery

The loader requires the first line to be exactly:

```text
Metadata Name,Metadata Value
```

Each following nonblank line is split once at the first comma into a metadata
key/value. The blank line terminates metadata; the remainder is read by pandas
as the data table.

Every column except exact lowercase `x` and `y` becomes a parameter button.
Plotting nevertheless requires both `x` and `y`. Coordinates are sorted into
unique axes and values fill `Z[y-index,x-index]`; duplicate coordinate rows
overwrite earlier values with the last row encountered.

No schema version, numeric dtype, coordinate completeness or unit validation is
performed. Sparse coordinates remain NaN. Any compatible third-party file can
load if it follows this envelope, but parameter meaning is not verified.

## Retrieval questions

- What header and blank-line structure must a result CSV use?
- How are parameter buttons created?
- What happens with duplicate x/y rows?
- Why did a CSV load but fail when plotting a parameter?

