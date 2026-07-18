# NEAT Technical Reference

This directory is the code-traceable technical reference for NEAT. It explains
what the application currently does, how data move through each functional
block, which assumptions are enforced, and where scientific judgement remains
the responsibility of the user.

The technical reference complements rather than replaces:

- the user manual, which explains how to operate NEAT;
- the assistant FAQ, which gives concise support answers;
- the troubleshooting guide, which maps symptoms to corrective actions; and
- scientific publications, which justify methods and support interpretation.

## Authority and review labels

Every technical document must declare one of these statuses:

| Status | Meaning |
|---|---|
| `planned` | Scope and source files are identified, but content is not drafted. |
| `draft` | Content has been derived from code but has not completed verification. |
| `code-verified` | Executable logic, defaults, validation and outputs have been checked against the listed source revision. |
| `domain-reviewed` | Scientific purpose, assumptions and interpretation have been reviewed by a suitable domain expert. |
| `approved-for-rag` | The document is code-verified, sufficiently reviewed and safe for retrieval by the NEAT assistant. |

An implementation detail can be `code-verified` without being
`domain-reviewed`. Scientific claims must not be presented as validated merely
because they appear in source code or comments.

## Documentation rules

1. Executable behavior and tests take precedence over comments.
2. A suspected defect is labelled as an observed implementation issue, not
   described as intended behavior.
3. Defaults, units, coordinate conventions and hard-coded constants are stated
   explicitly.
4. Instrument-specific behavior is separated from general behavior.
5. Every input, validation branch, output and user-visible failure mode is
   recorded.
6. Source files, key functions/classes and relevant tests are listed.
7. The NEAT version and verified Git revision are recorded.
8. Scientific limitations and required expert decisions are clearly bounded.
9. Documents remain modular and heading-rich so retrieval returns one coherent
   technical topic rather than an entire manual.

## Build documents

- [Documentation build plan](plan.md)
- [Functional coverage matrix](coverage_matrix.md)
- [Technical-document template](_template.md)

## Available code-verified drafts

### Batch 1: data conventions and preprocessing

- [Supported preprocessing images and sidecars](data/supported_images_and_sidecars.md)
- [Preprocessing folder layouts](data/folder_layouts.md)
- [Summation](preprocessing/summation.md)
- [Clean](preprocessing/clean.md)
- [Overlap correction](preprocessing/overlap_correction.md)
- [Classic FITS/TIFF normalisation](preprocessing/normalisation_fits.md)
- [RADEN TIFF normalisation](preprocessing/normalisation_raden.md)
- [Filtering and masks](preprocessing/filtering_masks.md)
- [Full Process](preprocessing/full_process.md)
- [Batch 1 domain-review checklist](batch1_review_checklist.md)

These pages have been checked against the named code revision. Scientific
review and RAG approval remain pending where shown in each page.

### Batch 2: loading, wavelength and phase configuration

- [Classic FITS/TIFF image-folder loading](loading/fits_stacks_orientation.md)
- [NeXus image stacks and geometry](loading/nexus_stacks_geometry.md)
- [RADEN TIFF stacks and metadata](loading/raden_tiff_metadata.md)
- [Intensity-profile import](loading/intensity_profile_import.md)
- [ToF-to-wavelength conversion](wavelength/tof_conversion.md)
- [Flight path and time delay](wavelength/flight_path_delay.md)
- [Manual wavelength and ToF anchors](wavelength/manual_anchors.md)
- [Phase definitions and theoretical edges](phase/phase_definitions_edges.md)
- [Batch 2 domain-review checklist](batch2_review_checklist.md)

### Batch 3: spectrum extraction and fitting

- [Spectrum extraction and ROI averaging](fitting/spectrum_roi.md)
- [Edge Table and fitting windows](fitting/edge_table_windows.md)
- [Pseudo-Voigt/exponential-tail model](fitting/pseudo_voigt_model.md)
- [Individual-edge fitting](fitting/individual_edge_fitting.md)
- [Multi-edge pattern fitting](fitting/pattern_fitting.md)
- [Bounds, fixed parameters and diagnostics](fitting/bounds_diagnostics.md)
- [Fitting uncertainty and planning estimator](fitting/uncertainty.md)
- [Batch 3 domain-review checklist](batch3_review_checklist.md)

### Batch 4: mapping and result files

- [Mapping geometry](mapping/mapping_geometry.md)
- [Individual-edge batch worker](mapping/individual_edge_worker.md)
- [Pattern batch worker](mapping/pattern_worker.md)
- [Interpolation and missing values](mapping/interpolation_missing.md)
- [Result CSV schemas](mapping/result_csv_schema.md)
- [Batch 4 domain-review checklist](batch4_review_checklist.md)

### Batch 5: post-processing and visualisation

- [CSV ingestion and metric discovery](postprocessing/csv_ingestion_metrics.md)
- [Parameter maps, masks and FITS export](postprocessing/maps_masks_export.md)
- [Coordinates and display units](postprocessing/coordinates_units.md)
- [Strain calculation](postprocessing/strain.md)
- [ROI statistics and line profiles](postprocessing/roi_line_profiles.md)
- [Batch 5 domain-review checklist](batch5_review_checklist.md)

### Batch 6: architecture and operational support

- [Application startup, window composition and persistent state](architecture/startup_state.md)
- [Worker ownership, progress, cancellation and shutdown](architecture/workers_shutdown.md)
- [Updates, packaging and operational errors](support/updates_packaging_errors.md)
- [AI assistant retrieval, privacy, feedback and evaluation](support/assistant_architecture.md)
- [Batch 6 architecture and support review checklist](batch6_review_checklist.md)

## Planned reference structure

```text
technical/
├── architecture/
├── data/
├── preprocessing/
├── fitting/
├── mapping/
├── postprocessing/
└── support/
```

Documents are added to assistant retrieval only after reaching
`approved-for-rag`. The coverage matrix is the authoritative progress tracker.
