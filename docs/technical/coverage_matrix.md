# NEAT Technical Documentation Coverage Matrix

Status values follow [README.md](README.md). Priority `P0` is foundational or
safety-critical, `P1` is core behavior, and `P2` is supporting behavior.

| Batch | Area | Planned document | Primary implementation | Current test evidence | Domain review | Priority | Status |
|---:|---|---|---|---|---|---:|---|
| 1 | Data | [Supported images and sidecars](data/supported_images_and_sidecars.md) | `workers/batch.py`, preprocessing workers | Image orientation and TIFF-loading tests | Instrument applicability | P0 | code-verified |
| 1 | Data | [Folder-layout detection](data/folder_layouts.md) | `services/preprocessing_layout.py`, preprocessing UI/worker call sites | Headless layout/order and call-site wiring regressions (`tests/test_preprocessing_layout.py`) | No | P0 | code-verified |
| 1 | Preprocessing | [Summation](preprocessing/summation.md) | `SummationWorker`, summation UI orchestration | Focused image/sidecar/pre-write validation tests | Complete | P0 | domain-reviewed |
| 1 | Preprocessing | [Outlier removal/Clean](preprocessing/clean.md) | `OutlierFilteringWorker` | Focused 5x5/7x7 replacement, spike and report tests | Complete | P1 | domain-reviewed |
| 1 | Preprocessing | [Overlap/pile-up correction](preprocessing/overlap_correction.md) | `OverlapCorrectionWorker` | Focused equation and shape-rejection tests | Required | P0 | code-verified |
| 1 | Preprocessing | [FITS normalisation](preprocessing/normalisation_fits.md) | `NormalisationWorker` | Focused local division and shutter-scale test | Required | P0 | code-verified |
| 1 | Preprocessing | [RADEN TIFF normalisation](preprocessing/normalisation_raden.md) | `RadenNormalisationWorker` | Headless TIFF normalisation test | Complete | P0 | domain-reviewed |
| 1 | Preprocessing | [Filtering and masks](preprocessing/filtering_masks.md) | `FilteringWorker`, `MaskGeneratorDialog` | Focused binary-mask, shape and completion-state tests | Complete | P1 | domain-reviewed |
| 1 | Preprocessing | [Full Process](preprocessing/full_process.md) | `FullProcessWorker` | TIFF loading helper test only | Required | P1 | code-verified |
| 2 | Loading | [FITS stacks and orientation](loading/fits_stacks_orientation.md) | `ImageLoadWorker`, image I/O helpers | FITS orientation round-trip tests | Instrument orientation | P0 | code-verified |
| 2 | Loading | [NeXus stacks and geometry](loading/nexus_stacks_geometry.md) | NeXus helpers and worker | Geometry/TOF headless tests | Instrument review | P0 | code-verified |
| 2 | Loading | [RADEN TIFF and metadata](loading/raden_tiff_metadata.md) | RADEN parsing/loading helpers | RADEN parser and loading tests | Instrument review | P0 | code-verified |
| 2 | Wavelength | [TOF-to-wavelength conversion](wavelength/tof_conversion.md) | axis conversion helpers | NeXus/RADEN conversion tests | Required | P0 | code-verified |
| 2 | Wavelength | [Flight path and time delay](wavelength/flight_path_delay.md) | fitting settings and recalculation | Flight-path recalculation test | Required | P0 | code-verified |
| 2 | Wavelength | [Manual wavelength anchors](wavelength/manual_anchors.md) | fitting configuration methods | Focused interpolation and ToF-anchor tests | Required | P0 | code-verified |
| 2 | Loading | [Intensity-profile import](loading/intensity_profile_import.md) | fitting profile readers | CSV/TXT/XLSX/NeXus reader tests | Input units | P1 | code-verified |
| 2 | Phase | [Phase definitions and theoretical edges](phase/phase_definitions_edges.md) | fitting phase UI, core Bragg helpers | Core Bragg-edge tests | Required | P1 | code-verified |
| 3 | Spectrum | [Display, ROI and macro-pixel extraction](fitting/spectrum_roi.md) | fitting UI selection methods | ROI interaction tests | Sampling guidance | P1 | code-verified |
| 3 | Fitting | [Edge Table and windows](fitting/edge_table_windows.md) | fitting table methods | Default-window and derived-window tests | Required | P0 | code-verified |
| 3 | Fitting | [Pseudo-Voigt model](fitting/pseudo_voigt_model.md) | fitting functions and core helpers | Headless numerical fitting tests | Required | P0 | code-verified |
| 3 | Fitting | [Individual-edge fitting](fitting/individual_edge_fitting.md) | `FittingEngine.fit_individual_edge`, `fit_region` adapter | Known/unknown golden and staged-failure tests | Required | P0 | code-verified |
| 3 | Fitting | [Pattern fitting](fitting/pattern_fitting.md) | typed domain models, `FittingEngine.fit_full_pattern`, legacy GUI adapter | Domain ownership, typed engine, golden and adapter regressions | Required | P0 | code-verified |
| 3 | Fitting | [Bounds, fixed parameters and diagnostics](fitting/bounds_diagnostics.md) | fitting UI/core | Fitting and residual helper tests | Required | P0 | code-verified |
| 3 | Fitting | [Uncertainty estimation](fitting/uncertainty.md) | `core/fitting.py`, estimator dialog | Core estimator and fitting tests | Required | P0 | code-verified |
| 4 | Mapping | [Mapping ROI, boxes and steps](mapping/mapping_geometry.md) | fitting batch controls | ROI geometry and fitting tests | Sampling guidance | P1 | code-verified |
| 4 | Mapping | [Individual-edge batch worker](mapping/individual_edge_worker.md) | `BatchFitEdgesWorker`, injected `FittingEngine` | Worker output, cancellation, ordering and CSV regressions | Required | P0 | code-verified |
| 4 | Mapping | [Pattern batch worker](mapping/pattern_worker.md) | `BatchFitWorker`, typed `FittingEngine` contract | Worker-level engine, failure, cancellation, persistence and signal tests | Required | P0 | code-verified |
| 4 | Mapping | [Interpolation and missing values](mapping/interpolation_missing.md) | batch worker interpolation | Focused individual interpolation test | Interpretation limits | P0 | code-verified |
| 4 | Output | [Result CSV schemas and metadata](mapping/result_csv_schema.md) | batch CSV writers | Focused schema snapshot test | Parameter meaning | P0 | code-verified |
| 5 | Post-processing | [CSV ingestion and metric buttons](postprocessing/csv_ingestion_metrics.md) | `PostProcessingMixin` | No focused parser test | Parameter meaning | P0 | code-verified |
| 5 | Post-processing | [Parameter maps, masks and export](postprocessing/maps_masks_export.md) | `ParameterPlotDialog` | Focused resize/helper tests | Export semantics | P1 | code-verified |
| 5 | Post-processing | [Coordinates and display units](postprocessing/coordinates_units.md) | `ParameterPlotDialog` | Focused cell-edge test | Detector applicability | P0 | code-verified |
| 5 | Post-processing | [Strain from reference spacing](postprocessing/strain.md) | `calculate_strain` | Formula documented; no isolated UI test | Required | P0 | code-verified |
| 5 | Post-processing | [ROI mean and line profiles](postprocessing/roi_line_profiles.md) | plot/line-profile dialogs | Focused line-interpolation test | Interpretation limits | P1 | code-verified |
| 6 | Architecture | [Structured preprocessing operation results](architecture/preprocessing_contracts.md) | `NEAT/domain/preprocessing.py` | Contract-level coverage only (`tests/test_preprocessing_domain.py`); Qt workers are not migrated | Not required | P1 | code-verified |
| 6 | Architecture | [Typed loaded-image run contract](architecture/preprocessing_contracts.md) | `NEAT/domain/preprocessing_inputs.py` | Contract-level coverage only (`tests/test_preprocessing_inputs.py`); existing loaders/workers remain dictionary-based | Not required | P1 | code-verified |
| 6 | Architecture | [Startup, window composition and state](architecture/startup_state.md) | `app.py`, `main_window.py` | Import and partial GUI tests | Not required | P2 | code-verified |
| 6 | Architecture | [Workers, progress and shutdown](architecture/workers_shutdown.md) | UI mixins and workers | Partial GUI/worker tests; no consolidated shutdown test | Not required | P1 | code-verified |
| 6 | Support | [Errors, updates and packaging](support/updates_packaging_errors.md) | main window, packaging config | Assistant error tests; no packaged smoke test | Security review | P2 | code-verified |
| 6 | Assistant | [RAG, privacy, feedback and evaluation](support/assistant_architecture.md) | assistant UI/tools | Extensive assistant tests | Safety review | P1 | code-verified |

## Initial test-risk observations

- Batch 1 now has focused worker tests for Summation, Clean, overlap
  correction, standard FITS normalisation and Filtering, plus headless
  directory-layout classification and call-site wiring tests. Complete
  Full Process orchestration and interactive mask editing still lack focused
  coverage.
- Fitting and input loading have materially better headless coverage, but the
  full GUI-to-worker-to-output path is not comprehensively tested.
- Documents for untested behavior can still reach `code-verified`, but the
  missing-test limitation must remain visible and should generate a test
  backlog item.
