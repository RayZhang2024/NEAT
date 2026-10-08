# NEAT Technical Documentation Coverage Matrix

Status values follow [README.md](README.md). Priority `P0` is foundational or
safety-critical, `P1` is core behavior, and `P2` is supporting behavior.

| Batch | Area | Planned document | Primary implementation | Current test evidence | Domain review | Priority | Status |
|---:|---|---|---|---|---|---:|---|
| 1 | Data | [Supported images and sidecars](data/supported_images_and_sidecars.md) | `services/image_io.py`, `workers/batch.py`, preprocessing workers | Image orientation, TIFF-loading and summation service tests | Instrument applicability | P0 | code-verified |
| 1 | Data | [Folder-layout detection](data/folder_layouts.md) | `services/preprocessing_layout.py`, preprocessing UI/worker call sites | Headless layout/order and call-site wiring regressions (`tests/test_preprocessing_layout.py`) | No | P0 | code-verified |
| 1 | Preprocessing | [Summation](preprocessing/summation.md) | `services/preprocessing_summation.py`, `SummationWorker` adapter, summation UI orchestration | Headless golden, pre-write validation, partial-output and Spectra warning tests; Qt worker success/cancellation tests | Complete | P0 | domain-reviewed |
| 1 | Preprocessing | [Outlier removal/Clean](preprocessing/clean.md) | `services/preprocessing_clean.py`, `OutlierFilteringWorker` adapter | Baseline golden, pure transform, sidecar, partial-output, warning and cancellation tests; Qt adapter tests | Complete | P1 | domain-reviewed |
| 1 | Preprocessing | [Overlap/pile-up correction](preprocessing/overlap_correction.md) | `services/preprocessing_overlap.py`, `OverlapCorrectionWorker` adapter | Baseline golden, pure equation, segmentation, partial-output, sidecar, cancellation and Qt adapter tests | Required; scientific review pending | P0 | code-verified |
| 1 | Preprocessing | [FITS normalisation](preprocessing/normalisation_fits.md) | `services/preprocessing_normalisation.py`, `NormalisationWorker` adapter | Original-worker golden, pure spatial/temporal calculation, shutter, partial-output, mutation, sidecar and cancellation tests | Required; scientific review pending | P0 | code-verified |
| 1 | Preprocessing | [RADEN TIFF normalisation](preprocessing/normalisation_raden.md) | `services/preprocessing_normalisation_raden.py`, shared local kernel, `RadenNormalisationWorker` adapter | Baseline golden pages; metadata/ToF/pulse validation; incremental/partial TIFF; sidecars; cancellation; worker helper and signal tests (`tests/test_preprocessing_normalisation_raden.py`) | Complete | P0 | domain-reviewed |
| 1 | Preprocessing | [Filtering and masks](preprocessing/filtering_masks.md) | `services/preprocessing_filtering.py`, `FilteringWorker` adapter, `MaskGeneratorDialog` | Baseline golden, pure transform, validation, sidecar, partial-output, dual-progress and cancellation tests; Qt adapter tests | Complete | P1 | domain-reviewed |
| 1 | Preprocessing | [Full Process](preprocessing/full_process.md) | `services/preprocessing_full_process.py`, `FullProcessWorker` Qt adapter, `services/image_io.py` | Current-main no-summation/one-child regressions; Epic #21 end-to-end exact-array golden; loader, cancellation, stage-result and adapter tests (`tests/test_preprocessing_full_process.py`) | Required; scientific review pending | P1 | code-verified |
| 2 | Loading | [FITS stacks and orientation](loading/fits_stacks_orientation.md) | `ImageLoadWorker`, `services/image_io.py`, compatibility import in `workers/batch.py` | FITS orientation round-trip and summation service tests | Instrument orientation | P0 | code-verified |
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
| 6 | Architecture | [Structured preprocessing operation results](architecture/preprocessing_contracts.md) | `NEAT/domain/preprocessing.py`, operation services, Full Process pipeline result | Contract tests, service/worker regressions, and Full Process stage/result aggregation tests; other operation workers remain unmigrated | Not required | P1 | code-verified |
| 6 | Architecture | [Typed loaded-image run contract](architecture/preprocessing_contracts.md) | `NEAT/domain/preprocessing_inputs.py`, classic operation services, Full Process loader | Input contract tests, classic service/adapter regressions, and Full Process loader/pipeline tests; RADEN uses resolved stack-info mappings, and other loaders/workers retain their contracts | Not required | P1 | code-verified |
| 6 | Architecture | [Startup, window composition and state](architecture/startup_state.md) | `app.py`, `main_window.py` | Import and partial GUI tests | Not required | P2 | code-verified |
| 6 | Architecture | [Workers, progress and shutdown](architecture/workers_shutdown.md) | `FitsViewer.closeEvent`, `PreprocessingWorkerRegistry`, `WindowThreadInventory`, assistant and fitting UI | `tests/test_application_shutdown.py` exercises actual offscreen `closeEvent()` with controlled QThreads: preprocessing/fitting/update work, public completion before native exit, queued callbacks emitted around retirement, missing completion and ambiguous startup, empty/closing generations, superseded references, batch-fit restart and overlap guards, cleanup ordering, stop exceptions, callback suppression, data preservation, close retry/resume, assistant retirement, and child-dialog workers. Existing Issue #43 ownership and Full Process suites remain covered | Not required | P0 | code-verified |
| 6 | Support | [Errors, updates and packaging](support/updates_packaging_errors.md) | main window, packaging config | Assistant error tests; no packaged smoke test | Security review | P2 | code-verified |
| 6 | Assistant | [RAG, privacy, feedback and evaluation](support/assistant_architecture.md) | assistant UI/tools | Extensive assistant tests | Safety review | P1 | code-verified |

## Initial test-risk observations

- Batch 1 now has headless service tests for Summation, Clean, Filtering,
  Overlap Correction, classic FITS/TIFF Normalisation, RADEN TIFF
  Normalisation, and Full Process orchestration. It also has headless
  directory-layout classification and call-site wiring tests. Interactive
  mask editing still lacks focused coverage.
- Issue #41 covers direct and real-QThread lifecycle behavior for all seven
  preprocessing adapters. Issue #43 adds GUI ownership and interactive Stop
  regression coverage. Application-close safety now has separate real-thread
  close-event coverage in `tests/test_application_shutdown.py`; non-cooperative
  operations can still delay a retry until they naturally exit.
- Fitting and input loading have materially better headless coverage, but the
  full GUI-to-worker-to-output path is not comprehensively tested.
- Documents for untested behavior can still reach `code-verified`, but the
  missing-test limitation must remain visible and should generate a test
  backlog item.
