# NEAT Technical Documentation Build Plan

## Decision: staged verification, continuous authoring

Generating every document in one unreviewed pass is technically possible but
not reliable enough for scientific software. The current implementation spans
thousands of lines across GUI controllers, background workers, numerical
fitting code, file readers and plotting dialogs. Some blocks have strong
headless tests; several preprocessing and post-processing behaviors currently
have limited direct test coverage.

The documentation will therefore be built in functional batches. All
code-derived drafting within a batch can proceed continuously, but each batch
has a verification gate before its documents become available to the AI
assistant.

Baseline for the initial build:

- NEAT version: `4.8.0`
- Git revision: `628c767ef44186e4301454f24a54fbc05ad71233`
- branch at planning time: `main`

Uncommitted workspace changes are not represented by that revision and must be
identified explicitly when they affect documented behavior.

## Deliverables

The complete technical-reference programme will produce:

1. 38 currently identified modular technical documents, with the matrix
   extended if code tracing reveals additional independent blocks;
2. a functional coverage matrix with review state and source traceability;
3. a consistent parameter, unit and hard-coded-constant inventory;
4. input/output schemas and folder-layout descriptions;
5. algorithm descriptions and equations derived from executable behavior;
6. validation, warning and failure-mode catalogues;
7. known assumptions, limitations and suspected implementation issues;
8. retrieval questions for every functional block;
9. assistant retrieval metadata and routing rules; and
10. regression checks ensuring approved technical sections remain retrievable.

## Batch sequence

### Batch 0 — Documentation foundation

Deliver:

- directory structure and authority rules;
- technical-document template;
- coverage matrix;
- source-version policy;
- RAG approval and evaluation workflow.

Gate: template and coverage matrix reviewed for completeness.

### Batch 1 — Data conventions and preprocessing

Build:

1. supported image and sidecar conventions;
2. folder-layout detection;
3. Summation;
4. outlier removal/Clean;
5. overlap or pile-up correction;
6. normalisation for FITS datasets;
7. RADEN TIFF normalisation;
8. filtering and binary masks; and
9. Full Process orchestration.

Primary sources:

- `NEAT/ui/mixins/preprocessing.py`
- `NEAT/workers/preprocessing.py`
- image I/O helpers in `NEAT/workers/batch.py`

Required review:

- confirm the scientific purpose and applicability of overlap correction;
- confirm intended shutter-count semantics;
- distinguish IMAT-style FITS workflows from RADEN TIFF workflows.

Gate: code verification, at least one retrieval question per document, and
domain review of correction/normalisation claims.

### Batch 2 — Data loading, wavelength and instrument configuration

Build:

1. FITS stack loading and image orientation;
2. NeXus signal, axis and geometry discovery;
3. RADEN TIFF, `.stat` and JSON metadata handling;
4. time-of-flight to wavelength conversion;
5. flight path and time-delay behavior;
6. manual wavelength anchors and interpolation;
7. intensity-profile import; and
8. phase definitions and theoretical-edge population.

Primary sources:

- loading helpers in `NEAT/workers/batch.py`
- loading/configuration paths in `NEAT/ui/mixins/fitting.py`
- phase and Bragg-edge helpers in `NEAT/core/`

Gate: conversion equations, units and instrument assumptions independently
checked; code tests identified or added for every supported input family.

### Batch 3 — Spectrum extraction and fitting

Build:

1. image display and macro-pixel extraction;
2. Edge Table generation and derived fitting interval;
3. pseudo-Voigt edge model;
4. staged individual-edge fitting;
5. unknown-phase behavior;
6. multi-edge pattern fitting;
7. fixed versus refined parameters;
8. parameter bounds and initialization;
9. residuals and fit diagnostics; and
10. uncertainty estimation.

Primary sources:

- `NEAT/ui/mixins/fitting.py`
- `NEAT/core/fitting.py`
- relevant fitting tests

Required review:

- equations and parameter definitions;
- physical interpretation of `s`, `t`, `eta`, FWHM and edge height;
- distinction between numerical convergence and scientific validity.

Gate: numerical equations verified against implementation, fitting tests
mapped, and domain review completed before RAG approval.

### Batch 4 — Batch mapping and result files

Build:

1. mapping ROI and macro-pixel geometry;
2. step size, skipping and interpolation;
3. individual-edge batch worker;
4. pattern batch worker;
5. progress, cancellation and failure handling;
6. gridded and ungridded CSV schemas;
7. metadata headers and configuration snapshots; and
8. output naming and missing-value behavior.

Primary sources:

- batch controls in `NEAT/ui/mixins/fitting.py`
- `NEAT/workers/batch.py`

Gate: output schemas checked against generated DataFrames and representative
files; interpolation and coordinate conventions explicitly tested.

### Batch 5 — Post-processing and visualisation

Build:

1. result CSV ingestion;
2. parameter-button generation and metric dictionary;
3. parameter-map construction;
4. masks and FITS export;
5. pixel-to-millimetre conversion;
6. colour limits and plotting;
7. strain calculation from `d0`;
8. ROI mean calculation;
9. point selection and line profiles; and
10. image-result loading.

Primary sources:

- `NEAT/ui/mixins/postprocessing.py`
- `ParameterPlotDialog` and `LineProfileDialog` in `NEAT/ui/dialogs.py`

Required review:

- scientific meaning of mapped quantities;
- correct reference-spacing guidance;
- limitations on strain and residual-stress interpretation.

Gate: formulas, units and coordinate conversions tested and domain reviewed.

### Batch 6 — Application architecture and support

Build:

1. application startup and main-window composition;
2. persistent settings and state lifecycle;
3. worker-thread ownership and shutdown;
4. progress reporting and cancellation;
5. update checking and packaging;
6. error-reporting conventions;
7. AI assistant architecture, privacy and feedback; and
8. developer extension points.

Primary sources:

- `NEAT/app.py`
- `NEAT/ui/main_window.py`
- worker modules
- assistant modules and tests

Gate: operational behavior verified on a clean launch and close; security and
privacy statements checked separately.

## Per-document production workflow

Each document follows the same sequence:

1. **Scope** — identify user questions and exact code paths.
2. **Trace** — follow GUI input through validation, worker/numerical logic and
   output.
3. **Draft** — populate the standard template with code-derived facts.
4. **Cross-check** — compare with the user manual, README, tests and actual
   defaults.
5. **Flag uncertainty** — record ambiguities, suspected bugs and scientific
   questions rather than guessing.
6. **Code verification** — verify names, equations, branches, units, output
   schemas and errors against the current workspace.
7. **Domain review** — obtain expert confirmation for scientific purpose,
   interpretation and applicability.
8. **Retrieval evaluation** — add basic, technical, troubleshooting and
   limitation questions.
9. **RAG approval** — expose the document to the assistant only after required
   checks pass.

## Definition of done for one functional block

A block is complete when:

- all source paths and relevant functions/classes are recorded;
- inputs, preconditions and supported structures are explicit;
- every parameter has a definition, unit, default and validation rule where
  applicable;
- the algorithm is described step by step with equations where used;
- output names, formats, shapes, types and metadata are documented;
- progress, cancellation and error behavior are described;
- hard-coded constants and assumptions are visible;
- scientific limitations are reviewed or explicitly marked pending;
- existing tests are mapped and missing-test risks are listed;
- at least four retrieval questions pass:
  - how to use it;
  - how it works;
  - why it failed;
  - when it should not be used; and
- the document contains no unsupported claim.

## Change and drift control

Each document records `neat_version`, `verified_commit`, source paths and key
symbols. A later code change touching a listed source does not automatically
make the document wrong, but it marks the document for review.

Recommended maintenance controls:

- a manifest mapping source paths to technical documents;
- a CI warning when mapped source files change without a corresponding
  technical-document review;
- retrieval regression tests for approved sections;
- version metadata filters when multiple NEAT versions must be supported; and
- a changelog entry when user-visible technical behavior changes.

## RAG integration strategy

Technical documents will not simply be mixed indiscriminately with short FAQs.
The retrieval layer should distinguish:

- `user_guide`
- `faq`
- `troubleshooting`
- `parameter_reference`
- `technical_reference`
- `scientific_guidance`

How-to questions should normally rank concise user/FAQ material first.
Questions asking how an algorithm works should favor technical references.
Scientific interpretation and guarantee requests must retain the existing
human-review and escalation rules.

Technical documents should be split by Markdown headings. Each retrieved chunk
must retain:

- document ID and type;
- functional area;
- NEAT version and verified revision;
- review status;
- source paths and symbols;
- instrument applicability; and
- scientific-review status.

Only `approved-for-rag` documents should be used for production answers.

## Recommended execution

Proceed batch by batch, beginning with Batch 1. Within each batch, generate all
code-derived drafts in one continuous pass, then resolve the batch review gate.
This approach is substantially faster than waiting after every document while
avoiding the accuracy risks of generating the entire reference without
intermediate verification.
