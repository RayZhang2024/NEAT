---
title: Individual-edge batch mapping worker
doc_id: neat-tech-mapping-individual-worker
doc_type: technical_reference
functional_area: mapping
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [image stacks]
scientific_review: pending
source_paths: [NEAT/domain/individual_edge.py, NEAT/services/fitting_engine.py, NEAT/ui/mixins/fitting.py, NEAT/workers/batch.py]
source_symbols: [IndividualEdgeFitConfig, FittingEngine.fit_individual_edge, FittingMixin.batch_fit_edges, BatchFitEdgesWorker]
test_paths: [tests/test_individual_edge_service.py, tests/test_individual_edge_adapter.py, tests/test_batch_mapping_outputs.py]
---

# Individual-edge batch mapping worker

The UI retains `_build_batch_fit_context()` as a plain snapshot, converts
batch-valid rows to ordered `IndividualEdgeFitConfig` objects, and passes
full-table provenance separately. `BatchFitEdgesWorker` has no MainWindow
parent or GUI fitting callback: each mapping box produces a summed spectrum
and calls its injected `FittingEngine.fit_individual_edge()` for every edge.
All valid rows are considered; there is no five-edge limit. Duplicate HKLs
remain distinct in-memory slots and service calls. A batch-valid row missing
a later service prerequisite still occupies its slot and maps to NaN on failure.

Arrays have full detector shape plus an edge dimension and begin as NaN.
Successful results are stored only at box centers:

- `d`, `s`, `t`, `eta`
- their uncertainties
- fitted edge height
- derivative FWHM

A failed edge fit leaves NaN for that edge/center while other edges continue.
If every `d` value is NaN, no CSV files are saved.
CSV schemas and full `bragg_rows_text` metadata are unchanged. The two legacy
unknown-edge fallback headers intentionally differ: ungridded uses `edge10`
for its first fallback slot, while gridded uses `edge1`. Duplicate-HKL column
collisions remain as before; this refactor does not redesign output naming.

Progress increments before fitting each box. Consequently 100% means the last
box began processing, not necessarily that output was saved. Stop is checked
between boxes and again before result writing, not between edge fits within a
box. A fit already in progress can finish the current box, but cancellation
then exits without saving partial results.

The constructor emits a message and returns early when no rows are valid, which
can leave a partially initialized object; `run` handles the zero-edge case.

## Retrieval questions

- How many Edge Table rows can individual mapping use?
- What happens when one edge fails in one box?
- Are partial results saved after Stop?
- Which quantities are stored for each edge?
