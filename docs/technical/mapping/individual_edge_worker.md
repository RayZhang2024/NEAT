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
source_paths: [NEAT/ui/mixins/fitting.py, NEAT/workers/batch.py]
source_symbols: [FittingMixin.batch_fit_edges, BatchFitEdgesWorker]
test_paths: [tests/test_batch_mapping_outputs.py]
---

# Individual-edge batch mapping worker

The worker snapshots valid Edge Table rows, phase/instrument metadata and fix
flags before starting. All valid Edge Table rows are considered; there is no
five-edge limit. Each mapping box produces a summed spectrum and calls
`fit_region` separately for
every valid edge.

Arrays have full detector shape plus an edge dimension and begin as NaN.
Successful results are stored only at box centers:

- `d`, `s`, `t`, `eta`
- their uncertainties
- fitted edge height
- derivative FWHM

A failed edge fit leaves NaN for that edge/center while other edges continue.
If every `d` value is NaN, no CSV files are saved.

Progress increments before fitting each box. Consequently 100% means the last
box began processing, not necessarily that output was saved. Stop is checked
between boxes and again before result writing. A fit already in progress is
allowed to return, but cancellation then exits without saving partial results.

The constructor emits a message and returns early when no rows are valid, which
can leave a partially initialized object; `run` handles the zero-edge case.

## Retrieval questions

- How many Edge Table rows can individual mapping use?
- What happens when one edge fails in one box?
- Are partial results saved after Stop?
- Which quantities are stored for each edge?
