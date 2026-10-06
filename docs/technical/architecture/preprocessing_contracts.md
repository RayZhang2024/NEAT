---
title: Structured preprocessing operation results
doc_id: neat-tech-architecture-preprocessing-contracts
doc_type: technical_reference
functional_area: architecture
audience: [developer]
neat_version: 4.8.0
verified_commit: 7d0e0002ab94621eac732dd0f9b3d41462752d13
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/domain/preprocessing.py, NEAT/domain/__init__.py, NEAT/__init__.py]
source_symbols: [PreprocessingStatus, ProducedOutput, PreprocessingOperationResult, NEAT.domain.__getattr__]
test_paths: [tests/test_preprocessing_domain.py, tests/test_assistant_semantic_retrieval.py]
---

# Structured preprocessing operation results

## Purpose and scope

`NEAT.domain.preprocessing` defines a small common vocabulary for reporting
the **final result** of a preprocessing operation. It can represent success,
failure, or cancellation together with work counts, ordered artifacts, errors,
and warnings. This is useful when an operation fails or is cancelled after it
has already produced valid outputs; a single success boolean cannot represent
that partial outcome.

This is a common result contract only. Future operation-specific services may
define their own input, configuration, logical work unit, and detailed result
semantics. This module contains no loaded-run models, operation configs,
cancellation tokens, metadata bags, or scientific payload fields.

**Existing Qt preprocessing workers do not yet use this contract.** No worker
was migrated as part of introducing these types.

## Public types

- `PreprocessingStatus` has exactly three final values: `SUCCEEDED`, `FAILED`,
  and `CANCELLED`. A result requires one of these enum values; arbitrary status
  values and strings are rejected at runtime. There are no queued, running, or
  other worker-lifecycle states.
- `ProducedOutput` contains a non-empty path string and an optional, non-empty
  role string. Construction describes an artifact; it does not check that the
  path exists or access the filesystem. The role is intentionally a lightweight
  string, not an exhaustive artifact enum.
- `PreprocessingOperationResult` requires an explicit final status and a
  `processed_count`. It also carries ordered outputs, an optional
  `expected_count`, ordered errors, and ordered warnings.

The types are re-exported from `NEAT.domain`, so existing-style imports such as
`from NEAT.domain import PreprocessingOperationResult` remain supported.

## Counts

Counts must be ordinary non-negative Python integers: booleans, floats,
strings, and other non-integer values are rejected. `expected_count=None`
means the expected total is unknown. When it is known,
`processed_count` cannot exceed it.

`processed_count` means the number of operation-defined logical work units
that **completed successfully**. It is not the attempt count, current loop
index, or number of output artifacts. For example, if four frames are
expected, three complete, and the fourth fails, a result can have
`processed_count=3`, `expected_count=4`, and `status=FAILED`, regardless of how
many artifacts were produced.

The common contract does not decide what a logical unit is. A later operation
service must define whether its units are frames, datasets, stages, or another
operation-specific concept.

## Status, errors, and partial outputs

Status is supplied explicitly and is never inferred from outputs:

- `SUCCEEDED` requires zero errors. Warnings are allowed, and a successful
  operation may legitimately produce zero artifacts.
- `FAILED` requires at least one error.
- `CANCELLED` may have no errors or may retain errors encountered before
  cancellation.

Failed and cancelled results may retain outputs already produced. Such partial
outputs do not create another status; the final status still describes how the
operation ended. Outputs, errors, and warnings retain their supplied order.

## Immutability

`ProducedOutput` and `PreprocessingOperationResult` are frozen dataclasses.
Result construction snapshots iterable collections into tuples and validates
their element types. Later changes to caller-owned lists cannot change the
result. The tuple members are immutable output descriptors or strings, making
the public result boundary deeply immutable.

## Dependency isolation

The dedicated `NEAT.domain.preprocessing` module uses only standard-library
types and does not import Qt, UI modules, workers, NumPy, or perform filesystem
I/O. A fresh-process import regression covers this direct module boundary.

To preserve that boundary, domain re-exports are resolved lazily; requesting a
fitting type still loads its existing fitting module as before. The optional
ONNX Runtime import at the top-level package is also deferred until `FitsViewer`
is requested. The application entry point continues to load ONNX Runtime before
Qt, and the assistant worker-thread test explicitly invokes the existing
`prepare_local_embedding_runtime()` preloader before importing Qt.

## Adoption boundary

This increment establishes contract-level coverage only. Existing workers,
their Qt signals and `succeeded` attributes, GUI completion behavior,
cancellation implementation, processing order, scientific calculations, and
output generation remain unchanged. Worker/service adoption belongs to later
Epic #21 issues and must define each operation's logical work unit explicitly.
