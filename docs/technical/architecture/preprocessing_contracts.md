---
title: Structured preprocessing operation results
doc_id: neat-tech-architecture-preprocessing-contracts
doc_type: technical_reference
functional_area: architecture
audience: [developer]
neat_version: 4.8.0
verified_commit: ee1aed5947d7c1c107cc60ecf9a887abeddf336a
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/domain/preprocessing.py, NEAT/domain/__init__.py]
source_symbols: [PreprocessingStatus, ProducedOutput, PreprocessingOperationResult]
test_paths: [tests/test_preprocessing_domain.py]
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

The types are added to the existing eager exports from `NEAT.domain`, so
existing-style imports such as
`from NEAT.domain import PreprocessingOperationResult` remain supported. The
established parent-package initialization and fitting/individual-edge import
behavior are unchanged.

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

The dedicated `NEAT.domain.preprocessing` module is dependency-light: its own
implementation uses only standard-library types, has no NumPy/Qt/UI/worker
dependency, and performs no filesystem I/O. Its boundary test loads and checks
that source in isolation, without running package initializers. This distinction
matters because a normal dotted import first executes the existing `NEAT` and
`NEAT.domain` initializers, which retain their established optional ONNX and
eager fitting/individual-edge imports. Those parent-package initialization
behaviors are unchanged by this contract. Existing Qt preprocessing workers
are still not migrated.

## Adoption boundary

This increment establishes contract-level coverage only. Existing workers,
their Qt signals and `succeeded` attributes, GUI completion behavior,
cancellation implementation, processing order, scientific calculations, and
output generation remain unchanged. Worker/service adoption belongs to later
Epic #21 issues and must define each operation's logical work unit explicitly.
