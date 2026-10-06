---
title: Preprocessing input and result contracts
doc_id: neat-tech-architecture-preprocessing-contracts
doc_type: technical_reference
functional_area: architecture
audience: [developer]
neat_version: 4.8.0
verified_commit: 9acc2c94e1927b9e282183b7066c69889430b655
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [NEAT/domain/preprocessing.py, NEAT/domain/preprocessing_inputs.py, NEAT/domain/__init__.py, NEAT/services/preprocessing_summation.py, NEAT/services/preprocessing_clean.py, NEAT/services/preprocessing_filtering.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingStatus, ProducedOutput, PreprocessingOperationResult, LoadedImageRun, sum_loaded_image_runs, SummationWorker, clean_loaded_image_runs, OutlierFilteringWorker, filter_loaded_image_runs, FilteringWorker]
test_paths: [tests/test_preprocessing_domain.py, tests/test_preprocessing_inputs.py, tests/test_preprocessing_summation.py, tests/test_preprocessing_clean.py, tests/test_preprocessing_filtering.py, tests/test_preprocessing_workers.py]
---

# Preprocessing input and result contracts

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

The headless Summation, Clean and Filtering services consume `LoadedImageRun`
inputs and return `PreprocessingOperationResult`. Their Qt worker adapters
convert existing dictionaries at the worker boundary. Summation was the first
adopter, Clean the second (#29), and Filtering the third (#31). Remaining
preprocessing workers/loaders still use their existing dictionaries and have
not migrated to the shared result contract.

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
behaviors are unchanged by this contract. Summation, Clean and Filtering have
migrated; other Qt preprocessing workers have not.

## Loaded classic image-run input

`NEAT.domain.preprocessing_inputs.LoadedImageRun` is the common in-memory input
vocabulary introduced by Issue #25. It is separate from
`PreprocessingOperationResult`: the latter describes a final outcome, while
`LoadedImageRun` describes a primary source identity, loaded frames, physical
source-folder provenance, and loader errors.

- `primary_source` is a non-empty identity string. It is retained as supplied;
  it is not normalized, resolved, or checked against the filesystem.
- `frames` accepts a mapping of string keys to NumPy arrays. Its iteration
  order is preserved exactly, keys need not be numeric, and empty mappings are
  valid. The common contract does not validate image shape, dimensions, dtype,
  orientation, or scientific suitability. NumPy array subclasses are accepted.
- The mapping is shallowly copied and exposed through a read-only mapping
  proxy. Replacing or adding entries in the caller's original mapping does not
  change the captured mapping. The arrays themselves are not copied or frozen:
  each stored value is the exact array object supplied by the caller, and
  later changes to that array's contents remain visible through the run.
- `source_folders` is an ordered, immutable snapshot of physical folders. If
  omitted, it defaults to `(primary_source,)`. Explicit provenance must be a
  non-empty ordered sequence of non-empty strings; its order and duplicates
  are retained without path normalization or existence checks. The primary
  source is not required to appear in this sequence, because a logical sample
  identity can correspond to separate physical run folders.
- `load_errors` is an ordered immutable snapshot of strings. Empty errors are
  allowed; the contract describes loader state and does not decide whether a
  particular operation should reject a run containing errors.
- Run equality is object identity; it does not compare NumPy payloads. The
  concise representation reports frame/error counts rather than dumping image
  contents.

The dedicated input module may depend on NumPy, but has no Qt, UI, worker,
`FitsViewer`, `QApplication`, or filesystem/path-inspection dependency. Its
construction does not inspect or access the filesystem and requires no
`QApplication`.

The input type began as contract-level coverage. Issue #27 adopted it for
Summation, Issue #29 for Clean, and Issue #31 for Filtering. Their headless
services receive loaded frames directly, while compatibility adapters convert
legacy worker dictionaries. Clean and Filtering use `primary_source` alone for
sidecars and do not reject nonempty `load_errors`; Summation's validation and
`source_folders` rules differ. Filtering sorts frame suffixes while Clean keeps
mapping insertion order. Existing loaders and other workers continue to use
their current dictionaries. Loader-specific suffix parsing, duplicate handling
and ordering remain unchanged. Spectra, shutter counts, masks, RADEN metadata
and operation configuration remain operation-specific; cancellation and
progress are passed to these services as plain callbacks rather than being
added to the common input model.

## Adoption boundary

Summation, Clean and Filtering use `LoadedImageRun` and
`PreprocessingOperationResult`. `SummationWorker`, `OutlierFilteringWorker` and
`FilteringWorker` convert legacy dictionaries at their compatibility
boundaries, delegate to headless services, and retain structured results
alongside Qt signals and `succeeded`. For Clean and Filtering, the logical work
unit is a successfully written image; reports and sidecars do not count.
Filtering's two progress streams and early duplicate `finished` signals remain
legacy adapter behavior, not a general result-contract rule. Remaining
preprocessing workers and loaders are unmigrated; their future migrations are
separate Epic #21 work and must define each operation's logical work unit.
