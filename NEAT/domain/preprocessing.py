"""GUI-independent final result contracts for preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PreprocessingStatus(str, Enum):
    """Final states for a completed preprocessing operation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProducedOutput:
    """A path produced by an operation, optionally labelled with a role."""

    path: str
    role: str | None = None

    def __post_init__(self) -> None:
        if type(self.path) is not str:
            raise TypeError("Output path must be a string")
        if not self.path:
            raise ValueError("Output path must not be empty")
        if self.role is not None:
            if type(self.role) is not str:
                raise TypeError("Output role must be a string when provided")
            if not self.role:
                raise ValueError("Output role must not be empty when provided")


def _snapshot(values: Iterable[object], field_name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable collection, not text")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable collection") from exc


@dataclass(frozen=True, slots=True)
class PreprocessingOperationResult:
    """Immutable final status, outputs, counts, errors, and warnings.

    ``processed_count`` counts operation-defined logical work units that
    completed successfully. It is independent of attempted work and the number
    of produced artifacts.
    """

    status: PreprocessingStatus
    processed_count: int
    outputs: tuple[ProducedOutput, ...] = ()
    expected_count: int | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, PreprocessingStatus):
            raise TypeError("Status must be a PreprocessingStatus value")

        _validate_count(self.processed_count, "processed_count")
        if self.expected_count is not None:
            _validate_count(self.expected_count, "expected_count")
            if self.processed_count > self.expected_count:
                raise ValueError("processed_count cannot exceed expected_count")

        outputs = _snapshot(self.outputs, "outputs")
        if not all(type(output) is ProducedOutput for output in outputs):
            raise TypeError("Outputs must contain only ProducedOutput values")

        errors = _snapshot(self.errors, "errors")
        warnings = _snapshot(self.warnings, "warnings")
        if not all(type(error) is str for error in errors):
            raise TypeError("Errors must contain only strings")
        if not all(type(warning) is str for warning in warnings):
            raise TypeError("Warnings must contain only strings")

        if self.status is PreprocessingStatus.SUCCEEDED and errors:
            raise ValueError("Succeeded results cannot contain errors")
        if self.status is PreprocessingStatus.FAILED and not errors:
            raise ValueError("Failed results must contain at least one error")

        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)


def _validate_count(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer count")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
