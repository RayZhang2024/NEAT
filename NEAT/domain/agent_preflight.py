"""Deterministic, Qt-free contracts for the local dataset preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .observation import (
    DatasetCapabilities,
    DatasetMode,
    DatasetObservation,
    ImageObservation,
    PlotObservation,
    RoiBounds,
    SpectrumObservation,
    WavelengthAxisStatus,
)


class PreflightReadiness(str, Enum):
    READY = "ready_for_exploration"
    LIMITED = "limited_or_inconclusive"
    INVALID = "invalid_or_no_data"


class PreflightRunState(str, Enum):
    COMPLETE = "complete"
    BUSY = "busy_retry"
    CANCELLED = "cancelled"
    STALE = "dataset_changed"
    FAILED = "failed"


@dataclass(frozen=True)
class DatasetAssessment:
    readiness: PreflightReadiness
    dataset_id: str
    revision: int
    mode: DatasetMode
    capabilities: DatasetCapabilities
    observations: tuple[str, ...]
    evidence_references: tuple[str, ...]
    warnings: tuple[str, ...]
    unsupported_operations: tuple[str, ...]


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    wavelength: float | None
    image: ImageObservation


@dataclass(frozen=True)
class PreflightResult:
    run_id: str
    dataset_id: str
    revision: int
    dataset: DatasetObservation
    assessment: DatasetAssessment
    roi: RoiBounds | None
    frames: tuple[FrameEvidence, ...]
    spectrum: SpectrumObservation | None
    plot: PlotObservation | None


@dataclass(frozen=True)
class PreflightOutcome:
    run_id: str
    dataset_id: str
    revision: int
    state: PreflightRunState
    result: PreflightResult | None = None
    message: str = ""


def representative_frame_indices(frame_count: int) -> tuple[int, ...]:
    """Choose first, lower midpoint, and last without duplicating short stacks."""
    if frame_count < 0:
        raise ValueError("Frame count cannot be negative.")
    if frame_count == 0:
        return ()
    return tuple(dict.fromkeys((0, (frame_count - 1) // 2, frame_count - 1)))


def assess_dataset(
    dataset: DatasetObservation,
    roi: RoiBounds | None,
    *,
    evidence_references: tuple[str, ...] = (),
    evidence_warnings: tuple[str, ...] = (),
    spectrum_observed: bool | None = None,
) -> DatasetAssessment:
    """Classify only observed capabilities; never infer a phase or sample region."""
    observations = [
        f"Dataset mode: {dataset.mode.value}.",
        f"Wavelength axis: {dataset.wavelength_axis_status.value}.",
    ]
    warnings = list(evidence_warnings)
    unsupported = []
    if dataset.mode is DatasetMode.IMAGES:
        observations.append(
            f"{dataset.frame_count or 0} frames; image shape {dataset.image_shape}."
        )
        if roi is None:
            warnings.append("No valid selected image ROI was available; ROI spectrum is not yet available.")
        else:
            observations.append(
                f"Selected ROI x[{roi.x_min},{roi.x_max}), y[{roi.y_min},{roi.y_max})."
            )
    elif dataset.mode is DatasetMode.PROFILE:
        observations.append("Imported spectrum-only input; image previews are not applicable.")
        unsupported.append("Image previews and mapping are not applicable to spectrum-only input.")
    else:
        warnings.append("No scientific dataset is loaded.")

    if dataset.wavelength_axis_status is not WavelengthAxisStatus.VALID:
        warnings.append("Wavelength axis is not valid for a conclusive spectral preflight.")
    if dataset.provenance.companion_metadata_status != "provided":
        warnings.append(
            "Companion metadata is not confirmed; calibration provenance is provisional."
        )
    if dataset.provenance.wavelength_source == "unknown":
        warnings.append("Wavelength calibration source is unknown.")
    if dataset.metadata_flags:
        observations.append("Metadata flags: " + ", ".join(dataset.metadata_flags) + ".")
    if spectrum_observed is False and dataset.capabilities.spectrum_available and (
        dataset.mode is DatasetMode.PROFILE or roi is not None
    ):
        warnings.append("Measured spectrum extraction did not yield evidence.")

    if dataset.mode is DatasetMode.EMPTY or not (
        dataset.capabilities.images_available or dataset.capabilities.spectrum_available
    ):
        readiness = PreflightReadiness.INVALID
    elif (
        dataset.wavelength_axis_status is not WavelengthAxisStatus.VALID
        or (dataset.mode is DatasetMode.IMAGES and roi is None)
        or spectrum_observed is False
    ):
        readiness = PreflightReadiness.LIMITED
    else:
        readiness = PreflightReadiness.READY

    return DatasetAssessment(
        readiness=readiness,
        dataset_id=dataset.dataset_id,
        revision=dataset.revision,
        mode=dataset.mode,
        capabilities=dataset.capabilities,
        observations=tuple(observations),
        evidence_references=evidence_references,
        warnings=tuple(warnings),
        unsupported_operations=tuple(unsupported),
    )
