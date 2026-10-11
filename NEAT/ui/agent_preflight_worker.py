"""Bounded observation work for the local AI Agent preflight."""

from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_preflight import (
    FrameEvidence,
    PreflightOutcome,
    PreflightResult,
    PreflightRunState,
    assess_dataset,
    representative_frame_indices,
)
from ..domain.observation import (
    DatasetMode,
    ImageViewSpec,
    ObservationStatus,
    RoiBounds,
    WavelengthAxisStatus,
)
from ..services.scientific_observation import (
    ObservationErrorBase,
    ScientificDatasetHandle,
    StaleDatasetError,
    get_image_preview,
    get_roi_spectrum,
    inspect_dataset,
    render_spectrum_preview,
)


class _Cancelled(Exception):
    pass


class PreflightWorker(QThread):
    """Uses only a captured handle; never touches a QWidget or live window."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)

    def __init__(
        self,
        run_id: str,
        handle: ScientificDatasetHandle,
        roi: RoiBounds | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.dataset_id = handle.dataset_id
        self.revision = handle.revision
        self._handle: ScientificDatasetHandle | None = handle
        self._roi = roi
        self._cancel = Event()

    def stop(self) -> None:
        """A running service call finishes before this request takes effect."""
        self._cancel.set()

    def _checkpoint(self, handle: ScientificDatasetHandle) -> None:
        if self._cancel.is_set():
            raise _Cancelled
        handle.assert_current()

    def run(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            self._checkpoint(handle)
            self.progress.emit(5, "Inspecting dataset metadata")
            dataset = inspect_dataset(handle)
            self._checkpoint(handle)

            frames: list[FrameEvidence] = []
            evidence_references: list[str] = []
            warnings: list[str] = []
            indices = representative_frame_indices(dataset.frame_count or 0)
            for position, index in enumerate(indices):
                self._checkpoint(handle)
                self.progress.emit(
                    15 + (45 * position // max(len(indices), 1)),
                    f"Rendering image {position + 1} of {len(indices)}",
                )
                try:
                    image = get_image_preview(
                        handle, index, ImageViewSpec(max_dimension=512)
                    )
                except (ObservationErrorBase, ValueError) as exc:
                    warnings.append(
                        f"Image frame {index} preview unavailable ({type(exc).__name__})."
                    )
                    continue
                if image.status is not ObservationStatus.OK or image.png_bytes is None:
                    warnings.append(f"Image frame {index} preview unavailable.")
                    continue
                wavelength = None
                if (
                    dataset.wavelength_axis_status is WavelengthAxisStatus.VALID
                    and handle.wavelengths is not None
                    and handle.wavelengths.size == dataset.frame_count
                ):
                    wavelength = float(handle.wavelengths[index])
                frames.append(FrameEvidence(index, wavelength, image))
                evidence_references.append(f"image:frame:{index}")
                if image.sampled_nonfinite_count:
                    warnings.append(
                        f"Image frame {index} preview sampled "
                        f"{image.sampled_nonfinite_count} non-finite pixels."
                    )

            self._checkpoint(handle)
            spectrum = None
            plot = None
            if dataset.capabilities.spectrum_available and (
                dataset.mode is DatasetMode.PROFILE or self._roi is not None
            ):
                self.progress.emit(70, "Extracting measured spectrum")
                try:
                    spectrum = get_roi_spectrum(handle, self._roi)
                    self._checkpoint(handle)
                    if spectrum.status is not ObservationStatus.OK:
                        warnings.append("Measured spectrum is invalid.")
                        spectrum = None
                    if spectrum is None:
                        raise ValueError("No valid measured spectrum was returned.")
                    evidence_references.append("spectrum:full-source")
                    if spectrum.nonfinite_intensity_count:
                        warnings.append(
                            "Measured spectrum contains "
                            f"{spectrum.nonfinite_intensity_count} non-finite intensities."
                        )
                    self.progress.emit(85, "Rendering measured spectrum")
                    plot = render_spectrum_preview(spectrum)
                    if plot.png_bytes is not None:
                        evidence_references.append("spectrum:plot")
                    else:
                        warnings.append("The measured spectrum has no samples in the plot window.")
                except (ObservationErrorBase, ValueError) as exc:
                    warnings.append(
                        f"Measured spectrum unavailable ({type(exc).__name__})."
                    )
                    spectrum = None
                    plot = None

            self._checkpoint(handle)
            assessment = assess_dataset(
                dataset,
                self._roi,
                evidence_references=tuple(evidence_references),
                evidence_warnings=tuple(warnings),
                spectrum_observed=spectrum is not None,
            )
            result = PreflightResult(
                run_id=self.run_id,
                dataset_id=self.dataset_id,
                revision=self.revision,
                dataset=dataset,
                assessment=assessment,
                roi=self._roi,
                frames=tuple(frames),
                spectrum=spectrum,
                plot=plot,
            )
            self.progress.emit(100, "Preflight complete")
            outcome = PreflightOutcome(
                self.run_id, self.dataset_id, self.revision,
                PreflightRunState.COMPLETE, result,
            )
        except _Cancelled:
            outcome = PreflightOutcome(
                self.run_id, self.dataset_id, self.revision,
                PreflightRunState.CANCELLED, message="Preflight cancelled.",
            )
        except StaleDatasetError:
            outcome = PreflightOutcome(
                self.run_id, self.dataset_id, self.revision,
                PreflightRunState.STALE, message="Dataset changed — rerun preflight.",
            )
        except (ObservationErrorBase, ValueError) as exc:
            outcome = PreflightOutcome(
                self.run_id, self.dataset_id, self.revision,
                PreflightRunState.FAILED,
                message=f"Preflight could not complete ({type(exc).__name__}).",
            )
        except Exception:  # noqa: BLE001 - contain unexpected worker failures for the Qt host
            outcome = PreflightOutcome(
                self.run_id, self.dataset_id, self.revision,
                PreflightRunState.FAILED,
                message="Preflight could not complete. Retry after checking the loaded dataset.",
            )
        finally:
            # Frame views are retained only until the worker exits. The result
            # contains bounded previews and derived spectra, not the old stack.
            self._handle = None
        self.completed.emit(outcome)
