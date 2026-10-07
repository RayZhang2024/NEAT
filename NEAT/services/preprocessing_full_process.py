"""Headless orchestration for the classic Full Process workflow."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from ..domain import (
    LoadedImageRun,
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
)
from .image_io import load_image_file
from .preprocessing_clean import clean_loaded_image_runs
from .preprocessing_layout import discover_full_process_summation
from .preprocessing_normalisation import normalise_loaded_image_runs
from .preprocessing_normalisation import validate_normalisation_windows
from .preprocessing_overlap import correct_loaded_image_run
from .preprocessing_summation import sum_loaded_image_runs

ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
CancellationToken = threading.Event
OperationStartedCallback = Callable[["FullProcessStage", CancellationToken], None]
OperationFinishedCallback = Callable[["FullProcessStage", CancellationToken], None]
VoidCallback = Callable[[], None]


class FullProcessStage(str, Enum):
    SAMPLE_SUMMATION = "sample_summation"
    OPEN_BEAM_SUMMATION = "open_beam_summation"
    SAMPLE_CLEAN = "sample_clean"
    OPEN_BEAM_CLEAN = "open_beam_clean"
    SAMPLE_OVERLAP = "sample_overlap"
    OPEN_BEAM_OVERLAP = "open_beam_overlap"
    NORMALISATION = "normalisation"


class FullProcessStageOutcome(str, Enum):
    SKIPPED = "skipped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FullProcessDiagnostic:
    stage: FullProcessStage | None
    message: str


@dataclass(frozen=True, slots=True)
class FullProcessStageResult:
    stage: FullProcessStage
    branch: str | None
    outcome: FullProcessStageOutcome
    input_folder: str | None
    artifact_folder: str | None
    propagation_folder: str | None
    operation_result: PreprocessingOperationResult | None
    secondary_input_folder: str | None = None


@dataclass(frozen=True, slots=True)
class FullProcessPipelineResult:
    status: PreprocessingStatus
    stages: tuple[FullProcessStageResult, ...]
    failed_stage: FullProcessStage | None
    cancelled_stage: FullProcessStage | None
    outputs: tuple[ProducedOutput, ...]
    errors: tuple[FullProcessDiagnostic, ...]
    warnings: tuple[FullProcessDiagnostic, ...]


class _StageAbort(RuntimeError):
    def __init__(self, message: str, stage: FullProcessStage):
        super().__init__(message)
        self.stage = stage


class _StageOperationFailure(RuntimeError):
    def __init__(self, error: Exception, wrapper_message: str):
        super().__init__(str(error))
        self.error = error
        self.wrapper_message = wrapper_message


@dataclass(slots=True)
class _StageBuilder:
    stage: FullProcessStage
    branch: str | None
    input_folder: str | None
    artifact_folder: str | None = None
    propagation_folder: str | None = None
    operation_result: PreprocessingOperationResult | None = None
    secondary_input_folder: str | None = None
    outcome: FullProcessStageOutcome = FullProcessStageOutcome.FAILED
    recorded: bool = False


def load_full_process_run(
    folder: str,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
) -> LoadedImageRun:
    """Load one Full Process FITS/TIFF folder in raw filesystem order.

    This intentionally has no cancellation check: a loader invocation always
    processes every eligible file, matching the historical worker.
    """
    frames: dict[str, np.ndarray] = {}
    errors: list[str] = []
    if not os.path.isdir(folder):
        if message_callback is not None:
            message_callback(f"Folder not found: {folder}")
        return LoadedImageRun(folder, frames, load_errors=errors)

    short_path = _short_path(folder, levels=2)
    try:
        filenames = [
            filename
            for filename in os.listdir(folder)
            if filename.lower().endswith((".fits", ".fit", ".tiff", ".tif"))
        ]
        total = len(filenames)
        for processed, filename in enumerate(filenames, start=1):
            stem = os.path.splitext(filename)[0]
            suffix = stem.rsplit("_", 1)[-1].strip()
            error: str | None = None
            if not suffix:
                error = f"File '{filename}' has an empty frame suffix."
            elif suffix in frames:
                error = (
                    f"File '{filename}' duplicates frame suffix '{suffix}'. "
                    "Frame names must be unique."
                )
            else:
                try:
                    data = load_image_file(os.path.join(folder, filename))
                    frames[suffix] = data.astype(np.float32)
                except Exception as exc:
                    error = (
                        f"Error loading file {filename} in \\{short_path}: {exc}"
                    )
            if error is not None:
                errors.append(error)
                if message_callback is not None:
                    message_callback(error)
            if progress_callback is not None:
                progress_callback(int((processed / total) * 100))
    except Exception as exc:
        error = f"Error reading folder \\{short_path}: {exc}"
        errors.append(error)
        if message_callback is not None:
            message_callback(error)

    return LoadedImageRun(folder, frames, load_errors=errors)


def _short_path(path: str, *, levels: int = 2) -> str:
    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    return os.path.join(*parts[-levels:]) if len(parts) >= levels else normalized


class FullProcessPipeline:
    """Compose existing preprocessing services without importing GUI/Qt code."""

    def __init__(
        self,
        sample_folder: str,
        open_beam_folder: str,
        output_folder: str,
        base_name: str,
        window_half: int,
        adjacent_sum: int,
        *,
        message_callback: MessageCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        load_progress_callback: ProgressCallback | None = None,
        parent_running: Callable[[], bool] | None = None,
        operation_started: OperationStartedCallback | None = None,
        operation_finished: OperationFinishedCallback | None = None,
        stage_completed: VoidCallback | None = None,
        normalisation_finished: VoidCallback | None = None,
        normalisation_pacing: Callable[[int], None] | None = None,
    ) -> None:
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half, self.adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self._message = message_callback or (lambda _message: None)
        self._progress = progress_callback or (lambda _value: None)
        self._load_progress = load_progress_callback or (lambda _value: None)
        self._parent_running = parent_running or (lambda: True)
        self._operation_started = operation_started or (lambda _stage, _token: None)
        self._operation_finished = operation_finished or (lambda _stage, _token: None)
        self._stage_completed = stage_completed or (lambda: None)
        self._normalisation_finished = normalisation_finished or (lambda: None)
        self._normalisation_pacing = normalisation_pacing or (lambda _seconds: None)
        self._stages: list[FullProcessStageResult] = []
        self._outputs: list[ProducedOutput] = []
        self._errors: list[FullProcessDiagnostic] = []
        self._warnings: list[FullProcessDiagnostic] = []
        self._failed_stage: FullProcessStage | None = None
        self._cancelled_stage: FullProcessStage | None = None
        self._current: _StageBuilder | None = None
        self._overall_status = PreprocessingStatus.SUCCEEDED

    def run(self) -> FullProcessPipelineResult:
        self._message("=== <b>Full Process Pipeline Started</b> ===")
        try:
            sample_sum = self.maybe_do_summation(self.sample_folder, "Sample")
            if not self._continue(FullProcessStage.SAMPLE_SUMMATION, "sample summation", "sample", self.sample_folder):
                return self.result()
            self._progress(0)

            beam_sum = self.maybe_do_summation(self.open_beam_folder, "OpenBeam")
            if not self._continue(FullProcessStage.OPEN_BEAM_SUMMATION, "open‑beam summation", "open_beam", self.open_beam_folder):
                return self.result()
            self._progress(0)

            sample_clean = self.do_outlier_removal(sample_sum, "Sample")
            if not self._continue(FullProcessStage.SAMPLE_CLEAN, "sample cleaning", "sample", sample_sum):
                return self.result()
            self._progress(0)

            beam_clean = self.do_outlier_removal(beam_sum, "OpenBeam")
            if not self._continue(FullProcessStage.OPEN_BEAM_CLEAN, "open‑beam cleaning", "open_beam", beam_sum):
                return self.result()
            self._progress(0)

            sample_overlap = self.do_overlap_correction(sample_clean, "Sample")
            if not self._continue(FullProcessStage.SAMPLE_OVERLAP, "sample overlap", "sample", sample_clean):
                return self.result()
            self._progress(0)

            beam_overlap = self.do_overlap_correction(beam_clean, "OpenBeam")
            if not self._continue(FullProcessStage.OPEN_BEAM_OVERLAP, "open‑beam overlap", "open_beam", beam_clean):
                return self.result()
            self._progress(0)

            self.do_normalisation(sample_overlap, beam_overlap)
            if not self._continue(FullProcessStage.NORMALISATION, "normalisation", None, None):
                return self.result()
            self._progress(0)
            self._message("=== <b>Full Process Completed Successfully</b> ===")
        except _StageAbort:
            return self.result()
        except _StageOperationFailure as exc:
            self._fail_current(exc.error, exc.wrapper_message)
            self._message(f"[ERROR] {exc.wrapper_message}")
        except Exception as exc:
            self._fail_current(exc)
            self._message(f"[ERROR] {exc}")
        return self.result()

    def result(self) -> FullProcessPipelineResult:
        if self._cancelled_stage is not None or not self._parent_running():
            status = PreprocessingStatus.CANCELLED
        elif self._overall_status is PreprocessingStatus.FAILED:
            status = PreprocessingStatus.FAILED
        else:
            status = PreprocessingStatus.SUCCEEDED
        return FullProcessPipelineResult(
            status,
            tuple(self._stages),
            self._failed_stage,
            self._cancelled_stage,
            tuple(self._outputs),
            tuple(self._errors),
            tuple(self._warnings),
        )

    def maybe_do_summation(self, folder: str, label: str) -> str:
        stage = (
            FullProcessStage.SAMPLE_SUMMATION
            if label == "Sample"
            else FullProcessStage.OPEN_BEAM_SUMMATION
        )
        branch = "sample" if label == "Sample" else "open_beam"
        if not self._parent_running():
            return folder
        builder = self._begin(stage, branch, folder)
        discovery = discover_full_process_summation(folder)
        short = _short_path(folder)
        if not discovery.should_sum:
            self._message(
                f"0_sumation_{label}: No subfolders found in '\\{short}'. "
                "Skipping Summation."
            )
            builder.outcome = FullProcessStageOutcome.SKIPPED
            builder.propagation_folder = folder
            self._record(builder)
            return folder

        original_name = os.path.basename(folder.rstrip(os.sep))
        out_folder = os.path.join(self.output_folder, f"0_summed_{original_name}")
        os.makedirs(out_folder, exist_ok=True)
        builder.artifact_folder = out_folder
        self._message(
            f"0_sumation_{label}: Found {len(discovery.folders)} subfolder(s) in "
            f"'\\{short}'. Performing Summation..."
        )
        runs = []
        for run_folder in discovery.folders:
            run = load_full_process_run(
                run_folder,
                progress_callback=self._load_progress,
                message_callback=self._message,
            )
            if run.frames:
                runs.append(run)
        if not runs:
            raise RuntimeError(
                f"0_summation_{label}: no valid FITS/TIFF images were found "
                f"in subfolders of '\\{short}'."
            )
        token = self._start_operation(stage)
        try:
            operation_result = sum_loaded_image_runs(
                runs,
                f"summed_{original_name}",
                out_folder,
                progress_callback=self._progress,
                message_callback=self._message,
                cancellation_check=token.is_set,
            )
        except Exception as exc:
            self._message(f"[FATAL] Summation aborted: {exc}")
            raise _StageOperationFailure(
                exc, f"0_summation_{label} failed."
            ) from exc
        finally:
            self._finish_operation(stage, token)
            self._stage_completed()
        builder.operation_result = operation_result
        if operation_result.status is not PreprocessingStatus.SUCCEEDED:
            for error in operation_result.errors:
                self._message(f"[FATAL] Summation aborted: {error}")
            outcome = self._operation_outcome(operation_result)
            builder.outcome = outcome
            builder.propagation_folder = folder
            self._record(builder)
            self._abort_for_result(stage, operation_result, f"0_summation_{label} failed.")
        if self._parent_running():
            self._message(
                f"<b>0_sumation_{label} complete</b>, saved at: "
                f"<b>\\{_short_path(out_folder, levels=3)}</b>"
            )
            builder.propagation_folder = out_folder
        else:
            builder.propagation_folder = folder
            self._cancelled_stage = stage
        builder.outcome = FullProcessStageOutcome.SUCCEEDED
        self._record(builder)
        return builder.propagation_folder or folder

    def do_outlier_removal(self, folder: str, label: str) -> str:
        stage = (
            FullProcessStage.SAMPLE_CLEAN
            if label == "Sample"
            else FullProcessStage.OPEN_BEAM_CLEAN
        )
        branch = "sample" if label == "Sample" else "open_beam"
        if not self._parent_running():
            return folder
        builder = self._begin(stage, branch, folder)
        short = _short_path(folder)
        self._message(f"1_clean_{label}: Starting Outlier Removal on \\{short}...")
        run = load_full_process_run(
            folder,
            progress_callback=self._load_progress,
            message_callback=self._message,
        )
        if not run.frames:
            raise RuntimeError(f"1_clean_{label}: no images found in \\{short}.")
        if run.load_errors:
            raise RuntimeError(
                f"1_clean_{label}: one or more input frames could not be loaded."
            )
        original_name = os.path.basename(folder.rstrip(os.sep))
        out_folder = os.path.join(self.output_folder, f"1_cleaned_{original_name}")
        os.makedirs(out_folder, exist_ok=True)
        builder.artifact_folder = out_folder
        token = self._start_operation(stage)
        try:
            operation_result = clean_loaded_image_runs(
                [run],
                out_folder,
                f"cleaned_{original_name}",
                progress_callback=self._progress,
                message_callback=self._message,
                cancellation_check=token.is_set,
            )
        except Exception as exc:
            raise _StageOperationFailure(
                exc, f"1_clean_{label} failed or skipped frames."
            ) from exc
        finally:
            self._finish_operation(stage, token)
            self._stage_completed()
        builder.operation_result = operation_result
        if operation_result.status is not PreprocessingStatus.SUCCEEDED:
            builder.outcome = self._operation_outcome(operation_result)
            builder.propagation_folder = folder
            self._record(builder)
            self._abort_for_result(
                stage,
                operation_result,
                f"1_clean_{label} failed or skipped frames.",
            )
        if self._parent_running():
            self._message(
                f"<b>1_clean_{label} complete</b>, saved at: "
                f"<b>\\{_short_path(out_folder)}</b>"
            )
            builder.propagation_folder = out_folder
        else:
            builder.propagation_folder = folder
            self._cancelled_stage = stage
        builder.outcome = FullProcessStageOutcome.SUCCEEDED
        self._record(builder)
        return builder.propagation_folder or folder

    def do_overlap_correction(self, folder: str, label: str) -> str:
        stage = (
            FullProcessStage.SAMPLE_OVERLAP
            if label == "Sample"
            else FullProcessStage.OPEN_BEAM_OVERLAP
        )
        branch = "sample" if label == "Sample" else "open_beam"
        if not self._parent_running():
            return folder
        builder = self._begin(stage, branch, folder)
        short = _short_path(folder)
        self._message(f"2_correction_{label}: Starting Overlap Correction on \\{short}...")
        run = load_full_process_run(
            folder,
            progress_callback=self._load_progress,
            message_callback=self._message,
        )
        if not run.frames:
            raise RuntimeError(f"2_correction_{label}: no images found in \\{short}.")
        if run.load_errors:
            raise RuntimeError(
                f"2_correction_{label}: one or more input frames could not be loaded."
            )
        try:
            files = os.listdir(folder)
        except Exception as exc:
            raise RuntimeError(f"2_correction_{label}: cannot access \\{short}: {exc}") from exc
        spectra_path = next(
            (os.path.join(folder, name) for name in files if name.endswith("_Spectra.txt")),
            None,
        )
        spectra = None
        if spectra_path:
            try:
                spectra = np.loadtxt(spectra_path)
            except Exception as exc:
                self._message(f"2_correction_{label}: Error loading Spectra: {exc}")
        shutter_path = next(
            (os.path.join(folder, name) for name in files if name.endswith("_ShutterCount.txt")),
            None,
        )
        shutter = None
        if shutter_path:
            try:
                counts = np.loadtxt(shutter_path)
                shutter = counts[counts != 0]
            except Exception as exc:
                self._message(f"2_correction_{label}: Error loading ShutterCount: {exc}")
        if spectra is None or shutter is None:
            raise RuntimeError(
                f"2_correction_{label}: Spectra or ShutterCount is missing; "
                "Full Process requires overlap correction."
            )
        original_name = os.path.basename(folder.rstrip(os.sep))
        out_folder = os.path.join(self.output_folder, f"2_corrected_{original_name}")
        os.makedirs(out_folder, exist_ok=True)
        builder.artifact_folder = out_folder
        token = self._start_operation(stage)
        try:
            operation_result = correct_loaded_image_run(
                run,
                spectra,
                shutter,
                f"corrected_{original_name}",
                out_folder,
                progress_callback=self._progress,
                message_callback=self._message,
                cancellation_check=token.is_set,
                fatal_error_callback=lambda exc: self._message(
                    f"Error in OverlapCorrectionWorker: {exc}"
                ),
            )
        except Exception as exc:
            self._message(f"Error in OverlapCorrectionWorker: {exc}")
            self._message("Overlap Correction did not complete successfully.")
            raise _StageOperationFailure(
                exc, f"2_correction_{label} failed."
            ) from exc
        finally:
            self._finish_operation(stage, token)
            self._stage_completed()
        builder.operation_result = operation_result
        if operation_result.status is not PreprocessingStatus.SUCCEEDED:
            builder.outcome = self._operation_outcome(operation_result)
            builder.propagation_folder = folder
            self._record(builder)
            self._abort_for_result(stage, operation_result, f"2_correction_{label} failed.")
        if self._parent_running():
            self._message(
                f"<b>2_correction_{label} complete</b>, saved at: "
                f"<b>\\{_short_path(out_folder, levels=3)}</b>"
            )
            builder.propagation_folder = out_folder
        else:
            builder.propagation_folder = folder
            self._cancelled_stage = stage
        builder.outcome = FullProcessStageOutcome.SUCCEEDED
        self._record(builder)
        return builder.propagation_folder or folder

    def do_normalisation(self, sample_folder: str, open_beam_folder: str) -> None:
        stage = FullProcessStage.NORMALISATION
        if not self._parent_running():
            return
        builder = self._begin(stage, None, sample_folder)
        builder.secondary_input_folder = open_beam_folder
        self._message(
            f"3_normalisation: Sample='{_short_path(sample_folder)}', "
            f"OpenBeam='{_short_path(open_beam_folder)}'"
        )
        sample = load_full_process_run(
            sample_folder,
            progress_callback=self._load_progress,
            message_callback=self._message,
        )
        beam = load_full_process_run(
            open_beam_folder,
            progress_callback=self._load_progress,
            message_callback=self._message,
        )
        if not sample.frames or not beam.frames:
            raise RuntimeError(
                "Normalisation cannot start because sample or open beam has no images."
            )
        if sample.load_errors or beam.load_errors:
            raise RuntimeError(
                "Normalisation cannot start because one or more input frames "
                "could not be loaded."
            )
        out_folder = os.path.join(self.output_folder, "3_normalised_original")
        os.makedirs(out_folder, exist_ok=True)
        builder.artifact_folder = out_folder
        token = self._start_operation(stage)
        operation_result = None
        operation_error: Exception | None = None
        try:
            operation_result = normalise_loaded_image_runs(
                [sample],
                [beam],
                out_folder,
                "normalised",
                self.window_half,
                self.adjacent_sum,
                progress_callback=self._progress,
                message_callback=self._message,
                cancellation_check=token.is_set,
                run_complete_callback=lambda _run_idx: self._normalisation_pacing(5),
                fatal_error_callback=lambda exc: self._message(
                    f"Fatal error in normalisation: {exc}"
                ),
            )
        except Exception as exc:
            operation_error = exc
            self._message(f"Fatal error in normalisation: {exc}")
        finally:
            self._finish_operation(stage, token)
        builder.operation_result = operation_result
        try:
            self._normalisation_finished()
        except Exception as exc:
            if operation_error is None:
                operation_error = exc
        if operation_error is not None:
            raise _StageOperationFailure(
                operation_error,
                "3_normalisation failed or skipped one or more frames.",
            ) from operation_error
        if operation_result is None:
            error = RuntimeError("Normalisation service returned no operation result.")
            raise _StageOperationFailure(
                error,
                "3_normalisation failed or skipped one or more frames.",
            ) from error
        if operation_result.status is not PreprocessingStatus.SUCCEEDED:
            builder.outcome = self._operation_outcome(operation_result)
            builder.propagation_folder = None
            self._record(builder)
            self._abort_for_result(
                stage,
                operation_result,
                "3_normalisation failed or skipped one or more frames.",
            )
        if self._parent_running():
            self._message(
                f"<b>3_normalisation complete</b>, saved at: "
                f"<b>\\{_short_path(out_folder, levels=3)}</b>"
            )
            builder.propagation_folder = out_folder
        else:
            builder.propagation_folder = None
            self._cancelled_stage = stage
        builder.outcome = FullProcessStageOutcome.SUCCEEDED
        self._record(builder)

    def _begin(
        self, stage: FullProcessStage, branch: str | None, input_folder: str | None
    ) -> _StageBuilder:
        builder = _StageBuilder(stage, branch, input_folder)
        self._current = builder
        return builder

    def _start_operation(self, stage: FullProcessStage) -> CancellationToken:
        token = threading.Event()
        self._operation_started(stage, token)
        return token

    def _finish_operation(
        self, stage: FullProcessStage, token: CancellationToken
    ) -> None:
        self._operation_finished(stage, token)

    def _record(self, builder: _StageBuilder) -> None:
        if builder.recorded:
            return
        builder.recorded = True
        self._stages.append(
            FullProcessStageResult(
                builder.stage,
                builder.branch,
                builder.outcome,
                builder.input_folder,
                builder.artifact_folder,
                builder.propagation_folder,
                builder.operation_result,
                builder.secondary_input_folder,
            )
        )
        if builder.operation_result is not None:
            self._outputs.extend(builder.operation_result.outputs)
            self._errors.extend(
                FullProcessDiagnostic(builder.stage, message)
                for message in builder.operation_result.errors
            )
            self._warnings.extend(
                FullProcessDiagnostic(builder.stage, message)
                for message in builder.operation_result.warnings
            )
        self._current = None

    def _fail_current(
        self, exc: Exception, wrapper_message: str | None = None
    ) -> None:
        builder = self._current
        if builder is None or builder.recorded:
            self._overall_status = PreprocessingStatus.FAILED
            stage = builder.stage if builder is not None else None
            self._errors.append(FullProcessDiagnostic(stage, str(exc)))
            if wrapper_message is not None:
                self._errors.append(FullProcessDiagnostic(stage, wrapper_message))
            if stage is not None:
                self._failed_stage = stage
                if not self._parent_running():
                    self._cancelled_stage = stage
            return
        builder.outcome = FullProcessStageOutcome.FAILED
        builder.propagation_folder = None
        self._overall_status = PreprocessingStatus.FAILED
        self._failed_stage = builder.stage
        if not self._parent_running():
            self._cancelled_stage = builder.stage
        self._record(builder)
        self._errors.append(FullProcessDiagnostic(builder.stage, str(exc)))
        if wrapper_message is not None:
            self._errors.append(FullProcessDiagnostic(builder.stage, wrapper_message))

    def _operation_outcome(
        self, result: PreprocessingOperationResult
    ) -> FullProcessStageOutcome:
        if result.status is PreprocessingStatus.CANCELLED:
            return FullProcessStageOutcome.CANCELLED
        return FullProcessStageOutcome.FAILED

    def _abort_for_result(
        self,
        stage: FullProcessStage,
        result: PreprocessingOperationResult,
        wrapper_message: str,
    ) -> None:
        if result.status is PreprocessingStatus.CANCELLED or not self._parent_running():
            self._cancelled_stage = stage
            self._overall_status = PreprocessingStatus.CANCELLED
        if result.status is PreprocessingStatus.FAILED:
            self._failed_stage = stage
            self._overall_status = PreprocessingStatus.FAILED
            if self._cancelled_stage is not None:
                self._overall_status = PreprocessingStatus.CANCELLED
        self._errors.append(FullProcessDiagnostic(stage, wrapper_message))
        self._message(f"[ERROR] {wrapper_message}")
        raise _StageAbort(wrapper_message, stage)

    def _continue(
        self,
        stage: FullProcessStage,
        phase: str,
        branch: str | None,
        input_folder: str | None,
    ) -> bool:
        if self._parent_running():
            return True
        self._message(f"Stopped during {phase}.")
        self._cancelled_stage = stage
        self._overall_status = PreprocessingStatus.CANCELLED
        if self._current is None and (
            not self._stages or self._stages[-1].stage is not stage
        ):
            builder = _StageBuilder(
                stage,
                branch,
                input_folder,
                propagation_folder=input_folder,
                outcome=FullProcessStageOutcome.CANCELLED,
            )
            self._record(builder)
        return False


def run_full_process(
    sample_folder: str,
    open_beam_folder: str,
    output_folder: str,
    base_name: str,
    window_half: int,
    adjacent_sum: int,
    **callbacks,
) -> FullProcessPipelineResult:
    """Convenience entry point for one headless Full Process execution."""
    return FullProcessPipeline(
        sample_folder,
        open_beam_folder,
        output_folder,
        base_name,
        window_half,
        adjacent_sum,
        **callbacks,
    ).run()
