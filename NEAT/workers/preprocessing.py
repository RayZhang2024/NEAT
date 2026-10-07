"""Worker threads for preprocessing and filtering tasks."""

import gc
import os
import shutil
from collections.abc import Sequence

import numpy as np
import pandas as pd
import psutil
from PyQt5.QtCore import QThread, pyqtSignal

from .batch import get_raden_tiff_stack_info
from ..domain import (
    LoadedImageRun,
    PreprocessingOperationResult,
    PreprocessingStatus,
)
from ..services.preprocessing_clean import clean_loaded_image_runs, positive_neighbor_mean
from ..services.preprocessing_filtering import (
    copy_filtering_related_files,
    filter_loaded_image_runs,
)
from ..services.preprocessing_full_process import (
    FullProcessDiagnostic,
    FullProcessPipeline,
    FullProcessPipelineResult,
    FullProcessStage,
    load_full_process_run,
)
from ..services.preprocessing_normalisation import (
    NORMALISATION_ADJACENT_RANGE,
    NORMALISATION_WINDOW_HALF_RANGE,
    copy_normalisation_related_files,
    normalise_loaded_image_runs,
    read_normalisation_shutter_count,
    validate_normalisation_windows,
)
from ..services.preprocessing_normalisation_raden import (
    copy_raden_sidecars,
    normalise_raden_frame,
    normalise_raden_tiff_stack,
    raden_pulse_count,
    read_raden_frame,
    same_raden_tof_axis,
    validate_raden_stacks,
)
from ..services.preprocessing_overlap import correct_loaded_image_run
from ..services.preprocessing_summation import sum_loaded_image_runs



class OutlierFilteringWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, image_runs, output_folder, base_name):
        super().__init__()
        self.image_runs = image_runs
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.report_path = os.path.join(output_folder, f"{base_name}_outlier_report.csv")
        self.result: PreprocessingOperationResult | None = None

    def run(self):
        try:
            runs = tuple(
                LoadedImageRun(
                    primary_source=run["folder_path"],
                    frames=run["images"],
                    source_folders=run.get("run_folders") or None,
                    load_errors=run.get("load_errors") or (),
                )
                for run in self.image_runs
            )
            self.result = clean_loaded_image_runs(
                runs,
                self.output_folder,
                self.base_name,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                frame_failure_callback=lambda suffix: self.failed_frames.append(suffix),
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"[FATAL] {exc}")
        finally:
            self.finished.emit()

    _positive_neighbor_mean = staticmethod(positive_neighbor_mean)

    def stop(self):
        self._is_running = False

class SummationWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished         = pyqtSignal()
    message          = pyqtSignal(str)

    def __init__(self, summation_image_runs: list, base_name: str, output_folder: str):
        super().__init__()
        self.summation_image_runs = summation_image_runs
        self.base_name            = base_name
        self.output_folder        = output_folder
        self._is_running          = True     # Cooperative‑cancel flag
        self.succeeded            = False
        self.result: PreprocessingOperationResult | None = None

    # ───────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────
    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received – cancelling at next safe point.")

    # ───────────────────────────────────────────────────────────────
    # Thread entry
    # ───────────────────────────────────────────────────────────────
    def run(self):
        try:
            runs = tuple(
                _loaded_image_run_from_legacy(run)
                for run in self.summation_image_runs
            )
            self.result = sum_loaded_image_runs(
                runs,
                self.base_name,
                self.output_folder,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
            for error in self.result.errors:
                self.message.emit(f"[FATAL] Summation aborted: {error}")
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"[FATAL] Summation aborted: {exc}")
        finally:
            gc.collect()
            self.finished.emit()


def _loaded_image_run_from_legacy(run: dict) -> LoadedImageRun:
    """Translate a historical GUI-worker dictionary at the adapter boundary."""
    primary_source = run.get("folder_path")
    if not isinstance(primary_source, str) or not primary_source:
        raise ValueError("Summation run has no valid source folder.")
    if "run_folders" in run and run["run_folders"] is None:
        raise ValueError("Summation run_folders must contain valid source folders.")
    source_folders = run.get("run_folders", [primary_source])
    return LoadedImageRun(
        primary_source=primary_source,
        frames=run.get("images", {}),
        source_folders=source_folders,
        load_errors=run.get("load_errors") or (),
    )

class OverlapCorrectionWorker(QThread):
    """Qt compatibility adapter for headless overlap correction."""

    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, run, base_name, output_folder):
        super().__init__()
        self.run_data = run
        self.base_name = base_name
        self.output_folder = output_folder
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.result: PreprocessingOperationResult | None = None

    def run(self):
        try:
            run = LoadedImageRun(
                primary_source=self.run_data["folder_path"],
                frames=self.run_data["images"],
                load_errors=self.run_data.get("load_errors") or (),
            )
            self.result = correct_loaded_image_run(
                run,
                self.run_data["spectra"],
                self.run_data["shutter_count"],
                self.base_name,
                self.output_folder,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                frame_failure_callback=lambda suffix: self.failed_frames.append(suffix),
                fatal_error_callback=lambda exc: self.message.emit(
                    f"Error in OverlapCorrectionWorker: {exc}"
                ),
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED,
                0,
                expected_count=len(self.run_data.get("images", {})),
                errors=(str(exc),),
            )
            self.message.emit(f"Error in OverlapCorrectionWorker: {exc}")
            self.message.emit("Overlap Correction did not complete successfully.")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Overlap Correction process.")

class _LegacyNormalisationRuns(Sequence):
    """Adapt one run at a time without retaining cleared frame arrays."""

    def __init__(self, runs):
        self._runs = runs

    def __len__(self):
        return len(self._runs)

    def __getitem__(self, index):
        run = self._runs[index]
        return LoadedImageRun(
            primary_source=run["folder_path"],
            frames=run["images"],
            load_errors=run.get("load_errors") or (),
        )

class NormalisationWorker(QThread):
    """Qt and legacy mutable-state adapter for classic normalisation."""

    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(
        self,
        normalisation_image_runs,
        normalisation_open_beam_runs,
        output_folder,
        base_name,
        window_half,
        adjacent_sum,
    ):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.normalisation_image_runs = normalisation_image_runs
        self.normalisation_open_beam_runs = normalisation_open_beam_runs
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.result: PreprocessingOperationResult | None = None

    _read_shutter_count = staticmethod(read_normalisation_shutter_count)

    def _delete_sample_suffix(self, run_idx, suffix):
        del self.normalisation_image_runs[run_idx - 1]["images"][suffix]

    def _cleanup_sample_run(self, run_idx):
        self.normalisation_image_runs[run_idx - 1]["images"].clear()

    def run(self):
        try:
            sample_runs = _LegacyNormalisationRuns(self.normalisation_image_runs)
            open_beam_runs = _LegacyNormalisationRuns(
                self.normalisation_open_beam_runs
            )
            self.result = normalise_loaded_image_runs(
                sample_runs,
                open_beam_runs,
                self.output_folder,
                self.base_name,
                self.window_half,
                self.adjacent_sum,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                failed_frame_callback=self.failed_frames.append,
                written_frame_callback=self._delete_sample_suffix,
                run_cleanup_callback=self._cleanup_sample_run,
                run_collect_callback=lambda _run_idx: gc.collect(),
                run_complete_callback=lambda _run_idx: QThread.sleep(5),
                fatal_error_callback=lambda exc: self.message.emit(
                    f"Fatal error in normalisation: {exc}"
                ),
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"Fatal error in normalisation: {exc}")
        finally:
            gc.collect()
            proc = psutil.Process(os.getpid())
            memMB = proc.memory_info().rss / (1024.**2)
            self.message.emit(f"<b>Final memory usage:</b> {memMB:.1f} MB")
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Normalisation process.")

    def copy_related_files(self, run_idx, data_run):
        copy_normalisation_related_files(
            run_idx,
            data_run["folder_path"],
            self.output_folder,
            message_callback=self.message.emit,
        )

class FullProcessWorker(QThread):
    """Single-thread Qt adapter for the headless Full Process pipeline."""

    message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    load_progress_updated = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        sample_folder: str,
        open_beam_folder: str,
        output_folder: str,
        base_name: str,
        window_half: int,
        adjacent_sum: int,
    ):
        super().__init__()
        self.window_half, self.adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True
        self.succeeded = False
        self.result: FullProcessPipelineResult | None = None
        self._active_child = None
        self._active_operation_token = None
        self._active_operation_stage = None
        self._pipeline = None

    def get_short_path(self, full_path, levels=2):
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= levels:
            return os.path.join(*path_parts[-levels:])
        return normalized_path

    def _make_pipeline(self):
        return FullProcessPipeline(
            self.sample_folder,
            self.open_beam_folder,
            self.output_folder,
            self.base_name,
            self.window_half,
            self.adjacent_sum,
            message_callback=self.message.emit,
            progress_callback=self.progress_updated.emit,
            load_progress_callback=self.load_progress_updated.emit,
            parent_running=lambda: self._is_running,
            operation_started=self._operation_started,
            operation_finished=self._operation_finished,
            stage_completed=gc.collect,
            normalisation_finished=self._normalisation_finished,
            normalisation_pacing=QThread.sleep,
        )

    def _operation_started(self, stage, token):
        self._active_operation_stage = stage
        self._active_operation_token = token

    def _operation_finished(self, stage, token):
        if self._active_operation_token is token:
            self._active_operation_stage = None
            self._active_operation_token = None

    def _normalisation_finished(self):
        gc.collect()
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)
        self.message.emit(f"<b>Final memory usage:</b> {memory_mb:.1f} MB")

    def run(self):
        try:
            self._pipeline = self._make_pipeline()
            self.result = self._pipeline.run()
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
        except Exception as exc:
            self.succeeded = False
            if self._pipeline is not None:
                prior = self._pipeline.result()
                self.result = FullProcessPipelineResult(
                    PreprocessingStatus.FAILED,
                    prior.stages,
                    prior.failed_stage,
                    prior.cancelled_stage,
                    prior.outputs,
                    (*prior.errors, FullProcessDiagnostic(None, str(exc))),
                    prior.warnings,
                )
            else:
                self.result = FullProcessPipelineResult(
                    PreprocessingStatus.FAILED,
                    (),
                    None,
                    None,
                    (),
                    (FullProcessDiagnostic(None, str(exc)),),
                    (),
                )
            self.message.emit(f"[ERROR] {exc}")
        finally:
            self.succeeded = bool(
                self.result is not None
                and self.result.status is PreprocessingStatus.SUCCEEDED
            )
            gc.collect()
            self.finished.emit()

    def stop(self):
        self._is_running = False
        token = self._active_operation_token
        stage = self._active_operation_stage
        if token is not None:
            token.set()
            if stage in (
                FullProcessStage.SAMPLE_SUMMATION,
                FullProcessStage.OPEN_BEAM_SUMMATION,
            ):
                self.message.emit("Stop signal received – cancelling at next safe point.")
            elif stage in (
                FullProcessStage.SAMPLE_OVERLAP,
                FullProcessStage.OPEN_BEAM_OVERLAP,
            ):
                self.message.emit(
                    "Stop signal received. Terminating Overlap Correction process."
                )
            elif stage is FullProcessStage.NORMALISATION:
                self.message.emit(
                    "Stop signal received. Terminating Normalisation process."
                )
        child = self._active_child
        if child is not None and hasattr(child, "stop"):
            child.stop()
        self.message.emit("FullProcessWorker: Stop signal received.")

    def maybe_do_summation(self, folder: str, label: str) -> str:
        return self._make_pipeline().maybe_do_summation(folder, label)

    def do_outlier_removal(self, folder: str, label: str) -> str:
        return self._make_pipeline().do_outlier_removal(folder, label)

    def do_overlap_correction(self, folder: str, label: str) -> str:
        return self._make_pipeline().do_overlap_correction(folder, label)

    def do_normalisation(self, sample_folder: str, openbeam_folder: str):
        return self._make_pipeline().do_normalisation(sample_folder, openbeam_folder)

    def load_run_dict(self, folder: str) -> dict:
        run = load_full_process_run(
            folder,
            progress_callback=self.load_progress_updated.emit,
            message_callback=self.message.emit,
        )
        return {
            "folder_path": run.primary_source,
            "images": dict(run.frames),
            "load_errors": list(run.load_errors),
        }


class FilteringWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, filtering_image_runs, filtering_mask, output_folder, base_name):
        super().__init__()
        self.filtering_image_runs = filtering_image_runs
        self.filtering_mask = filtering_mask
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.result: PreprocessingOperationResult | None = None

    def run(self):
        try:
            runs = tuple(
                LoadedImageRun(
                    primary_source=run["folder_path"],
                    frames=run["images"],
                    load_errors=run.get("load_errors") or (),
                )
                for run in self.filtering_image_runs
            )
            self.result = filter_loaded_image_runs(
                runs,
                self.filtering_mask,
                self.output_folder,
                self.base_name,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                frame_failure_callback=lambda suffix: self.failed_frames.append(suffix),
                validated_mask_callback=self._set_validated_mask,
                summary_path_callback=self._set_summary_path,
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
            if not self.filtering_image_runs or self.filtering_mask is None:
                self.finished.emit()  # Preserve the legacy early-return signal.
                return
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"Error during filtering: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    def _set_validated_mask(self, mask: np.ndarray) -> None:
        self.filtering_mask = mask

    def _set_summary_path(self) -> str:
        self.output_folder_short = self.get_short_path(self.output_folder, levels=2)
        return self.output_folder_short

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Filtering process.")

    @staticmethod
    def get_short_path(full_path, levels=2):
        """Return the final path components for concise progress messages."""
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= levels:
            return os.path.join(*path_parts[-levels:])
        return normalized_path

    def copy_related_files(self, run_idx, data_run):
        """Keep the public sidecar-copy entry point as a service delegate."""
        copy_filtering_related_files(
            run_idx,
            data_run.get("folder_path", None),
            self.output_folder,
            message_callback=self.message.emit,
        )

class RadenNormalisationWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(
        self,
        sample_run,
        open_beam_run,
        output_folder,
        base_name,
        window_half,
        adjacent_sum,
    ):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.sample_run = sample_run
        self.open_beam_run = open_beam_run
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True
        self.succeeded = False
        self.result: PreprocessingOperationResult | None = None

    @staticmethod
    def _pulse_count(info):
        return raden_pulse_count(info)

    @staticmethod
    def _read_frame(tiff, index):
        return read_raden_frame(tiff, index)

    @staticmethod
    def _same_tof_axis(sample_info, open_beam_info):
        return same_raden_tof_axis(sample_info, open_beam_info)

    def _validate(self, sample_info, open_beam_info):
        validate_raden_stacks(sample_info, open_beam_info)

    def _normalise_frame(self, sample_frame, open_beam_sum, frame_count, scale):
        return normalise_raden_frame(
            sample_frame,
            open_beam_sum,
            frame_count,
            self.window_half,
            scale,
        )

    def _copy_sidecars(self, sample_info, output_tiff):
        copy_raden_sidecars(sample_info, output_tiff, self.output_folder)

    def run(self):
        try:
            # The legacy worker creates the output directory before resolving
            # stack metadata; keep that observable ordering at this boundary.
            os.makedirs(self.output_folder, exist_ok=True)
            sample_info = self.sample_run.get("info") or get_raden_tiff_stack_info(self.sample_run["folder_path"])
            open_beam_info = self.open_beam_run.get("info") or get_raden_tiff_stack_info(self.open_beam_run["folder_path"])
            def collect_at_page_boundary(index: int) -> None:
                if index % 100 == 0:
                    gc.collect()

            self.result = normalise_raden_tiff_stack(
                sample_info,
                open_beam_info,
                self.output_folder,
                self.base_name,
                self.window_half,
                self.adjacent_sum,
                cancellation_check=lambda: not self._is_running,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                page_complete_callback=collect_at_page_boundary,
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED

        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                status=PreprocessingStatus.FAILED,
                processed_count=0,
                expected_count=None,
                errors=(str(exc),),
            )
            self.message.emit(f"Fatal error in RADEN normalisation: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating RADEN normalisation process.")


__all__ = [
    "OutlierFilteringWorker",
    "SummationWorker",
    "OverlapCorrectionWorker",
    "NormalisationWorker",
    "RadenNormalisationWorker",
    "FullProcessWorker",
    "FilteringWorker",
    "validate_normalisation_windows",
]
