"""Worker threads for preprocessing and filtering tasks."""

import gc
import os
import shutil
from collections.abc import Sequence

import numpy as np
import pandas as pd
import psutil
from PyQt5.QtCore import Qt, QEventLoop, QThread, pyqtSignal

from .batch import get_raden_tiff_stack_info, load_image_file, write_fits_image_file
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
from ..services.preprocessing_layout import discover_full_process_summation
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
    # Signals for messages, progress updates and final completion
    message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    load_progress_updated = pyqtSignal(int) 
    finished = pyqtSignal()

    def __init__(self, sample_folder: str, open_beam_folder: str, output_folder: str,
                 base_name: str, window_half: int, adjacent_sum: int):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True  # Flag to track whether the user requested a stop
        self.succeeded = False
        self._active_child = None
        
    def get_short_path(self, full_path, levels=2):
        """
        Returns the last `levels` parts of a path.
        
        Args:
            full_path (str): The full file or folder path.
            levels (int): How many trailing parts to keep. Default is 2.
            
        Returns:
            str: The shortened path.
        """
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= levels:
            short_path = os.path.join(*path_parts[-levels:])
        else:
            short_path = normalized_path
        return short_path


    def run(self):
        try:
            self.message.emit("=== <b>Full Process Pipeline Started</b> ===")

            # 1) Summation ---------------------------------------------------
            sample_after_sum = self.maybe_do_summation(self.sample_folder, "Sample")
            if not self._continue("sample summation"): return
            self.progress_updated.emit(0)       # reset for next stage

            openbeam_after_sum = self.maybe_do_summation(self.open_beam_folder, "OpenBeam")
            if not self._continue("open‑beam summation"): return
            self.progress_updated.emit(0)

            # 2) Clean sample ----------------------------------------------
            sample_after_clean = self.do_outlier_removal(sample_after_sum, "Sample")
            if not self._continue("sample cleaning"): return
            self.progress_updated.emit(0)

            # 3) Clean OB ---------------------------------------------------
            openbeam_after_clean = self.do_outlier_removal(openbeam_after_sum, "OpenBeam")
            if not self._continue("open‑beam cleaning"): return
            self.progress_updated.emit(0)

            # 4) Overlap sample --------------------------------------------
            sample_after_overlap = self.do_overlap_correction(sample_after_clean, "Sample")
            if not self._continue("sample overlap"): return
            self.progress_updated.emit(0)

            # 5) Overlap OB -------------------------------------------------
            openbeam_after_overlap = self.do_overlap_correction(openbeam_after_clean, "OpenBeam")
            if not self._continue("open‑beam overlap"): return
            self.progress_updated.emit(0)

            # 6) Normalisation ---------------------------------------------
            self.do_normalisation(sample_after_overlap, openbeam_after_overlap)
            if not self._continue("normalisation"): return
            self.progress_updated.emit(0)

            self.succeeded = True
            self.message.emit("=== <b>Full Process Completed Successfully</b> ===")

        except Exception as exc:
            self.succeeded = False
            self.message.emit(f"[ERROR] {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    # ------------------------------------------------------------------
    # HELPER: continue or exit early
    # ------------------------------------------------------------------
    def _continue(self, phase):
        if not self._is_running:
            self.message.emit(f"Stopped during {phase}.")
            return False
        return True
    
    def _early_exit(self, reason: str):
        """Emits a final message and returns quickly."""
        self.message.emit(reason)
        # (The final self.finished.emit() is in the 'finally' block of run().) 
        
    def stop(self):
        """
        Called from the main UI when the user clicks 'Stop Full Process'.
        Sets _is_running=False so the while loops in each step can stop the worker.
        """
        self._is_running = False
        child = self._active_child
        if child is not None and hasattr(child, "stop"):
            child.stop()
        self.message.emit("FullProcessWorker: Stop signal received.")

    # ---------------------------------------------------------------
    # Summation Step (conditionally skipped if no subfolders)
    # ---------------------------------------------------------------
    def maybe_do_summation(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
    
        # Check for subfolders
        discovery = discover_full_process_summation(folder)
        subfolders = discovery.folders
        
        short_path = self.get_short_path(folder, levels=2)
    
        if not discovery.should_sum:
            self.message.emit(f"0_sumation_{label}: No subfolders found in '\\{short_path}'. Skipping Summation.")
            return folder
    
        # Get the original folder name (e.g., "sample_data")
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Build the output folder: e.g., "0_summed_sample_data"
        summation_output = os.path.join(self.output_folder, f"0_summed_{original_folder_name}")
        os.makedirs(summation_output, exist_ok=True)
    
        self.message.emit(f"0_sumation_{label}: Found {len(subfolders)} subfolder(s) in '\\{short_path}'. Performing Summation...")
        runs = []
        for sf in subfolders:
            r = self.load_run_dict(sf)
            if r.get("images"):
                runs.append(r)
    
        if not runs:
            raise RuntimeError(
                f"0_summation_{label}: no valid FITS/TIFF images were found "
                f"in subfolders of '\\{short_path}'."
            )
    
        # Create a modified base name for image naming (e.g., "summed_sample_data")
        modified_base_name = f"summed_{original_folder_name}"
        worker = SummationWorker(runs, modified_base_name, summation_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
               
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError(f"0_summation_{label} failed.")
    
        # Free memory from runs
        del runs
        import gc
        gc.collect()
        
        summation_output_short = self.get_short_path(summation_output, levels = 3)
    
        if self._is_running:
            self.message.emit(f"<b>0_sumation_{label} complete</b>, saved at: <b>\\{summation_output_short}</b>")
            return summation_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Outlier Removal (Clean) Step
    # ---------------------------------------------------------------

    def do_outlier_removal(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
        
        short_path = self.get_short_path(folder, levels=2)
    
        self.message.emit(f"1_clean_{label}: Starting Outlier Removal on \\{short_path}...")
        run = self.load_run_dict(folder)
        if not run.get("images"):
            raise RuntimeError(
                f"1_clean_{label}: no images found in \\{short_path}."
            )
        if run.get("load_errors"):
            raise RuntimeError(
                f"1_clean_{label}: one or more input frames could not be loaded."
            )
    
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Output folder: e.g., "1_cleaned_sample_data" or "1_cleaned_openbeam_data"
        outlier_output = os.path.join(self.output_folder, f"1_cleaned_{original_folder_name}")
        os.makedirs(outlier_output, exist_ok=True)
    
        # Create a base name for cleaned image names (e.g., "cleaned_sample_data")
        modified_base_name = f"cleaned_{original_folder_name}"
        worker = OutlierFilteringWorker([run], outlier_output, modified_base_name)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(25 + v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not getattr(worker, "succeeded", True):
            raise RuntimeError(f"1_clean_{label} failed or skipped frames.")
    
        del run
        import gc
        gc.collect()
        
        short_path = self.get_short_path(outlier_output, levels=2)
    
        if self._is_running:
            self.message.emit(f"<b>1_clean_{label} complete</b>, saved at: <b>\\{short_path}</b>")
            return outlier_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Overlap Correction Step
    # ---------------------------------------------------------------

    def do_overlap_correction(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
        
        short_path = self.get_short_path(folder, levels=2)
    
        self.message.emit(f"2_correction_{label}: Starting Overlap Correction on \\{short_path}...")
        run = self.load_run_dict(folder)
        if not run.get("images"):
            raise RuntimeError(
                f"2_correction_{label}: no images found in \\{short_path}."
            )
        if run.get("load_errors"):
            raise RuntimeError(
                f"2_correction_{label}: one or more input frames could not be loaded."
            )
    
        try:
            all_files = os.listdir(folder)
        except Exception as e:
            raise RuntimeError(
                f"2_correction_{label}: cannot access \\{short_path}: {e}"
            ) from e
    
        # Load additional data (Spectra and ShutterCount)
        spectra_file = next((os.path.join(folder, f) for f in all_files if f.endswith("_Spectra.txt")), None)
        # short_spectra=self.get_short_path(spectra_file, levels=2)
        if spectra_file:
            try:
                run["spectra"] = np.loadtxt(spectra_file)
                # self.message.emit(f"2_correction_{label}: Loaded Spectra from {short_spectra}")
            except Exception as e:
                self.message.emit(f"2_correction_{label}: Error loading Spectra: {e}")
                run["spectra"] = None
        else:
            run["spectra"] = None
    
        shutter_file = next((os.path.join(folder, f) for f in all_files if f.endswith("_ShutterCount.txt")), None)
        # short_shutter=self.get_short_path(shutter_file, levels=2)
        if shutter_file:
            try:
                sc_data = np.loadtxt(shutter_file)
                run["shutter_count"] = sc_data[sc_data != 0]
                # self.message.emit(f"2_correction_{label}: Loaded ShutterCount from {short_shutter}")
            except Exception as e:
                self.message.emit(f"2_correction_{label}: Error loading ShutterCount: {e}")
                run["shutter_count"] = None
        else:
            run["shutter_count"] = None
    
        if run["spectra"] is None or run["shutter_count"] is None:
            raise RuntimeError(
                f"2_correction_{label}: Spectra or ShutterCount is missing; "
                "Full Process requires overlap correction."
            )
    
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Output folder: e.g., "2_corrected_sample_data"
        overlap_output = os.path.join(self.output_folder, f"2_corrected_{original_folder_name}")
        os.makedirs(overlap_output, exist_ok=True)
    
        # Create a base name to be used for image naming (e.g., "corrected_sample_data")
        modified_base_name = f"corrected_{original_folder_name}"
        worker = OverlapCorrectionWorker(run, modified_base_name, overlap_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(50 + v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError(f"2_correction_{label} failed.")
    
        del run
        import gc
        gc.collect()
        overlap_output_short=self.get_short_path(overlap_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>2_correction_{label} complete</b>, saved at: <b>\\{overlap_output_short}</b>")
            return overlap_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Normalisation Step
    # ---------------------------------------------------------------

    def do_normalisation(self, sample_folder: str, openbeam_folder: str):
        if not self._is_running:
            return
        short_path_sample = self.get_short_path(sample_folder, levels=2)
        short_path_ob = self.get_short_path(openbeam_folder, levels=2)
    
        self.message.emit(
            f"3_normalisation: Sample='{short_path_sample}', OpenBeam='{short_path_ob}'"
        )
    
        sample_run = self.load_run_dict(sample_folder)
        openbeam_run = self.load_run_dict(openbeam_folder)
        if (not sample_run.get("images")) or (not openbeam_run.get("images")):
            raise RuntimeError(
                "Normalisation cannot start because sample or open beam has no images."
            )
        if sample_run.get("load_errors") or openbeam_run.get("load_errors"):
            raise RuntimeError(
                "Normalisation cannot start because one or more input frames "
                "could not be loaded."
            )
    
        # Set output folder to a fixed name for the merged result
        normalised_output = os.path.join(self.output_folder, "3_normalised_original")
        os.makedirs(normalised_output, exist_ok=True)
    
        # Base name for normalised images (here using "normalised" as the prefix)
        modified_base_name = "normalised"
        worker = NormalisationWorker(
            [sample_run],
            [openbeam_run],
            normalised_output,
            modified_base_name,
            self.window_half,
            self.adjacent_sum
        )
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(75 + v // 4))
        worker.progress_updated.connect(self.progress_updated.emit, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError("3_normalisation failed or skipped one or more frames.")
    
        del sample_run, openbeam_run
        import gc
        gc.collect()
        
        normalised_output_short=self.get_short_path(normalised_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>3_normalisation complete</b>, saved at: <b>\\{normalised_output_short}</b>")

  
    def load_run_dict(self, folder: str) -> dict:
        """
        Scan a folder for FITS/TIFF files and use the final underscore-delimited
        stem component as the frame suffix. The suffix need not contain five
        digits.

        Returns a dictionary:
            { 'folder_path': folder, 'images': {suffix: data} }.
        Also emits loading progress via load_progress_updated.
        """
        run = {"folder_path": folder, "images": {}, "load_errors": []}
        if not os.path.isdir(folder):
            self.message.emit(f"Folder not found: {folder}")
            return run
        
        short_path = self.get_short_path(folder, levels=2)

        try:
            image_files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".fits", ".fit", ".tiff", ".tif"))
            ]
            total_files = len(image_files)
            processed_files = 0

            for f in image_files:
                processed_files += 1
                stem = os.path.splitext(f)[0]
                suffix = stem.rsplit("_", 1)[-1].strip()
                if not suffix:
                    error = f"File '{f}' has an empty frame suffix."
                    run["load_errors"].append(error)
                    self.message.emit(error)
                    self.load_progress_updated.emit(
                        int((processed_files / total_files) * 100)
                    )
                    continue
                if suffix in run["images"]:
                    error = (
                        f"File '{f}' duplicates frame suffix '{suffix}'. "
                        "Frame names must be unique."
                    )
                    run["load_errors"].append(error)
                    self.message.emit(error)
                    self.load_progress_updated.emit(
                        int((processed_files / total_files) * 100)
                    )
                    continue
                try:
                    path = os.path.join(folder, f)
                    data = load_image_file(path)
                    run["images"][suffix] = data.astype(np.float32)
                except Exception as e:
                    error = f"Error loading file {f} in \\{short_path}: {e}"
                    run["load_errors"].append(error)
                    self.message.emit(error)
                # Update loading progress after processing each file
                self.load_progress_updated.emit(int((processed_files / total_files) * 100))
        except Exception as e:
            error = f"Error reading folder \\{short_path}: {e}"
            run["load_errors"].append(error)
            self.message.emit(error)
        return run

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
