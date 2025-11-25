"""Preprocessing tab functionality."""

import gc
import glob
import os
import time

import numpy as np
from astropy.io import fits
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QWidget,
    QSlider,
    QLabel,
    QLineEdit,
    QGridLayout,
    QProgressBar,
    QMessageBox,
    QTabWidget,
    QGroupBox,
    QDoubleSpinBox,
    QSplitter,
    QInputDialog,
    QCheckBox,
    QScrollArea,
    QComboBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QSizePolicy,
    QShortcut,
)

from ...workers.batch import ImageLoadWorker, OpenBeamLoadWorker
from ...workers.preprocessing import (
    FilteringWorker,
    FullProcessWorker,
    NormalisationWorker,
    OutlierFilteringWorker,
    OverlapCorrectionWorker,
    SummationWorker,
)


class PreprocessingMixin:
    def setup_preprocessing_tab(self):
        # Main layout for the tab
        main_layout = QVBoxLayout(self.PreProcessingTab)

        # Upper and lower layouts
        upper_layout = QHBoxLayout()
        lower_layout = QVBoxLayout()

        # Summation Section
        summation_group = QGroupBox("Summation")
        summation_layout = QVBoxLayout()

        # "Add" and "Remove" buttons to load and remove folder of images
        summation_buttons_layout = QHBoxLayout()
        self.summation_add_button = QPushButton("Add data")
        self.summation_add_button.clicked.connect(self.add_summation_images)
        # self.summation_remove_button = QPushButton("Remove")
        # self.summation_remove_button.clicked.connect(self.remove_summation_images)
        summation_buttons_layout.addWidget(self.summation_add_button)
        # summation_buttons_layout.addWidget(self.summation_remove_button)
        summation_layout.addLayout(summation_buttons_layout)

        # Output folder input box with "Browse" button
        summation_output_layout = QHBoxLayout()
        self.summation_output_input = QLineEdit()
        self.summation_output_input.setPlaceholderText("Select output folder to save summed images")
        self.summation_output_browse = QPushButton("Set output")
        self.summation_output_browse.clicked.connect(lambda: self.browse_output_folder(self.summation_output_input))
        summation_output_layout.addWidget(self.summation_output_input)
        summation_output_layout.addWidget(self.summation_output_browse)
        summation_layout.addLayout(summation_output_layout)

        # Base name input box
        self.summation_basename_input = QLineEdit()
        self.summation_basename_input.setPlaceholderText("Enter base name for summed images (e.g., Fe_images)")
        summation_layout.addWidget(self.summation_basename_input)

        # "Sum" button
        summation_start_stop_layout = QHBoxLayout()
        self.summation_sum_button = QPushButton("Sum")
        self.summation_sum_button.clicked.connect(self.sum_images)
        self.summation_stop_button = QPushButton("Stop")
        self.summation_stop_button.setEnabled(False)  # Initially disabled
        self.summation_stop_button.clicked.connect(self.stop_summation)
        summation_start_stop_layout.addWidget(self.summation_sum_button)
        summation_start_stop_layout.addWidget(self.summation_stop_button)
        summation_layout.addLayout(summation_start_stop_layout)

        # Progress bars for loading and summation
        self.summation_load_progress = QProgressBar()
        self.summation_load_progress.setValue(0)
        summation_layout.addWidget(QLabel("Image Loading Progress:"))
        summation_layout.addWidget(self.summation_load_progress)

        self.summation_progress = QProgressBar()
        self.summation_progress.setValue(0)
        summation_layout.addWidget(QLabel("Summation Progress:"))
        summation_layout.addWidget(self.summation_progress)

        summation_group.setLayout(summation_layout)
        # Add sections to the upper layout
        upper_layout.addWidget(summation_group)

    # Outlier Filtering Section
        outlier_group = QGroupBox("Clean")
        outlier_layout = QVBoxLayout()

        # "Add" and "Remove" buttons
        outlier_buttons_layout = QHBoxLayout()
        self.outlier_add_button = QPushButton("Add data")
        self.outlier_add_button.clicked.connect(self.add_outlier_images)
        # self.outlier_remove_button = QPushButton("Remove")
        # self.outlier_remove_button.clicked.connect(self.remove_outlier_images)
        outlier_buttons_layout.addWidget(self.outlier_add_button)
        # outlier_buttons_layout.addWidget(self.outlier_remove_button)
        outlier_layout.addLayout(outlier_buttons_layout)

        # Output folder
        outlier_output_layout = QHBoxLayout()
        self.outlier_output_input = QLineEdit()
        self.outlier_output_input.setPlaceholderText("Select output folder for cleaned images")
        self.outlier_output_browse = QPushButton("Set output")
        self.outlier_output_browse.clicked.connect(lambda: self.browse_output_folder(self.outlier_output_input))
        outlier_output_layout.addWidget(self.outlier_output_input)
        outlier_output_layout.addWidget(self.outlier_output_browse)
        outlier_layout.addLayout(outlier_output_layout)

        # Base name
        self.outlier_basename_input = QLineEdit()
        self.outlier_basename_input.setPlaceholderText("Enter base name (e.g., Cleaned_Fe)")
        outlier_layout.addWidget(self.outlier_basename_input)

        # Action buttons
        outlier_action_layout = QHBoxLayout()
        self.outlier_process_button = QPushButton("Clean")
        self.outlier_process_button.clicked.connect(self.remove_outliers)
        self.outlier_stop_button = QPushButton("Stop")
        self.outlier_stop_button.setEnabled(False)
        self.outlier_stop_button.clicked.connect(self.stop_outlier_removal)
        outlier_action_layout.addWidget(self.outlier_process_button)
        outlier_action_layout.addWidget(self.outlier_stop_button)
        outlier_layout.addLayout(outlier_action_layout)

        # Progress bars
        self.outlier_load_progress = QProgressBar()
        self.outlier_load_progress.setValue(0)
        outlier_layout.addWidget(QLabel("Image Loading Progress:"))
        outlier_layout.addWidget(self.outlier_load_progress)

        self.outlier_progress = QProgressBar()
        self.outlier_progress.setValue(0)
        outlier_layout.addWidget(QLabel("Clean Progress:"))
        outlier_layout.addWidget(self.outlier_progress)

        outlier_group.setLayout(outlier_layout)
        upper_layout.addWidget(outlier_group)   



        overlap_correction_group = QGroupBox("Overlap Correction")
        overlap_correction_layout = QVBoxLayout()

        # "Add" and "Remove" buttons to load and remove folder of images
        overlap_correction_buttons_layout = QHBoxLayout()
        self.overlap_correction_add_button = QPushButton("Add data")
        self.overlap_correction_add_button.clicked.connect(self.add_overlap_correction_images)
        # self.overlap_correction_remove_button = QPushButton("Remove")
        # self.overlap_correction_remove_button.clicked.connect(self.remove_overlap_correction_images)
        overlap_correction_buttons_layout.addWidget(self.overlap_correction_add_button)
        # overlap_correction_buttons_layout.addWidget(self.overlap_correction_remove_button)
        overlap_correction_layout.addLayout(overlap_correction_buttons_layout)

        # Output folder input box with "Browse" button
        overlap_correction_output_layout = QHBoxLayout()
        self.overlap_correction_output_input = QLineEdit()
        self.overlap_correction_output_input.setPlaceholderText("Select output folder to save overlap corrected images")
        self.overlap_correction_output_browse = QPushButton("Set output")
        self.overlap_correction_output_browse.clicked.connect(
            lambda: self.browse_output_folder(self.overlap_correction_output_input)
        )
        overlap_correction_output_layout.addWidget(self.overlap_correction_output_input)
        overlap_correction_output_layout.addWidget(self.overlap_correction_output_browse)
        overlap_correction_layout.addLayout(overlap_correction_output_layout)

        # Base name input box
        self.overlap_correction_basename_input = QLineEdit()
        self.overlap_correction_basename_input.setPlaceholderText("Enter base name for overlap corrected images (e.g., Overlap_Corrected_Fe)")
        overlap_correction_layout.addWidget(self.overlap_correction_basename_input)

        # "Correct Overlap" button
        overlap_correction_start_stop_layout = QHBoxLayout()
        self.overlap_correction_correct_button = QPushButton("Correct")
        self.overlap_correction_stop_button = QPushButton("Stop")
        self.overlap_correction_stop_button.setEnabled(False)
        self.overlap_correction_correct_button.clicked.connect(self.correct_overlap)
        self.overlap_correction_stop_button.clicked.connect(self.stop_overlap_correction)
        overlap_correction_start_stop_layout.addWidget(self.overlap_correction_correct_button)
        overlap_correction_start_stop_layout.addWidget(self.overlap_correction_stop_button)
        overlap_correction_layout.addLayout(overlap_correction_start_stop_layout)


        # Progress bars for loading and overlap correction
        self.overlap_correction_load_progress = QProgressBar()
        self.overlap_correction_load_progress.setValue(0)
        overlap_correction_layout.addWidget(QLabel("Image Loading Progress:"))
        overlap_correction_layout.addWidget(self.overlap_correction_load_progress)

        self.overlap_correction_progress = QProgressBar()
        self.overlap_correction_progress.setValue(0)
        overlap_correction_layout.addWidget(QLabel("Overlap Correction Progress:"))
        overlap_correction_layout.addWidget(self.overlap_correction_progress)

        overlap_correction_group.setLayout(overlap_correction_layout)

        # normalisation Section
        normalisation_group = QGroupBox("Normalisation")
        normalisation_layout = QVBoxLayout()

        # "Add Data" and "Remove Data" buttons to load and remove folder of data images
        normalisation_data_buttons_layout = QHBoxLayout()
        self.normalisation_add_data_button = QPushButton("Add Data")
        self.normalisation_add_data_button.clicked.connect(self.add_normalisation_data_images)
        # self.normalisation_remove_data_button = QPushButton("Remove Data")
        # self.normalisation_remove_data_button.clicked.connect(self.remove_normalisation_data_images)
        normalisation_data_buttons_layout.addWidget(self.normalisation_add_data_button)
        # normalisation_data_buttons_layout.addWidget(self.normalisation_remove_data_button)
        normalisation_layout.addLayout(normalisation_data_buttons_layout)

        # "Load Open Beam" and "Remove Open Beam" buttons to load and remove open beam images
        normalisation_openbeam_buttons_layout = QHBoxLayout()
        self.normalisation_load_openbeam_button = QPushButton("Add Open Beam")
        self.normalisation_load_openbeam_button.clicked.connect(self.load_open_beam_images)
        # self.normalisation_remove_openbeam_button = QPushButton("Remove Open Beam")
        # self.normalisation_remove_openbeam_button.clicked.connect(self.remove_normalisation_open_beam_images)
        normalisation_openbeam_buttons_layout.addWidget(self.normalisation_load_openbeam_button)
        # normalisation_openbeam_buttons_layout.addWidget(self.normalisation_remove_openbeam_button)
        normalisation_layout.addLayout(normalisation_openbeam_buttons_layout)

        # Output folder input box with "Browse" button
        normalisation_output_layout = QHBoxLayout()
        self.normalisation_output_input = QLineEdit()
        self.normalisation_output_input.setPlaceholderText("Select output folder to save normalised images")
        self.normalisation_output_browse = QPushButton("Set output")
        self.normalisation_output_browse.clicked.connect(lambda: self.browse_output_folder(self.normalisation_output_input))
        normalisation_output_layout.addWidget(self.normalisation_output_input)
        normalisation_output_layout.addWidget(self.normalisation_output_browse)
        normalisation_layout.addLayout(normalisation_output_layout)

        # Base name input box
        self.normalisation_basename_input = QLineEdit()
        self.normalisation_basename_input.setPlaceholderText("Enter base name for normalised images (e.g., Normalised_Fe)")
        normalisation_layout.addWidget(self.normalisation_basename_input)

        window_layout = QHBoxLayout()
        # Additional input box for window half (an integer, e.g., 10 for a 21x21 window)
        self.normalisation_window_half_input = QLineEdit("10")
        self.normalisation_window_half_input.setToolTip('Set moving binning pixel size, i.e. "n", size will be (2n+1)x(2n+1)')
        window_layout.addWidget(self.normalisation_window_half_input)
        normalisation_layout.addLayout(window_layout)

        self.normalisation_adjacent_input = QLineEdit("0")
        self.normalisation_adjacent_input.setToolTip('Set moving frame window size, i.e. "m", size will be (2m+1)x(2m+1)')
        window_layout.addWidget(self.normalisation_adjacent_input)
        normalisation_layout.addLayout(window_layout)

        # # Base name input box
        # self.normalisation_basename_input = QLineEdit()
        # self.normalisation_basename_input.setPlaceholderText("Enter base name for normalised images (e.g., normalised_Fe)")
        # normalisation_layout.addWidget(self.normalisation_basename_input)


        # "normalise" button
        normalisation_start_stop_layout = QHBoxLayout()
        self.normalisation_normalise_button = QPushButton("Normalise")
        self.normalisation_normalise_button.clicked.connect(self.normalise_images)
        self.normalisation_stop_button = QPushButton("Stop")
        self.normalisation_stop_button.setEnabled(False)
        self.normalisation_stop_button.clicked.connect(self.stop_normalisation)
        normalisation_start_stop_layout.addWidget(self.normalisation_normalise_button)
        normalisation_start_stop_layout.addWidget(self.normalisation_stop_button)
        normalisation_layout.addLayout(normalisation_start_stop_layout)

        # Progress bars for loading and normalisation
        self.normalisation_load_progress = QProgressBar()
        self.normalisation_load_progress.setValue(0)
        normalisation_layout.addWidget(QLabel("Image Loading Progress:"))
        normalisation_layout.addWidget(self.normalisation_load_progress)

        self.normalisation_progress = QProgressBar()
        self.normalisation_progress.setValue(0)
        normalisation_layout.addWidget(QLabel("Normalisation Progress:"))
        normalisation_layout.addWidget(self.normalisation_progress)

        normalisation_group.setLayout(normalisation_layout)


        # Add the overlap_correction_group to the upper_layout
        upper_layout.addWidget(overlap_correction_group)
        # upper_layout.addWidget(scaling_group)
        upper_layout.addWidget(normalisation_group)

        # Message Box in the lower layout
        self.preproc_message_box = QTextEdit()
        self.preproc_message_box.setReadOnly(True)
        self.preproc_message_box.setPlaceholderText("Messages will appear here...")
        lower_layout.addWidget(QLabel("Messages:"))
        lower_layout.addWidget(self.preproc_message_box)

        # Add Clear and Save buttons below the message box
        msg_button_layout = QHBoxLayout()
        msg_button_layout.setAlignment(Qt.AlignLeft)
        self.clear_msg_button = QPushButton("Clear message")
        self.clear_msg_button.setFixedWidth(250)
        self.clear_msg_button.clicked.connect(self.clear_messages)
        self.save_msg_button = QPushButton("Save message")
        self.save_msg_button.setFixedWidth(250)
        self.save_msg_button.clicked.connect(self.save_messages)
        msg_button_layout.addWidget(self.clear_msg_button)
        msg_button_layout.addWidget(self.save_msg_button)
        lower_layout.addLayout(msg_button_layout)

        # Add upper and lower layouts to the main layout
        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)


        # ------------------------------------------------------
        # Filtering Section (NEW)
        # ------------------------------------------------------
        filtering_group = QGroupBox("Filtering")
        filtering_layout = QVBoxLayout()

        # "Add" and "Remove" buttons to load and remove folder of FITS data images
        filtering_data_buttons_layout = QHBoxLayout()
        self.filtering_add_button = QPushButton("Add data")
        self.filtering_add_button.clicked.connect(self.add_filter_data_images)
        # self.filtering_remove_button = QPushButton("Remove")
        # self.filtering_remove_button.clicked.connect(self.remove_filter_data_images)
        filtering_data_buttons_layout.addWidget(self.filtering_add_button)
        # filtering_data_buttons_layout.addWidget(self.filtering_remove_button)
        filtering_layout.addLayout(filtering_data_buttons_layout)

        # "Add Mask" and "Remove Mask" buttons for a single FITS mask image
        filtering_mask_buttons_layout = QHBoxLayout()
        self.filtering_add_mask_button = QPushButton("Add Mask")
        self.filtering_add_mask_button.clicked.connect(self.add_filter_mask_image)
        # self.filtering_remove_mask_button = QPushButton("Remove Mask")
        # self.filtering_remove_mask_button.clicked.connect(self.remove_filter_mask_image)
        filtering_mask_buttons_layout.addWidget(self.filtering_add_mask_button)
        # filtering_mask_buttons_layout.addWidget(self.filtering_remove_mask_button)
        filtering_layout.addLayout(filtering_mask_buttons_layout)

        # Output folder input box with "Browse" button
        filtering_output_layout = QHBoxLayout()
        self.filtering_output_input = QLineEdit()
        self.filtering_output_input.setPlaceholderText("Select output folder to save filtered images")
        self.filtering_output_browse = QPushButton("Set output")
        self.filtering_output_browse.clicked.connect(
            lambda: self.browse_output_folder(self.filtering_output_input)
        )
        filtering_output_layout.addWidget(self.filtering_output_input)
        filtering_output_layout.addWidget(self.filtering_output_browse)
        filtering_layout.addLayout(filtering_output_layout)

        # Base name input box
        self.filtering_basename_input = QLineEdit()
        self.filtering_basename_input.setPlaceholderText("Enter base name for filtered images (e.g., Filtered_Fe)")
        filtering_layout.addWidget(self.filtering_basename_input)

        filtering_buttons_layout = QHBoxLayout()
        # Filter button
        self.filtering_filter_button = QPushButton("Filter")
        self.filtering_filter_button.clicked.connect(self.filter_images)
        filtering_buttons_layout.addWidget(self.filtering_filter_button)

        # Stop button (initially disabled)
        self.filtering_stop_button = QPushButton("Stop")
        self.filtering_stop_button.setEnabled(False)
        self.filtering_stop_button.clicked.connect(self.stop_filtering)
        filtering_buttons_layout.addWidget(self.filtering_stop_button)

        # Add this horizontal layout to the main filtering layout
        filtering_layout.addLayout(filtering_buttons_layout)

        # Progress bars for loading and filtering
        self.filtering_load_progress = QProgressBar()
        self.filtering_load_progress.setValue(0)
        filtering_layout.addWidget(QLabel("Image Loading Progress:"))
        filtering_layout.addWidget(self.filtering_load_progress)

        self.filtering_progress = QProgressBar()
        self.filtering_progress.setValue(0)
        filtering_layout.addWidget(QLabel("Filtering Progress:"))
        filtering_layout.addWidget(self.filtering_progress)

        filtering_group.setLayout(filtering_layout)

        # Finally, add the Filtering group to the upper_layout
        upper_layout.addWidget(filtering_group)
        # ------------------------------------------------------

    # --------------------------------------------------------------------
        # Full Process Section (Similar Layout to Normalisation)
        # --------------------------------------------------------------------
        full_process_group = QGroupBox("Full Process")
        full_process_layout = QVBoxLayout()

        # Output folder
        full_process_output_layout = QHBoxLayout()
        self.full_process_output_input = QLineEdit()
        self.full_process_output_input.setPlaceholderText("Select output folder to save full-process images")
        self.full_process_output_browse = QPushButton("Set output")
        self.full_process_output_browse.clicked.connect(lambda: self.browse_output_folder(self.full_process_output_input))
        full_process_output_layout.addWidget(self.full_process_output_input)
        full_process_output_layout.addWidget(self.full_process_output_browse)
        full_process_layout.addLayout(full_process_output_layout)

        # Base name
        self.full_process_basename_input = QLineEdit()
        self.full_process_basename_input.setPlaceholderText("Enter base name for processed images (e.g., FullProcess_Fe)")
        full_process_layout.addWidget(self.full_process_basename_input)

        # Window half & adjacent input
        full_process_window_layout = QHBoxLayout()
        self.full_process_window_half_input = QLineEdit("10")
        self.full_process_window_half_input.setToolTip('Set binning pixel size, i.e. "n", size will be (2n+1)x(2n+1)')
        full_process_window_layout.addWidget(self.full_process_window_half_input)

        # self.full_process_adjacent_input = QLineEdit("10")
        # self.full_process_adjacent_input.setToolTip('Set moving frame window size, i.e. "10", size will be 21 (10x2+1)')
        # full_process_window_layout.addWidget(self.full_process_adjacent_input)
        full_process_layout.addLayout(full_process_window_layout)

        # "Run Full Process" + "Stop" buttons
        full_process_start_stop_layout = QHBoxLayout()
        self.full_process_start_button = QPushButton("Full Process")
        self.full_process_start_button.clicked.connect(self.run_full_process)  # Or your custom method
        self.full_process_stop_button = QPushButton("Stop")
        self.full_process_stop_button.setEnabled(False)
        self.full_process_stop_button.clicked.connect(self.stop_full_process)  # Or your custom method
        full_process_start_stop_layout.addWidget(self.full_process_start_button)
        full_process_start_stop_layout.addWidget(self.full_process_stop_button)
        full_process_layout.addLayout(full_process_start_stop_layout)

        # Progress bars
        self.full_process_load_progress = QProgressBar()
        self.full_process_load_progress.setValue(0)
        full_process_layout.addWidget(QLabel("Image Loading Progress:"))
        full_process_layout.addWidget(self.full_process_load_progress)

        self.full_process_progress = QProgressBar()
        # self.full_process_progress.setRange(0, 100)   # explicit range
        # self.full_process_progress.setFormat("%p%")   # show “xx %”
        # self.full_process_progress.setTextVisible(True)

        self.full_process_progress.setValue(0)
        full_process_layout.addWidget(QLabel("Full Process Progress:"))
        full_process_layout.addWidget(self.full_process_progress)

        full_process_group.setLayout(full_process_layout)
        upper_layout.addWidget(full_process_group)

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

    def clear_messages(self):
        """Clears the messages in the message box."""
        self.preproc_message_box.clear()

    def save_messages(self):
        """Saves the messages in the message box to a text file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Messages", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.preproc_message_box.toPlainText())
                self.preproc_message_box.append("Messages saved to: " + file_path)
            except Exception as e:
                self.preproc_message_box.append("Error saving messages: " + str(e))

    def run_full_process(self):
        """
        Orchestrates the entire pipeline:
        1) Summation (if multiple subfolders)
        2) Outlier Removal (Clean)
        3) Overlap Correction
        4) Normalisation
        """
        # -- Grab needed UI values from the 'Full Process' block --
        sample_folder = QFileDialog.getExistingDirectory(
            self, "Select Sample Folder for Full Process", ""
        )
        if not sample_folder:
            self.preproc_message_box.append("No sample folder selected. Aborting full process.")
            return

        open_beam_folder = QFileDialog.getExistingDirectory(
            self, "Select Open Beam Folder for Full Process", ""
        )
        if not open_beam_folder:
            self.preproc_message_box.append("No open beam folder selected. Aborting full process.")
            return

        output_folder = self.full_process_output_input.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            self.preproc_message_box.append("Please select a valid output folder for the full process.")
            return

        base_name = self.full_process_basename_input.text().strip()
        if not base_name:
            base_name = "FullProcess"

        try:
            window_half = int(self.full_process_window_half_input.text().strip())
        except ValueError:
            window_half = 10


        # try:
        #     adjacent_sum = int(self.full_process_adjacent_input.text().strip())
        # except ValueError:
        #     adjacent_sum = 10
        adjacent_sum = 0

        # Disable the button to prevent duplicates; enable the Stop button
        self.full_process_start_button.setEnabled(False)
        self.full_process_stop_button.setEnabled(True)

        self.preproc_message_box.append("Starting the Full Process pipeline...")

        # Create and launch our combined worker
        self.full_process_worker = FullProcessWorker(
            sample_folder,
            open_beam_folder,
            output_folder,
            base_name,
            window_half,
            adjacent_sum
        )
        # # Connect worker signals to your existing “universal” slots (or create new ones).
        # self.full_process_worker.message.connect(self.preproc_message_box.append)
        # self.full_process_worker.finished.connect(self._full_process_finished)
        # self.full_process_worker.progress_updated.connect(self._full_process_progress_update)
        # self.full_process_worker.load_progress_updated.connect(self._full_process_load_progress_update)

        self.full_process_worker.message.connect(
        self.preproc_message_box.append, Qt.QueuedConnection
        )
        self.full_process_worker.progress_updated.connect(
            self._full_process_progress_update, Qt.QueuedConnection
        )
        self.full_process_worker.load_progress_updated.connect(
            self._full_process_load_progress_update, Qt.QueuedConnection
        )
        self.full_process_worker.finished.connect(self._full_process_finished, Qt.QueuedConnection)


        self.full_process_worker.start()

    def _full_process_progress_update(self, value: int):
        """
        Updates the Full Process progress bar with the given value.
        """
        self.full_process_progress.setValue(value)

    def _full_process_load_progress_update(self, value: int):
        """Updates the Image Loading progress bar with the given value."""
        self.full_process_load_progress.setValue(value)

    def stop_full_process(self):
        if hasattr(self, 'full_process_worker') and self.full_process_worker:
            self.full_process_worker.stop()
            # Optionally also call self.full_process_worker.quit() and wait() if you like
            self.full_process_stop_button.setEnabled(False)
            self.preproc_message_box.append("Stop signal sent to Full Process.")
        else:
            self.preproc_message_box.append("No active Full Process to stop.")

    def _full_process_finished(self):
        """Handle the finishing of the full process, re-enabling buttons as needed."""
        self.full_process_start_button.setEnabled(True)
        self.full_process_stop_button.setEnabled(False)

    def add_outlier_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing FITS Images for Outlier Removal", ""
        )
        if folder_path:
            # Check if the selected folder has child folders.
            child_folders = [
                os.path.join(folder_path, item)
                for item in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, item))
            ]
            if child_folders:
                self.preproc_message_box.append(
                    f"Detected {len(child_folders)} sub-folders. They will be processed sequentially for outlier removal."
                )
                self._outlier_batch_paths = child_folders
            else:
                self.preproc_message_box.append(
                    "No sub-folders detected; the selected folder will be treated as a single dataset."
                )
                self._outlier_batch_paths = [folder_path]

    def remove_outliers(self):
        """
        Batch process outlier removal:
        - If a parent folder with child folders is selected, each child folder is treated as a separate dataset.
        - For each dataset, images are loaded on demand, then OutlierFilteringWorker is run,
          results are saved to an output folder (named "outlier_removed_<foldername>"),
          and memory is cleaned up before moving to the next dataset.
        """
        if not hasattr(self, '_outlier_batch_paths') or not self._outlier_batch_paths:
            self.preproc_message_box.append("No outlier dataset folder selected. Please add images first.")
            return

        output_folder = self.outlier_output_input.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            self.preproc_message_box.append("Please select a valid output folder.")
            return

        base_name = self.outlier_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append("Please enter a base name.")
            return

        # Store parameters and initialize batch index
        self._outlier_output_folder = output_folder
        self._outlier_base_name = base_name
        self._current_outlier_index = 0

        self.outlier_process_button.setEnabled(False)
        self.outlier_stop_button.setEnabled(True)

        self.preproc_message_box.append("Starting batch outlier removal...")
        self._process_next_outlier_dataset()

    def _process_next_outlier_dataset(self):
        """
        Process the next dataset folder in the outlier removal batch.
        """
        if self._current_outlier_index >= len(self._outlier_batch_paths):
            self.preproc_message_box.append("Batch outlier removal completed.")
            self.outlier_process_button.setEnabled(True)
            self.outlier_stop_button.setEnabled(False)
            return

        current_folder = self._outlier_batch_paths[self._current_outlier_index]
        # # Clean up folder path
        # normalized_path = os.path.normpath(current_folder)
        # path_parts = normalized_path.split(os.sep)
        # if len(path_parts) >= 2:
        #     short_path = os.path.join(path_parts[-2], path_parts[-1])
        # else:
        #     short_path = normalized_path

        short_path = self.get_short_path(current_folder, levels=2)

        self.preproc_message_box.append("Loading dataset from folder: \\" + short_path)
        # Start an ImageLoadWorker to load images from the current dataset folder.
        self._outlier_data_load_worker = ImageLoadWorker(current_folder)
        self._outlier_data_load_worker.progress_updated.connect(self.update_outlier_load_progress)
        self._outlier_data_load_worker.message.connect(self.preproc_message_box.append)
        self._outlier_data_load_worker.run_loaded.connect(self._on_outlier_dataset_loaded)
        self._outlier_data_load_worker.finished.connect(self._on_outlier_data_load_finished)
        self._outlier_data_load_worker.start()

    def _on_outlier_dataset_loaded(self, folder_path, run_dict):
        if run_dict:
            self._current_outlier_run = {
                'folder_path': folder_path,
                'images': run_dict
            }
        else:
            self._current_outlier_run = None

    def _on_outlier_data_load_finished(self):
        """
        After loading images from the current dataset, start outlier removal for that dataset.
        """
        if not self._current_outlier_run:
            self.preproc_message_box.append("Failed to load dataset; skipping...")
            self._current_outlier_index += 1
            self._process_next_outlier_dataset()
            return

        folder_path = self._current_outlier_run['folder_path']
        folder_path_short = self.get_short_path(folder_path, levels =2)

        # Create an output folder specific to this dataset (e.g., "outlier_removed_<foldername>")
        output_folder_run = os.path.join(self._outlier_output_folder, "outlier_removed_" + os.path.basename(folder_path))
        try:
            if not os.path.exists(output_folder_run):
                os.makedirs(output_folder_run)
        except Exception as e:
            self.preproc_message_box.append(f"Failed to create output folder for {folder_path_short}: {e}")
            self._current_outlier_index += 1
            self._process_next_outlier_dataset()
            return

        self.preproc_message_box.append("Starting outlier removal for dataset: " + folder_path_short)
        self.outlier_worker = OutlierFilteringWorker(
            [self._current_outlier_run],  # Process the loaded dataset as a single-item list
            output_folder_run,
            self._outlier_base_name
        )
        self.outlier_worker.progress_updated.connect(self.update_outlier_progress)
        self.outlier_worker.message.connect(self.preproc_message_box.append)
        self.outlier_worker.finished.connect(self._on_outlier_removal_finished)
        self.outlier_worker.start()

    def _on_outlier_removal_finished(self):
        self.preproc_message_box.append(
            f"Finished outlier removal for dataset {self._current_outlier_index+1} of {len(self._outlier_batch_paths)}."
        )
        # Clean up references and force garbage collection
        self._current_outlier_run = None
        self._outlier_data_load_worker = None
        self.outlier_worker = None
        gc.collect()
        # Move on to the next dataset
        self._current_outlier_index += 1

        # If there are more datasets, re-enable the stop button for the next run.
        if self._current_outlier_index < len(self._outlier_batch_paths):
            self.outlier_stop_button.setEnabled(True)
        else:
            # Batch is complete; re-enable the normalise button and disable the stop button.
            self.outlier_process_button.setEnabled(True)
            self.outlier_stop_button.setEnabled(False)

        self._process_next_outlier_dataset()

    def remove_outlier_images(self):
        if hasattr(self, 'outlier_image_runs') and self.outlier_image_runs:
            self.outlier_image_runs = []
            self.preproc_message_box.append("Outlier images removed.")
            self.outlier_load_progress.setValue(0)
            self.outlier_progress.setValue(0)
        else:
            self.preproc_message_box.append("No outlier images to remove.")

    def handle_outlier_run_loaded(self, folder_path, run_dict):
        if run_dict:
            self.outlier_image_runs.append({
                'folder_path': folder_path,
                'images': run_dict
            })
            self.preproc_message_box.append(f"Loaded {len(run_dict)} images from {folder_path}")

    def outlier_image_loading_finished(self):
        if self.outlier_image_load_worker:
            self.outlier_image_load_worker = None

    def update_outlier_load_progress(self, value):
        self.outlier_load_progress.setValue(value)

    def stop_outlier_removal(self):
        if hasattr(self, 'outlier_worker') and self.outlier_worker:
            self.outlier_worker.stop()
            self.outlier_stop_button.setEnabled(False)
            self.preproc_message_box.append("Outlier removal stopped.")
        else:
            self.preproc_message_box.append("No active outlier removal process to stop.")

    def outlier_process_finished(self):
        self.outlier_process_button.setEnabled(True)
        self.outlier_stop_button.setEnabled(False)
        self.outlier_progress.setValue(0)
        self.outlier_worker = None

    def update_outlier_progress(self, value):
        self.outlier_progress.setValue(value)

    def add_overlap_correction_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing FITS Images for Overlap Correction", ""
        )
        if folder_path:
            # Check for child folders inside the selected folder
            child_folders = [
                os.path.join(folder_path, item)
                for item in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, item))
            ]
            if child_folders:
                self.preproc_message_box.append(
                    f"Detected {len(child_folders)} sub-folders. They will be processed sequentially for overlap correction."
                )
                self._overlap_batch_paths = child_folders
            else:
                self.preproc_message_box.append(
                    "No sub-folders detected; the selected folder will be treated as a single dataset."
                )
                self._overlap_batch_paths = [folder_path]

    def remove_overlap_correction_images(self):
        if hasattr(self, 'overlap_correction_image_runs') and self.overlap_correction_image_runs:
            self.overlap_correction_image_runs = []
            self.preproc_message_box.append("Overlap Correction images have been removed.")
            self.overlap_correction_load_progress.setValue(0)
            self.overlap_correction_progress.setValue(0)
        else:
            self.preproc_message_box.append("No Overlap Correction images to remove.")

    def handle_overlap_correction_run_loaded(self, folder_path, run_dict):
        if run_dict:
            # Add the run to overlap_correction_image_runs
            self.overlap_correction_image_runs.append({
                'folder_path': folder_path,
                'images': run_dict,
                'spectra': None,
                'shutter_count': None
            })

            # self.preproc_message_box.append(f"Successfully loaded Overlap Correction from '{folder_path}' with {len(run_dict)} images.")
        else:
            self.preproc_message_box.append("No valid FITS images were loaded from the selected folder.")

    def overlap_correction_image_loading_finished(self):
        if self.overlap_correction_image_load_worker:
            self.overlap_correction_image_load_worker = None  # Cleanup

    def update_overlap_correction_load_progress(self, value):
        self.overlap_correction_load_progress.setValue(value)

    def correct_overlap(self):
        """
        Batch process overlap correction:
        - If a parent folder with child folders is selected, each child folder is treated as a separate dataset.
        - For each dataset, images are loaded on demand, necessary text files (Spectra and ShutterCount) are loaded,
          and OverlapCorrectionWorker is run.
        """
        if not hasattr(self, '_overlap_batch_paths') or not self._overlap_batch_paths:
            self.preproc_message_box.append(
                "No overlap correction dataset folder selected. Please add image sets first."
            )
            return

        overall_output_folder = self.overlap_correction_output_input.text().strip()
        if not overall_output_folder or not os.path.isdir(overall_output_folder):
            self.preproc_message_box.append(
                "Please select a valid output folder to save overlap corrected images."
            )
            return

        base_name = self.overlap_correction_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append(
                "Please enter a base name for overlap corrected images."
            )
            return

        # Store parameters and initialize batch index
        self._overlap_output_folder = overall_output_folder
        self._overlap_base_name = base_name
        self._current_overlap_index = 0

        # Disable the Correct button to prevent duplicate runs
        self.overlap_correction_correct_button.setEnabled(False)
        self.overlap_correction_stop_button.setEnabled(True)

        self.preproc_message_box.append("Starting batch overlap correction...")
        self._process_next_overlap_dataset()

    def _process_next_overlap_dataset(self):
        """
        Load and process the next dataset folder in the overlap correction batch.
        """
        if self._current_overlap_index >= len(self._overlap_batch_paths):
            self.preproc_message_box.append("Batch overlap correction completed.")
            self.overlap_correction_correct_button.setEnabled(True)
            self.overlap_correction_stop_button.setEnabled(False)
            return

        current_folder = self._overlap_batch_paths[self._current_overlap_index]
        # Clean up folder path
        # normalized_path = os.path.normpath(current_folder)
        # path_parts = normalized_path.split(os.sep)
        # if len(path_parts) >= 2:
        #     short_path = os.path.join(path_parts[-2], path_parts[-1])
        # else:
        #     short_path = normalized_path
        short_path = self.get_short_path(current_folder, levels=2)

        self.preproc_message_box.append("Loading dataset from folder: " + short_path)
        # Start an ImageLoadWorker for the current folder
        self._overlap_data_load_worker = ImageLoadWorker(current_folder)
        self._overlap_data_load_worker.progress_updated.connect(self.update_overlap_correction_load_progress)
        self._overlap_data_load_worker.message.connect(self.preproc_message_box.append)
        self._overlap_data_load_worker.run_loaded.connect(self._on_overlap_dataset_loaded)
        self._overlap_data_load_worker.finished.connect(self._on_overlap_data_load_finished)
        self._overlap_data_load_worker.start()

    def _on_overlap_dataset_loaded(self, folder_path, run_dict):
        # Store the loaded dataset along with placeholders for spectra and shutter count data.
        if run_dict:
            self._current_overlap_run = {
                'folder_path': folder_path,
                'images': run_dict,
                'spectra': None,
                'shutter_count': None
            }
        else:
            self._current_overlap_run = None

    def _on_overlap_data_load_finished(self):
        """
        After loading images from the current dataset folder, load required text files.
        """
        if not self._current_overlap_run:
            self.preproc_message_box.append("Failed to load dataset; skipping...")
            self._current_overlap_index += 1
            self._process_next_overlap_dataset()
            return

        folder_path = self._current_overlap_run['folder_path']

        short_path = self.get_short_path(folder_path, levels=2)

        try:
            all_files = os.listdir(folder_path)
        except Exception as e:
            self.preproc_message_box.append(f"Error accessing folder '\\{short_path}': {e}")
            self._current_overlap_index += 1
            self._process_next_overlap_dataset()
            return

        # Identify Spectra and ShutterCount files (using first match)
        spectra_files = [f for f in all_files if f.endswith('_Spectra.txt')]
        shutter_files = [f for f in all_files if f.endswith('_ShutterCount.txt')]

        # Load Spectra
        if spectra_files:
            spectra_file = os.path.join(folder_path, spectra_files[0])
            try:
                spectra_data = np.loadtxt(spectra_file)
                self._current_overlap_run['spectra'] = spectra_data
                # self.preproc_message_box.append("Loaded Spectra file: " + spectra_file)
            except Exception as e:
                self.preproc_message_box.append(f"Failed to load Spectra file '{spectra_file}': {e}")
                self._current_overlap_run['spectra'] = None
        else:
            self.preproc_message_box.append(f"Spectra file not found in '\\{short_path}'.")
            self._current_overlap_run['spectra'] = None

        # Load ShutterCount
        if shutter_files:
            shutter_file = os.path.join(folder_path, shutter_files[0])
            try:
                shutter_data = np.loadtxt(shutter_file)
                self._current_overlap_run['shutter_count'] = shutter_data[shutter_data != 0]
                # self.preproc_message_box.append("Loaded ShutterCount file: " + shutter_file)
            except Exception as e:
                self.preproc_message_box.append(f"Failed to load ShutterCount file '{shutter_file}': {e}")
                self._current_overlap_run['shutter_count'] = None
        else:
            self.preproc_message_box.append(f"ShutterCount file not found in '\\{short_path}'.")
            self._current_overlap_run['shutter_count'] = None

        # If required text data is missing, skip this dataset.
        if self._current_overlap_run['spectra'] is None or self._current_overlap_run['shutter_count'] is None:
            self.preproc_message_box.append("Dataset in " + folder_path + " lacks necessary Spectra or ShutterCount data; skipping.")
            self._current_overlap_index += 1
            self._process_next_overlap_dataset()
            return

        # Create an output folder for this dataset (e.g. "Corrected_<foldername>")
        output_folder_run = os.path.join(self._overlap_output_folder, "Corrected_" + os.path.basename(folder_path))
        try:
            if not os.path.exists(output_folder_run):
                os.makedirs(output_folder_run)
        except Exception as e:
            self.preproc_message_box.append(f"Failed to create output folder for \\{short_path}: {e}")
            self._current_overlap_index += 1
            self._process_next_overlap_dataset()
            return

        self.preproc_message_box.append("Starting overlap correction for dataset: <b>\\" + short_path + "</b>")
        # Start the OverlapCorrectionWorker for the current run.
        self.overlap_correction_worker = OverlapCorrectionWorker(
            self._current_overlap_run, self._overlap_base_name, output_folder_run
        )
        self.overlap_correction_worker.progress_updated.connect(self.update_overlap_correction_progress)
        self.overlap_correction_worker.message.connect(self.preproc_message_box.append)
        self.overlap_correction_worker.finished.connect(self._on_overlap_correction_finished)
        self.overlap_correction_worker.start()

    def _on_overlap_correction_finished(self):
        """
        Called when the overlap correction worker finishes processing the current dataset.
        Cleans up references and moves on to the next dataset.
        """
        self.preproc_message_box.append(
            f"Finished overlap correction for dataset {self._current_overlap_index + 1} of {len(self._overlap_batch_paths)}."
        )
        # Clean up
        self._current_overlap_run = None
        self._overlap_data_load_worker = None
        self.overlap_correction_worker = None
        gc.collect()
        # Move to next dataset
        self._current_overlap_index += 1

        # If there are more datasets, re-enable the stop button for the next run.
        if self._current_overlap_index < len(self._overlap_batch_paths):
            self.overlap_correction_stop_button.setEnabled(True)
        else:
            # Batch is complete; re-enable the normalise button and disable the stop button.
            self.overlap_correction_correct_button.setEnabled(True)
            self.overlap_correction_stop_button.setEnabled(False)

        self._process_next_overlap_dataset()

    def stop_overlap_correction(self):
        """
        Stop the ongoing Overlap Correction process.
        """
        if hasattr(self, 'overlap_correction_worker') and self.overlap_correction_worker:
            self.overlap_correction_worker.stop()
            self.preproc_message_box.append("Stop signal sent to Overlap Correction process.")
            # logging.info("Stop signal sent to Overlap Correction process.")
            # Disable the stop button to prevent multiple stop signals
            self.overlap_correction_stop_button.setEnabled(False)
            self.overlap_correction_worker.quit()
            self.overlap_correction_worker.wait(1000)
            self.overlap_correction_worker = None
        else:
            self.preproc_message_box.append("No active Overlap Correction process to stop.")

    def overlap_correction_finished(self):
        if self.overlap_correction_worker:
            # self.preproc_message_box.append("Overlap Correction process finished.")
            self.overlap_correction_progress.setValue(0)
            self.overlap_correction_worker = None  # Cleanup
            # Re-enable the correct button
            self.overlap_correction_correct_button.setEnabled(True)
            self.overlap_correction_stop_button.setEnabled(False)

    def update_overlap_correction_progress(self, value):
        self.overlap_correction_progress.setValue(value)

    def remove_summation_images(self):
        if self.summation_image_runs:
            self.summation_image_runs = []
            self.preproc_message_box.append("Summation images have been removed.")
            self.summation_load_progress.setValue(0)
            self.summation_progress.setValue(0)
            self.summation_sum_button.setEnabled(False)
        else:
            self.preproc_message_box.append("No summation images to remove.")

    def handle_summation_run_loaded(self, folder_path, run_dict):
        if run_dict:
            # Add the run to summation_image_runs

            self.summation_image_runs.append({
                'folder_path': folder_path,
                'images': run_dict
            })

            run_number = len(self.summation_image_runs)
            self.preproc_message_box.append(f"Successfully loaded Summation Run {run_number} with {len(run_dict)} images.")
            self.summation_sum_button.setEnabled(True)

    def summation_image_loading_finished(self):
        if self.summation_image_load_worker:
            self.summation_image_load_worker = None  # Cleanup

    def update_summation_load_progress(self, value):
        self.summation_load_progress.setValue(value)

    def summation_finished(self):
        if self.summation_worker:
            self.preproc_message_box.append("Summation process finished.")
            self.summation_progress.setValue(0)
            self.summation_worker = None  # Cleanup
            # Re-enable the sum button if there are still runs loaded
            if self.summation_image_runs:
                self.summation_sum_button.setEnabled(True)
                self.summation_stop_button.setEnabled(False)
            else:
                self.summation_sum_button.setEnabled(False)
                self.summation_stop_button.setEnabled(False)
            QMessageBox.information(self, "Summation", "Summation process completed successfully!")

    def update_summation_progress(self, value):
        self.summation_progress.setValue(value)

    def add_summation_images(self):
        """
        The user selects a folder for summation.  We now enforce:

          • All children are *either* 3‑level (sample/run)
            *or* 2‑level (runs directly) – mixing is an error.
          • Each summation job needs ≥ 2 last‑level folders
            (runs) – otherwise we abort with a message.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing FITS Images for Summation", "")
        if not folder_path:
            return

        sample_folders = [
            os.path.join(folder_path, d)
            for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ]

        # -------------------------------------------------------------
        # Guard #1 – at least two children at the next level
        # -------------------------------------------------------------
        if len(sample_folders) < 2:                           # <-- NEW
            self.preproc_message_box.append(
                "❌  The selected folder contains only ONE sub‑folder – "
                "need at least two datasets for a summation.")
            return                                            # <-- NEW

        # -------------------------------------------------------------
        # Decide whether it is 2‑ or 3‑level … and detect mixing
        # -------------------------------------------------------------
        has_subfolders = [                                     # <-- NEW
            any(os.path.isdir(os.path.join(s, sub))
                for sub in os.listdir(s))
            for s in sample_folders
        ]

        if all(has_subfolders):            # ✔ pure 3‑level
            structure = "3"
        elif not any(has_subfolders):      # ✔ pure 2‑level
            structure = "2"
        else:                              # ❌ mixture
            self.preproc_message_box.append(
                "❌  Mixed folder structure detected – some samples contain "
                "sub‑folders (3‑level) while others do not (2‑level).  "
                "Please reorganise the data so all samples follow the same depth.")
            return                                         # <-- NEW

        # -------------------------------------------------------------
        # Continue exactly as before, but only for the chosen structure
        # -------------------------------------------------------------
        if structure == "3":
            self.preproc_message_box.append(
                "Detected three‑level structure (folder → sample → run).")
            self._summation_samples_3level = {
                sf: [os.path.join(sf, sub) for sub in os.listdir(sf)
                     if os.path.isdir(os.path.join(sf, sub))]
                for sf in sample_folders
            }
            self._summation_samples_2level = None
            self.summation_image_runs = []
        else:  # structure == "2"
            self.preproc_message_box.append(
                "Detected two‑level structure (folder → runs).")
            self._summation_samples_2level = sample_folders
            self._summation_samples_3level = None
            self.summation_image_runs = []

    def sum_images(self):
        """
        Summation entry point. We handle 3-level, 2-level, or 1-level structures.
        """
        # 3-level => multi-run
        if getattr(self, '_summation_samples_3level', None):
            self._init_summation_3level()
            return
        # 2-level => multiple subfolders
        if getattr(self, '_summation_samples_2level', None):
            self._init_summation_2level()
            return

        # Otherwise => 1-level => we use SummationWorker on self.summation_image_runs
        if not self.summation_image_runs:
            self.preproc_message_box.append("No summation images loaded. Please add image sets first.")
            return
        output_folder = self.summation_output_input.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            self.preproc_message_box.append("Please select a valid output folder to save summed images.")
            return
        base_name = self.summation_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append("Please enter a base name for the summed images.")
            return

        self.summation_sum_button.setEnabled(False)
        self._summation_cancelled = False
        self.summation_stop_button.setEnabled(True)
        self.summation_worker = SummationWorker(self.summation_image_runs, base_name, output_folder)
        self.summation_worker.progress_updated.connect(self.update_summation_progress)
        self.summation_worker.message.connect(self.preproc_message_box.append)
        self.summation_worker.finished.connect(self.summation_finished)
        self.summation_worker.start()

    def _init_summation_3level(self):
        self._expected_suffix_set = None       # <-- NEW
        self._expected_image_cnt  = None       # <-- NEW
        self._summation_cancelled = False      # (already there in most builds)


        output_folder = self.summation_output_input.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            self.preproc_message_box.append("Please select a valid output folder for multi-run summation.")
            return
        base_name = self.summation_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append("Please enter a base name for the summed images.")
            return
        bad_samples = [s for s, runs in self._summation_samples_3level.items()
                       if len(runs) < 2]                       # <-- NEW
        if bad_samples:                                        # <-- NEW
            self.preproc_message_box.append(
                "❌  The following samples have only ONE run and cannot be "
                "summed: " + ", ".join(map(os.path.basename, bad_samples)))
            return   

        self._batch_sum_output_folder = output_folder
        self._batch_sum_base_name = base_name
        self._summation_sample_keys_3 = list(self._summation_samples_3level.keys())
        self._current_sample_index_3 = 0

        self.preproc_message_box.append("Starting 3-level lazy summation (folder->sample->run).")
        self.summation_sum_button.setEnabled(False)
        self.summation_stop_button.setEnabled(True)
        self._process_next_sample_3level()

    def _process_next_sample_3level(self):
        if self._summation_cancelled:              # <-- NEW
            return

        if self._current_sample_index_3 >= len(self._summation_sample_keys_3):
            self.preproc_message_box.append("All multi-run samples have been processed.")
            self.summation_sum_button.setEnabled(True)
            self.summation_stop_button.setEnabled(False)
            QMessageBox.information(self, "Summation", "Batch summation (3-level) completed successfully!")
            return

        self._current_sample_folder_3 = self._summation_sample_keys_3[self._current_sample_index_3]
        self._current_sample_folder_3_short = self.get_short_path(self._current_sample_folder_3, levels=2)
        run_folders = self._summation_samples_3level[self._current_sample_folder_3]
        self.preproc_message_box.append(
            f"Processing sample: {self._current_sample_folder_3_short} with {len(run_folders)} run(s)."
        )
        # We'll store a single "combined run" including 'run_folders'
        self._combined_run_3 = {
            'folder_path': self._current_sample_folder_3,
            'images': {},
            # This is the critical part: store all actual run folders
            'run_folders': run_folders  
        }
        self._sample_run_folders_3 = run_folders
        self._current_run_index_3 = 0
        self._load_next_run_3level()

    def _load_next_run_3level(self):
        if self._summation_cancelled:            # <-- NEW
            return

        if self._current_run_index_3 >= len(self._sample_run_folders_3):
            self._finish_sample_3level()
            return

        current_run_folder = self._sample_run_folders_3[self._current_run_index_3]
        current_run_folder_short = self.get_short_path(current_run_folder, levels = 2)
        self.preproc_message_box.append(
            f"Loading run {self._current_run_index_3+1} of sample {os.path.basename(self._current_sample_folder_3_short)}: {current_run_folder_short}"
        )
        self._lazy_load_worker_3 = ImageLoadWorker(current_run_folder)
        self._lazy_load_worker_3.progress_updated.connect(self.update_summation_load_progress)
        self._lazy_load_worker_3.message.connect(self.preproc_message_box.append)
        self._lazy_load_worker_3.run_loaded.connect(
            lambda folder, run_dict: self._on_run_loaded_3level(folder, run_dict)
        )
        self._lazy_load_worker_3.finished.connect(self._on_run_finished_3level)
        self._lazy_load_worker_3.start()

    def _on_run_loaded_3level(self, folder, run_dict):
        if not run_dict or self._summation_cancelled:
            return

        # ---------------------------------------------------------------
        # NEW ❶ – compare suffix set / count with the first run
        # ---------------------------------------------------------------
        suffix_set = set(run_dict.keys())
        img_cnt    = len(suffix_set)

        if self._expected_suffix_set is None:
            # first run of the sample establishes the standard
            self._expected_suffix_set = suffix_set
            self._expected_image_cnt  = img_cnt
        else:
            if suffix_set != self._expected_suffix_set:
                self.preproc_message_box.append(
                    f"❌  Run '{folder}' has {img_cnt} images – expected "
                    f"{self._expected_image_cnt}.  Aborting summation.")
                self.stop_summation()                 # global abort
                return

        # ---- existing merge logic (unchanged) -------------------------
        for suffix, image_data in run_dict.items():
            if suffix not in self._combined_run_3['images']:
                self._combined_run_3['images'][suffix] = image_data.copy()
            else:
                self._combined_run_3['images'][suffix] += image_data
        gc.collect()

    def _on_run_finished_3level(self):
        if self._summation_cancelled:            # <-- NEW
            return
        self._current_run_index_3 += 1
        self._load_next_run_3level()

    def _finish_sample_3level(self):
        if self._summation_cancelled:            # <-- NEW
            return
        # Now we have a single combined run with images from all sub-runs, plus run_folders for shuttercount lookup
        self.preproc_message_box.append(f"All runs for sample <b>'{self._current_sample_folder_3_short}'</b> loaded and merged.")
        output_folder_run = os.path.join(
            self._batch_sum_output_folder,
            "Summed_" + os.path.basename(self._current_sample_folder_3)
        )
        if not os.path.exists(output_folder_run):
            try:
                os.makedirs(output_folder_run)
            except Exception as e:
                self.preproc_message_box.append(f"Failed to create output folder for {self._current_sample_folder_3_short}: {e}")
                self._current_sample_index_3 += 1
                self._process_next_sample_3level()
                return

        self.summation_worker = SummationWorker(
            [self._combined_run_3],
            self._batch_sum_base_name,
            output_folder_run
        )
        self.summation_worker.progress_updated.connect(self.update_summation_progress)
        self.summation_worker.message.connect(self.preproc_message_box.append)
        self.summation_worker.finished.connect(self._on_sample_summation_finished_3level)
        self.summation_worker.start()

    def _on_sample_summation_finished_3level(self):
        self.preproc_message_box.append(f"Finished summation for sample '{self._current_sample_folder_3_short}'.")
        self._combined_run_3 = None
        self.summation_worker = None
        gc.collect()
        self._current_sample_index_3 += 1
        self._process_next_sample_3level()

    def _init_summation_2level(self):

        self._expected_suffix_set = None       # <-- NEW
        self._expected_image_cnt  = None       # <-- NEW
        self._summation_cancelled = False      # (already there in most builds)

        output_folder = self.summation_output_input.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            self.preproc_message_box.append("Please select a valid output folder for two-level summation.")
            return
        base_name = self.summation_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append("Please enter a base name for the summed images.")
            return

        if not getattr(self, "_summation_samples_2level", None):
            self.preproc_message_box.append("❌  Internal error: 2‑level folder list is empty.")
            return

        if len(self._summation_samples_2level) < 2:              # <-- FIXED
            self.preproc_message_box.append(
                "❌  Need at least two sub‑folders (runs) for 2‑level summation.")
            return

        self._two_level_output_folder = output_folder
        self._two_level_base_name = base_name
        self._two_level_subfolders = list(self._summation_samples_2level)
        self._two_level_index = 0
        # We'll accumulate them into one combined run
        parent_folder_name = os.path.basename(os.path.commonpath(self._two_level_subfolders)) or "Combined"
        self._combined_run_2 = {
            'folder_path': parent_folder_name,
            'images': {},
            'run_folders': self._two_level_subfolders  # <-- store all subfolders here
        }
        self._summation_cancelled = False        # reset global stop flag
        self.preproc_message_box.append("Starting 2-level lazy summation.")
        self.summation_sum_button.setEnabled(False)
        self.summation_stop_button.setEnabled(True)
        self._load_next_run_2level()

    def _load_next_run_2level(self):
        if self._summation_cancelled:            # <-- NEW
            return

        if self._two_level_index >= len(self._two_level_subfolders):
            self._finish_2level_summation()
            return

        current_run_folder = self._two_level_subfolders[self._two_level_index]
        current_run_folder_short = self.get_short_path(current_run_folder, levels=2)

        self.preproc_message_box.append(f"Loading subfolder {self._two_level_index+1}: \\{current_run_folder_short}")
        self._lazy_load_worker_2 = ImageLoadWorker(current_run_folder)
        self._lazy_load_worker_2.progress_updated.connect(self.update_summation_load_progress)
        self._lazy_load_worker_2.message.connect(self.preproc_message_box.append)
        self._lazy_load_worker_2.run_loaded.connect(
            lambda folder, run_dict: self._on_run_loaded_2level(folder, run_dict)
        )
        self._lazy_load_worker_2.finished.connect(self._on_run_finished_2level)
        self._lazy_load_worker_2.start()

    def _on_run_loaded_2level(self, folder, run_dict):
        if not run_dict or self._summation_cancelled:
            return

        # ---------------------------------------------------------------
        # NEW ❷ – identical check for 2‑level
        # ---------------------------------------------------------------
        suffix_set = set(run_dict.keys())
        img_cnt    = len(suffix_set)

        if self._expected_suffix_set is None:
            self._expected_suffix_set = suffix_set
            self._expected_image_cnt  = img_cnt
        else:
            if suffix_set != self._expected_suffix_set:
                self.preproc_message_box.append(
                    f"❌  Sub‑folder '{folder}' has {img_cnt} images – "
                    f"expected {self._expected_image_cnt}.  "
                    "Aborting summation.")
                self.stop_summation()
                return

        # ---- existing merge logic (unchanged) -------------------------
        for suffix, image_data in run_dict.items():
            if suffix not in self._combined_run_2['images']:
                self._combined_run_2['images'][suffix] = image_data.copy()
            else:
                self._combined_run_2['images'][suffix] += image_data
        gc.collect()

    def _on_run_finished_2level(self):
        if self._summation_cancelled:            # <-- NEW
            return
        self._two_level_index += 1
        self._load_next_run_2level()

    def _finish_2level_summation(self):
        if self._summation_cancelled:            # <-- NEW
            return

        self.preproc_message_box.append("All 2-level subfolders loaded and merged => SummationWorker.")
        final_folder = os.path.join(
            self._two_level_output_folder,
            "Summed_" + self._combined_run_2['folder_path']
        )
        short_path = self.get_short_path(final_folder, levels=2)

        if not os.path.exists(final_folder):
            try:
                os.makedirs(final_folder)
            except Exception as e:
                self.preproc_message_box.append(f"Failed to create output folder \\{short_path}: {e}")
                self.summation_sum_button.setEnabled(True)
                self.summation_stop_button.setEnabled(False)
                return

        self.summation_worker = SummationWorker(
            [self._combined_run_2],
            self._two_level_base_name,
            final_folder
        )
        self.summation_worker.progress_updated.connect(self.update_summation_progress)
        self.summation_worker.message.connect(self.preproc_message_box.append)
        self.summation_worker.finished.connect(self._on_two_level_summation_finished)
        self.summation_worker.start()

    def _on_two_level_summation_finished(self):
        self.preproc_message_box.append("Finished summation for two-level folder structure.")
        self._combined_run_2 = None
        self.summation_worker = None
        gc.collect()
        self.summation_sum_button.setEnabled(True)
        self.summation_stop_button.setEnabled(False)
        QMessageBox.information(self, "Summation", "Summation process (2-level) completed successfully!")

    def stop_summation(self):
        self._summation_cancelled = True
        if getattr(self, 'summation_worker', None):
            self.summation_worker.stop()
        if getattr(self, '_lazy_load_worker_3', None):
            self._lazy_load_worker_3.terminate()   # optional: kill loader threads
        if getattr(self, '_lazy_load_worker_2', None):
            self._lazy_load_worker_2.terminate()
        self._summation_cancelled = True           # <-- NEW
        self.summation_sum_button.setEnabled(True)
        self.summation_stop_button.setEnabled(False)
        self.preproc_message_box.append("Stop signal sent – aborting all processes.")

    def add_normalisation_data_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Data Images for normalisation", ""
        )
        if folder_path:
            # Check if the selected folder has child folders.
            child_folders = [
                os.path.join(folder_path, item)
                for item in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, item))
            ]
            if child_folders:
                self.preproc_message_box.append(
                    f"Detected {len(child_folders)} sub-folders. They will be processed sequentially when normalisation starts."
                )
                self._normalisation_batch_paths = child_folders
            else:
                self.preproc_message_box.append(
                    "No sub-folders detected; the selected folder will be treated as a single dataset."
                )
                self._normalisation_batch_paths = [folder_path]

    def remove_normalisation_data_images(self):
        if self.normalisation_image_runs:
            self.normalisation_image_runs = []
            self.preproc_message_box.append("Normalisation data images have been removed.")
            self.normalisation_load_progress.setValue(0)
            self.normalisation_progress.setValue(0)
        else:
            self.preproc_message_box.append("No normalisation data images to remove.")

    def handle_normalisation_data_run_loaded(self, folder_path, run_dict):
        if run_dict:
            # Add the run to normalisation_image_runs
            self.normalisation_image_runs.append({
                'folder_path': folder_path,
                'images': run_dict
            })

    def normalisation_data_image_loading_finished(self):
        if self.normalisation_data_load_worker:
            self.normalisation_data_load_worker = None  # Cleanup

    def normalisation_open_beam_loading_finished(self):
        if self.open_beam_load_worker:
            self.open_beam_load_worker = None  # Cleanup
            self.preproc_message_box.append("Normalisation open beam image loading thread has finished.")

    def remove_normalisation_open_beam_images(self):
        if self.normalisation_open_beam_runs:
            self.normalisation_open_beam_runs = []
            self.preproc_message_box.append("Normalisation open beam images have been removed.")
        else:
            self.preproc_message_box.append("No normalisation open beam images to remove.")

    def normalise_images(self):
        # Ensure a sample dataset folder (or folders) has been selected.
        if not hasattr(self, '_normalisation_batch_paths') or not self._normalisation_batch_paths:
            self.preproc_message_box.append("No sample dataset folder selected. Please add data images first.")
            return

        # Check that an open beam dataset has been loaded.
        if not self.normalisation_open_beam_runs:
            self.preproc_message_box.append("No open beam data loaded. Please load open beam images first.")
            return

        overall_output_folder = self.normalisation_output_input.text().strip()
        if not overall_output_folder or not os.path.isdir(overall_output_folder):
            self.preproc_message_box.append("Please select a valid overall output folder to save normalised images.")
            return

        # Read UI parameters.
        try:
            self._window_half = int(self.normalisation_window_half_input.text().strip())
        except ValueError:
            self._window_half = 10

        try:
            self._adjacent_sum = int(self.normalisation_adjacent_input.text().strip())
            self._adjacent_sum = max(0, self._adjacent_sum)
        except ValueError:
            self._adjacent_sum = 0

        # self._adjacent_sum = 0

        self._base_name = self.normalisation_basename_input.text().strip() or "normalised"
        self._overall_output_folder = overall_output_folder

        # Prepare to process the batch sequentially.
        self._current_dataset_index = 0

        # Disable the normalise button to prevent duplicate runs.
        self.normalisation_normalise_button.setEnabled(False)
        self.normalisation_stop_button.setEnabled(True)

        self.preproc_message_box.append("Starting batch normalisation (lazy loading)...")
        self._process_next_dataset()

    def _process_next_dataset(self):
        """
        Process the next sample dataset in the batch.
        Loads the dataset, then normalises it with the open beam dataset.
        """
        if self._current_dataset_index >= len(self._normalisation_batch_paths):
            self.preproc_message_box.append("<b>Normalisation completed.</b>")
            self.normalisation_normalise_button.setEnabled(True)
            self.normalisation_stop_button.setEnabled(False)
            return

        current_folder = self._normalisation_batch_paths[self._current_dataset_index]

        short_path = self.get_short_path(current_folder, levels=2)

        self.preproc_message_box.append(f"Loading dataset from folder: <b> {short_path} </b>")
        # Launch the data loading worker for the current folder.
        self._data_load_worker = ImageLoadWorker(current_folder)
        self._data_load_worker.progress_updated.connect(self.update_normalisation_load_progress)
        self._data_load_worker.message.connect(self.preproc_message_box.append)
        self._data_load_worker.run_loaded.connect(self._on_dataset_loaded)
        self._data_load_worker.finished.connect(self._on_data_load_finished)
        self._data_load_worker.start()

    def _on_dataset_loaded(self, folder_path, run_dict):
        # Store the loaded dataset temporarily.
        if run_dict:
            self._current_loaded_run = {'folder_path': folder_path, 'images': run_dict}
        else:
            self._current_loaded_run = None

    def _on_data_load_finished(self):
        # Once data loading is finished, start normalisation for the loaded dataset.
        if not self._current_loaded_run:
            self.preproc_message_box.append("Failed to load dataset, skipping...")
            self._current_dataset_index += 1
            self._process_next_dataset()
            return

        sample_run = self._current_loaded_run
        folder_path = sample_run['folder_path']

        # normalized_path = os.path.normpath(folder_path)
        # path_parts = normalized_path.split(os.sep)
        # if len(path_parts) >= 2:
        #     short_path = os.path.join(path_parts[-2], path_parts[-1])
        # else:
        #     short_path = normalized_path
        short_path = self.get_short_path(folder_path, levels=2)       

        # Create an output folder specific to this dataset.
        output_folder_run = os.path.join(self._overall_output_folder, "normalised_" + os.path.basename(folder_path))
        try:
            if not os.path.exists(output_folder_run):
                os.makedirs(output_folder_run)
        except Exception as e:
            self.preproc_message_box.append(f"Failed to create output folder for \\{short_path}: <b>{str(e)}</b>")
            self._current_dataset_index += 1
            self._process_next_dataset()
            return

        self.preproc_message_box.append(f"Starting normalisation for dataset: <b> \\{short_path} </b>")
        # Use the open beam dataset (assumed to be the first one loaded).
        open_beam_run = self.normalisation_open_beam_runs[0]
        # Start the normalisation worker.
        self.normalisation_worker = NormalisationWorker(
            [sample_run],              # Pass the loaded sample dataset as a single-item list.
            [open_beam_run],           # Use the open beam dataset.
            output_folder_run,
            self._base_name,
            self._window_half,
            adjacent_sum=self._adjacent_sum
        )
        self.normalisation_worker.progress_updated.connect(self.update_normalisation_progress)
        self.normalisation_worker.message.connect(self.preproc_message_box.append)
        self.normalisation_worker.finished.connect(self._on_normalisation_finished)
        self.normalisation_worker.start()

    def _on_normalisation_finished(self):
        """
        Callback when a normalisation run finishes.
        Clean up and, if there are more datasets, re-enable the stop button.
        """
        self.preproc_message_box.append(
            f"Finished normalisation for dataset {self._current_dataset_index+1} of {len(self._normalisation_batch_paths)}."
        )
        # Clean up references.
        self._current_loaded_run = None
        self._data_load_worker = None
        self.normalisation_worker = None
        import gc
        gc.collect()
        self._current_dataset_index += 1

        # If there are more datasets, re-enable the stop button for the next run.
        if self._current_dataset_index < len(self._normalisation_batch_paths):
            self.normalisation_stop_button.setEnabled(True)
        else:
            # Batch is complete; re-enable the normalise button and disable the stop button.
            self.normalisation_normalise_button.setEnabled(True)
            self.normalisation_stop_button.setEnabled(False)

        # Start processing the next dataset.
        self._process_next_dataset()

    def _start_next_normalisation_run(self):
        """
        Start normalisation for the next sample dataset in batch mode.
        Creates an output folder named "normalised_<foldername>" under the overall output folder.
        """
        if self._current_normalisation_index >= len(self._normalisation_batch_runs):
            self.preproc_message_box.append("Batch normalisation completed.")
            self.normalisation_normalise_button.setEnabled(True)
            self.normalisation_stop_button.setEnabled(False)
            return

        sample_run = self._normalisation_batch_runs[self._current_normalisation_index]
        sample_folder = sample_run['folder_path']
        output_folder_run = os.path.join(self._overall_output_folder, "normalised_" + os.path.basename(sample_folder))
        try:
            if not os.path.exists(output_folder_run):
                os.makedirs(output_folder_run)
        except Exception as e:
            self.preproc_message_box.append(f"Failed to create output folder for {sample_folder}: {str(e)}")
            self._current_normalisation_index += 1
            self._start_next_normalisation_run()
            return

        self.preproc_message_box.append(f"Normalising data from folder: {sample_folder}")
        self.normalisation_worker = NormalisationWorker(
            [sample_run],              # Single sample run in a list
            [self._open_beam_run],       # Single open beam run in a list
            output_folder_run,
            self._base_name,
            self._window_half,
            adjacent_sum=self._adjacent_sum
        )
        self.normalisation_worker.progress_updated.connect(self.update_normalisation_progress)
        self.normalisation_worker.message.connect(self.preproc_message_box.append)
        self.normalisation_worker.finished.connect(self._on_individual_normalisation_finished)
        self.normalisation_worker.start()

    def _on_individual_normalisation_finished(self):
        """
        Callback for when an individual normalisation run finishes.
        Moves to the next sample dataset in the batch.
        """
        self.preproc_message_box.append(
            f"Finished normalisation for run {self._current_normalisation_index + 1} of {len(self._normalisation_batch_runs)}."
        )
        self._current_normalisation_index += 1
        self._start_next_normalisation_run()

    def update_normalisation_progress(self, value):
        self.normalisation_progress.setValue(value)

    def stop_normalisation(self):
        """
        Stop the ongoing normalisation process.
        """
        if hasattr(self, 'normalisation_worker') and self.normalisation_worker:
            self.normalisation_worker.stop()
            self.preproc_message_box.append("Stop signal sent to Normalisation process.")
            # logging.info("Stop signal sent to normalisation process.")
            # Disable the stop button to prevent multiple stop signals
            self.normalisation_stop_button.setEnabled(False)
            self.normalisation_worker.quit()
            self.normalisation_worker.wait(1000)
            self.normalisation_worker = None
        else:
            self.preproc_message_box.append("No active Normalisation process to stop.")

    def normalisation_finished(self):
        if self.normalisation_worker:
            # self.preproc_message_box.append("normalisation process finished.")
            self.normalisation_progress.setValue(0)
            self.normalisation_worker = None  # Cleanup
            # self.remove_normalisation_data_images()
            # self.remove_normalisation_open_beam_images()
            # Re-enable the normalise button
            self.normalisation_normalise_button.setEnabled(True)
            self.normalisation_stop_button.setEnabled(False)

    def load_open_beam_images(self):
        # Open a folder, start OpenBeamLoadWorker, connect signals
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Open Beam Images", "")
        if folder_path:
            # Create and start the worker
            self.open_beam_load_worker = OpenBeamLoadWorker(folder_path, area_x=(10, 500), area_y=(10, 500))
            self.open_beam_load_worker.progress_updated.connect(self.update_normalisation_load_progress)
            self.open_beam_load_worker.message.connect(self.preproc_message_box.append)
            self.open_beam_load_worker.open_beam_loaded.connect(self.handle_normalisation_open_beam_loaded)
            self.open_beam_load_worker.finished.connect(self.open_beam_loading_finished)
            self.open_beam_load_worker.start()

    def handle_normalisation_open_beam_loaded(self, folder_path, summed_intensities):
        if not summed_intensities:
            return

        self.normalisation_open_beam_runs.append({
            "folder_path": folder_path,       # ← real directory
            "images"     : summed_intensities
        })

        # ----- the plotting block stays the same ----------------------
        wavelengths       = self.wavelengths
        sorted_suffixes   = sorted(summed_intensities)
        sorted_intensities = [summed_intensities[suf] for suf in sorted_suffixes]

    def update_normalisation_load_progress(self, value):
        self.normalisation_load_progress.setValue(value)

    def open_beam_loading_finished(self):
        if self.open_beam_load_worker:
            self.open_beam_load_worker = None  # Cleanup

    def browse_output_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            line_edit.setText(folder)

    def filter_images(self):
        """
        Triggered by the Filter button to start the filtering process on loaded FITS data images
        with the specified FITS mask image.
        """
        # Check that data images have been loaded
        if not hasattr(self, 'filtering_image_runs') or not self.filtering_image_runs:
            self.preproc_message_box.append("No data images loaded for filtering. Please add data images first.")
            return

        # Check that a mask image has been loaded
        if not hasattr(self, 'filtering_mask_image') or self.filtering_mask_image is None:
            self.preproc_message_box.append("No mask image loaded. Please add a FITS mask image first.")
            return

        # Get the output folder
        output_folder = self.filtering_output_input.text().strip()
        if not output_folder:
            self.preproc_message_box.append("Please select an output folder to save filtered images.")
            return
        if not os.path.isdir(output_folder):
            self.preproc_message_box.append(f"Specified output folder '{output_folder}' does not exist.")
            return

        # Get the base name from the input box
        base_name = self.filtering_basename_input.text().strip()
        if not base_name:
            self.preproc_message_box.append("Please enter a base name for filtered images.")
            return

        # Disable the filter button to prevent multiple concurrent filtering operations
        self.filtering_filter_button.setEnabled(False)
        # Enable the stop button
        if hasattr(self, 'filtering_stop_button'):
            self.filtering_stop_button.setEnabled(True)

        # Start filtering in a separate thread
        self.filtering_worker = FilteringWorker(
            filtering_image_runs=self.filtering_image_runs,
            filtering_mask=self.filtering_mask_image,
            output_folder=output_folder,
            base_name=base_name
        )

        # Connect signals
        self.filtering_worker.progress_updated.connect(self.update_filtering_progress)
        self.filtering_worker.message.connect(self.preproc_message_box.append)
        self.filtering_worker.finished.connect(self.filtering_finished)

        self.filtering_worker.start()
        self.preproc_message_box.append("--- Starting Filtering ---")

    def stop_filtering(self):
        """
        Stop the ongoing Filtering process, if running.
        """
        if hasattr(self, 'filtering_worker') and self.filtering_worker:
            self.filtering_worker.stop()
            self.preproc_message_box.append("Stop signal sent to Filtering process.")
            # logging.info("Stop signal sent to Filtering process.")
            self.filtering_worker.quit()
            self.filtering_worker.wait(1000)
            self.filtering_worker = None
            if hasattr(self, 'filtering_stop_button'):
                self.filtering_stop_button.setEnabled(False)
        else:
            self.preproc_message_box.append("No active Filtering process to stop.")

    def update_filtering_progress(self, value: int):
        """
        Update the filtering progress bar.
        """
        if hasattr(self, 'filtering_progress'):
            self.filtering_progress.setValue(value)

    def filtering_finished(self):
        """
        Called once filtering is complete or stopped.
        """
        self.preproc_message_box.append("Filtering process finished.")
        self.filtering_progress.setValue(0)

        # Clean up worker
        if hasattr(self, 'filtering_worker'):
            self.filtering_worker = None

        # Re-enable the filter button and disable the stop button
        self.filtering_filter_button.setEnabled(True)
        if hasattr(self, 'filtering_stop_button'):
            self.filtering_stop_button.setEnabled(False)

    def add_filter_data_images(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing FITS Data Images", "")
        if folder_path:
            # Similar to your normalisation worker approach:
            self.filtering_data_load_worker = ImageLoadWorker(folder_path)
            self.filtering_data_load_worker.progress_updated.connect(self.update_filtering_load_progress)
            self.filtering_data_load_worker.message.connect(self.preproc_message_box.append)
            self.filtering_data_load_worker.run_loaded.connect(self.handle_filtering_data_run_loaded)
            self.filtering_data_load_worker.finished.connect(self.filtering_data_loading_finished)
            self.filtering_data_load_worker.start()

    def handle_filtering_data_run_loaded(self, folder_path, run_dict):
        # run_dict is suffix -> 2D numpy array
        if not hasattr(self, 'filtering_image_runs'):
            self.filtering_image_runs = []
        self.filtering_image_runs.append({
            'folder_path': folder_path,
            'images': run_dict
        })

    def filtering_data_loading_finished(self):
        self.filtering_data_load_worker = None

    def remove_filter_data_images(self):
        if hasattr(self, 'filtering_image_runs') and self.filtering_image_runs:
            self.filtering_image_runs = []
            self.preproc_message_box.append("Filtering data images have been removed.")
            self.filtering_load_progress.setValue(0)
            self.filtering_progress.setValue(0)
        else:
            self.preproc_message_box.append("No filtering data images to remove.")

    def add_filter_mask_image(self):
        # Single-file selection for the FITS mask:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select FITS Mask Image", "", "FITS Files (*.fits)")

        file_path_short = self.get_short_path(file_path, levels=2)
        if file_path:
            try:
                with fits.open(file_path) as hdul:
                    self.filtering_mask_image = hdul[0].data
                self.preproc_message_box.append(f"Mask image loaded from: {file_path_short}")
            except Exception as e:
                self.preproc_message_box.append(f"Failed to load mask image from {file_path}: {e}")

    def remove_filter_mask_image(self):
        if hasattr(self, 'filtering_mask_image') and self.filtering_mask_image is not None:
            self.filtering_mask_image = None
            self.preproc_message_box.append("Filtering mask image has been removed.")
        else:
            self.preproc_message_box.append("No filtering mask image to remove.")

    def update_filtering_load_progress(self, value):
        if hasattr(self, 'filtering_load_progress'):
            self.filtering_load_progress.setValue(value)

    def handle_open_beam_loaded(self, summed_intensities):
        if summed_intensities:
            self.open_beam_runs.append(summed_intensities)  # Add to runs
            # Generate wavelength array assuming same as data images
            wavelengths = self.wavelengths  # Assuming same as data
            # For plotting, we need the intensities in order.
            # Since summed_intensities is now a dict mapping suffix to intensity, we need to order them.
            # Let's get the sorted suffixes:
            sorted_suffixes = sorted(summed_intensities.keys())
            sorted_intensities = [summed_intensities[suffix] for suffix in sorted_suffixes]
            # Plot intensity vs wavelength in a pop-up window
            dialog = OpenBeamPlotDialog(wavelengths, sorted_intensities, parent=self)
            self.open_beam_plot_dialogs.append(dialog)  # Keep a reference
            dialog.show()
