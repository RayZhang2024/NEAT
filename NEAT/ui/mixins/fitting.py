"""Fitting tab functionality."""

import gc
import glob
import os
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QListWidget,
    QWidget,
    QSlider,
    QLabel,
    QLineEdit,
    QGridLayout,
    QProgressBar,
    QProgressDialog,
    QMessageBox,
    QTabWidget,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
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
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit, least_squares

from ...core import (
    PHASE_DATA,
    calculate_x_hkl_general,
    fitting_function_1,
    fitting_function_2,
    fitting_function_3,
)
from ...workers.batch import (
    BatchFitEdgesWorker,
    BatchFitWorker,
    ImageLoadWorker,
)
from ..dialogs import (
    AdjustmentsDialog,
    FitVisualizationDialog,
    MplCanvas,
    OpenBeamPlotDialog,
)


class FittingMixin:
    def _build_batch_fit_context(self):
        """Snapshot fit inputs from UI so worker threads do not read widgets directly."""
        def _safe_float_text(text):
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        table = self.bragg_table
        selected_phase = self.phase_dropdown.currentText() if hasattr(self, "phase_dropdown") else "Unknown_Phase"
        bragg_rows = []
        bragg_rows_text = []

        for row_idx in range(table.rowCount()):
            row_items = []
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                row_items.append(cell_item.text() if cell_item else "")
            bragg_rows_text.append("|".join(v.replace(",", ";") for v in row_items))

            hkl_tuple = None
            hkl_text = row_items[0].strip().strip("()") if row_items else ""
            if hkl_text:
                try:
                    h, k, l = map(int, [v.strip() for v in hkl_text.split(",")])
                    hkl_tuple = (h, k, l)
                except (TypeError, ValueError):
                    hkl_tuple = None

            regions = []
            valid_regions = True
            for cmin, cmax in ((2, 3), (4, 5), (6, 7)):
                min_w = _safe_float_text(row_items[cmin] if len(row_items) > cmin else "")
                max_w = _safe_float_text(row_items[cmax] if len(row_items) > cmax else "")
                if min_w is None or max_w is None or min_w >= max_w:
                    valid_regions = False
                    break
                regions.append({"min_wavelength": min_w, "max_wavelength": max_w})

            s_val = _safe_float_text(row_items[8] if len(row_items) > 8 else "")
            t_val = _safe_float_text(row_items[9] if len(row_items) > 9 else "")
            eta_val = _safe_float_text(row_items[10] if len(row_items) > 10 else "")
            d_val = _safe_float_text(row_items[1] if len(row_items) > 1 else "")

            bragg_rows.append(
                {
                    "row": row_idx,
                    "hkl": hkl_tuple,
                    "d": d_val,
                    "regions": regions if valid_regions else None,
                    "s": s_val,
                    "t": t_val,
                    "eta": eta_val,
                    "valid": bool(
                        valid_regions
                        and s_val is not None
                        and t_val is not None
                        and eta_val is not None
                        and (hkl_tuple is not None or selected_phase == "Unknown_Phase")
                    ),
                }
            )

        return {
            "structure_type": getattr(self, "structure_type", "cubic"),
            "lattice_params": dict(getattr(self, "lattice_params", {})),
            "selected_phase": selected_phase,
            "flight_path": getattr(self, "flight_path", None),
            "min_wavelength": _safe_float_text(self.min_wavelength_input.text() if hasattr(self, "min_wavelength_input") else ""),
            "max_wavelength": _safe_float_text(self.max_wavelength_input.text() if hasattr(self, "max_wavelength_input") else ""),
            "bragg_rows": bragg_rows,
            "bragg_rows_text": bragg_rows_text,
        }

    def setup_FittingTab(self):
        """
        Bragg edge fitting
        """
        self.plot_font_size = getattr(self, "plot_font_size", 12)
        self.symbol_size = getattr(self, "symbol_size", 4)
        self._fit_canvas_nav_state = {}

        main_layout = QHBoxLayout(self.FittingTab)  # Main horizontal layout

        # Create splitter for left and right panels
        main_splitter = QSplitter(Qt.Horizontal)

        # Left container
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # Use a second splitter to separate top and bottom within the left container
        left_splitter = QSplitter(Qt.Vertical)

        # --- Upper left widget (contains a canvas) ---
        upper_left_widget = QWidget()
        upper_left_layout = QVBoxLayout(upper_left_widget)

        # Matplotlib canvas
        self.canvas = MplCanvas(self, width=4, height=4, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setToolTip(
            "Drag inside a box to move it. Ctrl+drag a corner to resize it. "
            "Drag outside boxes to pan the image."
        )

        # Create and add Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
        upper_left_layout.addWidget(self.toolbar) 

        upper_left_layout.addWidget(self.canvas)
        upper_left_widget.setMinimumSize(200, 200)

        # Enable dragging of the yellow macro‑pixel ROI directly on the canvas
        self._dragging_roi = False
        self._roi_drag_offset = (0.0, 0.0)
        self._dragging_roi_mode = "move"
        self._roi_active_corner = None
        # Enable dragging of the orange batch ROI as well
        self._dragging_batch_roi = False
        self._batch_drag_offset = (0.0, 0.0)
        self._dragging_batch_mode = "move"
        self._batch_active_corner = None
        self._dragging_image_pan = False
        self._image_pan_start_xpixel = None
        self._image_pan_start_ypixel = None
        self._image_pan_start_xlim = None
        self._image_pan_start_ylim = None
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)


        # --- Lower left widget (contains all controls) ---
        lower_left_widget = QWidget()
        lower_left_layout = QVBoxLayout(lower_left_widget)

        # Put top and bottom widgets into the left splitter
        left_splitter.addWidget(upper_left_widget)
        left_splitter.addWidget(lower_left_widget)

        # Let the top and bottom areas grow proportionally (stretch factors)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        left_container.setMinimumWidth(300)


        # Add the left splitter into the left_layout
        left_layout.addWidget(left_splitter)

        # ============== Controls in Lower Left Layout ==============
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_fits_images)
        self.load_button.setToolTip('Select a folder to load fits images')
        button_layout.addWidget(self.load_button)

        self.config_button = QPushButton("Config")
        self.config_button.setToolTip('Configure fitting display and metadata settings')
        self.config_button.clicked.connect(self.open_general_settings_dialog)
        button_layout.addWidget(self.config_button)

        self.smooth_checkbox = QCheckBox("Smooth")
        self.smooth_checkbox.setChecked(False)
        button_layout.addWidget(self.smooth_checkbox)

        self.phase_dropdown = QComboBox()
        self.phase_placeholder = "Select Phase"
        self.phase_dropdown.addItem(self.phase_placeholder)  # Placeholder/default item
        self.phase_dropdown.setToolTip('Select a phase from the drop down list, or select "unknown_phase" if the material is unknown')
        self.phase_dropdown.addItems(sorted(getattr(self, "phase_data", PHASE_DATA).keys()))  # Dynamically add phase names
        self.phase_dropdown.currentIndexChanged.connect(self.phase_selection_changed)  # Connect to handler
        button_layout.addWidget(self.phase_dropdown)

        self.add_phase_button = QPushButton("Manage")
        self.add_phase_button.setToolTip("Add, edit, or delete phases")
        self.add_phase_button.clicked.connect(self.open_add_phase_dialog)
        button_layout.addWidget(self.add_phase_button)

        lower_left_layout.addLayout(button_layout)

        # Slider
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setEnabled(False)  # Disabled until images are loaded
        self.image_slider.setMinimumWidth(300)
        self.image_slider.valueChanged.connect(self.update_image)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Image Selection"))

        self.show_theoretical_checkbox = QCheckBox("Show")
        self.show_theoretical_checkbox.setToolTip('Check to show the theorectical edges of a selected phase')
        self.show_theoretical_checkbox.setChecked(True)  # Default to shown
        self.show_theoretical_checkbox.stateChanged.connect(self.update_plots)  # Trigger plot update on toggle
        slider_layout.addWidget(self.show_theoretical_checkbox)
        self.fix_s_checkbox = QCheckBox("Fix s")
        self.fix_s_checkbox.setToolTip(
            "Keep the Bragg-edge broadening parameter s fixed at the value in the table. "
            "Uncheck to fit s for each selected edge."
        )
        self.fix_s_checkbox.setChecked(True)
        slider_layout.addWidget(self.fix_s_checkbox)

        self.fix_t_checkbox = QCheckBox("Fix t")
        self.fix_t_checkbox.setToolTip(
            "Keep the moderator decay parameter t fixed at the value in the table. "
            "Uncheck to fit t for each selected edge."
        )
        self.fix_t_checkbox.setChecked(True)
        slider_layout.addWidget(self.fix_t_checkbox)

        self.fix_eta_checkbox = QCheckBox("Fix eta")
        self.fix_eta_checkbox.setToolTip(
            "Keep the pseudo-Voigt mixing parameter eta fixed at the value in the table. "
            "Uncheck to fit eta; eta=0 is Gaussian-like and eta=1 is Lorentzian-like."
        )
        self.fix_eta_checkbox.setChecked(True)
        slider_layout.addWidget(self.fix_eta_checkbox)

        lower_left_layout.addLayout(slider_layout)
        lower_left_layout.addWidget(self.image_slider)

        # Rectangle selection inputs
        roi_bounds_tooltip = self._roi_bounds_tooltip(
            "Initial fitting ROI",
            "x[0, 1), y[0, 1) includes only pixel (0, 0).",
        )
        selection_layout = QGridLayout()
        self.xmin_input = QLineEdit("300")
        self.xmin_input.setToolTip(roi_bounds_tooltip)
        self.xmax_input = QLineEdit("320")
        self.xmax_input.setToolTip(roi_bounds_tooltip)
        self.ymin_input = QLineEdit("300")
        self.ymin_input.setToolTip(roi_bounds_tooltip)
        self.ymax_input = QLineEdit("320")
        self.ymax_input.setToolTip(roi_bounds_tooltip)

        selection_layout.addWidget(QLabel("X Min:"), 0, 0)
        selection_layout.addWidget(self.xmin_input, 0, 1)
        selection_layout.addWidget(QLabel("X Max:"), 0, 2)
        selection_layout.addWidget(self.xmax_input, 0, 3)
        selection_layout.addWidget(QLabel("Y Min:"), 0, 4)
        selection_layout.addWidget(self.ymin_input, 0, 5)
        selection_layout.addWidget(QLabel("Y Max:"), 0, 6)
        selection_layout.addWidget(self.ymax_input, 0, 7)

        self.select_area_button = QPushButton("Pick")
        self.select_area_button.clicked.connect(self.select_area)
        # self.select_area_button.setStyleSheet("""
        #     QPushButton {
        #         font-size: 32px;
        #         background-color: #4CAF50;
        #         color: white;
        #         border: none;
        #         border-radius: 3px;
        #     }
        #     QPushButton:hover {
        #         background-color: #45a049;
        #     }
        # """)
        self.select_area_button.setToolTip('Click to display intensity vs wavelength from the selected area')
        selection_layout.addWidget(self.select_area_button, 0, 8)

        self.export_data_button = QPushButton("Export")
        self.export_data_button.setToolTip('Click to export intensity vs wavelength profile')
        self.export_data_button.clicked.connect(self.export_data)
        selection_layout.addWidget(self.export_data_button, 0, 9)

        lower_left_layout.addLayout(selection_layout)

        # Wavelength window inputs
        wavelength_layout = QGridLayout()
        self.min_wavelength_input = QLineEdit("2.1")
        self.min_wavelength_input.setToolTip('Set lower bound of wavelength range')
        button_layout.addWidget(QLabel("Min WL (Å):"))
        button_layout.addWidget(self.min_wavelength_input)
        self.max_wavelength_input = QLineEdit("3.4")
        self.max_wavelength_input.setToolTip('Set upper bound of wavelength range')
        button_layout.addWidget(QLabel("Max WL (Å):"))
        button_layout.addWidget(self.max_wavelength_input)

        lower_left_layout.addLayout(wavelength_layout)

        # Bragg Table
        self.bragg_table = QTableWidget()
        self.bragg_table.setColumnCount(11)
        # Start with an empty table; only populate after user clicks "Pick"
        self._bragg_table_ready = False
        self.bragg_table.setRowCount(0)

        self.bragg_table.setHorizontalHeaderLabels([
            "hkl", "d",
            "1 Min", "1 Max",
            "2 Min", "2 Max",
            "3 Min", "3 Max",
            "s", "t", "eta"
        ])

        self.bragg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bragg_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked | QAbstractItemView.EditKeyPressed)
        self.bragg_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bragg_table.setSelectionMode(QTableWidget.SingleSelection)
        self.bragg_table.itemSelectionChanged.connect(self.on_bragg_edge_selected)
        lower_left_layout.addWidget(self.bragg_table)

        # 1) Define the tooltip text, in the same order as the columns
        header_tips = [
            "Miller indices of the Bragg edge, written as (h, k, l).",
            "Theoretical d-spacing for this hkl, in Angstrom.",
            "Region 1 lower wavelength bound. Region 1 is the pre-edge baseline used in edge fitting.",
            "Region 1 upper wavelength bound. Region 1 is the pre-edge baseline used in edge fitting.",
            "Region 2 lower wavelength bound. Region 2 is the post-edge baseline used in edge fitting.",
            "Region 2 upper wavelength bound. Region 2 is the post-edge baseline used in edge fitting.",
            "Region 3 lower wavelength bound. Region 3 is the full edge window used for the fit.",
            "Region 3 upper wavelength bound. Region 3 is the full edge window used for the fit.",
            "Initial/fixed value for s, the edge broadening parameter. Controlled by the Fix s checkbox.",
            "Initial/fixed value for t, the moderator decay parameter. Controlled by the Fix t checkbox.",
            "Initial/fixed value for eta, the pseudo-Voigt mixing parameter. 0 is Gaussian-like, 1 is Lorentzian-like."
        ]

        # 2) Attach each tooltip
        for col, tip in enumerate(header_tips):
            item = self.bragg_table.horizontalHeaderItem(col)
            if item is not None:                      # safety-check
                item.setToolTip(tip)

        # Connect phase/wavelength changes
        self.phase_dropdown.currentIndexChanged.connect(self.update_bragg_edge_table)
        self.min_wavelength_input.editingFinished.connect(self.update_bragg_edge_table)
        self.max_wavelength_input.editingFinished.connect(self.update_bragg_edge_table)

        # Fitting Buttons
        fit_buttons_layout = QHBoxLayout()
        self.fit_all_regions_button = QPushButton("Fit edges")
        self.fit_all_regions_button.setToolTip('Perform individual edge fitting')
        self.fit_all_regions_button.clicked.connect(self.fit_all_regions)
        fit_buttons_layout.addWidget(self.fit_all_regions_button)

        self.batch_fit_edges_button = QPushButton("Batch Edges")
        self.batch_fit_edges_button.setToolTip('Start batch fitting of individual edge across the selected ROI')
        self.batch_fit_edges_button.clicked.connect(self.batch_fit_edges)
        fit_buttons_layout.addWidget(self.batch_fit_edges_button)

        self.fit_full_pattern_button = QPushButton("Fit Pattern")
        self.fit_full_pattern_button.setToolTip('Perform Bragg edge pattern fitting')
        self.fit_full_pattern_button.clicked.connect(self.fit_full_pattern)
        fit_buttons_layout.addWidget(self.fit_full_pattern_button)

        self.batch_fit_button = QPushButton("Batch Pattern")
        self.batch_fit_button.setToolTip('Start batch fitting of edge pattern across the selected ROI')
        self.batch_fit_button.clicked.connect(self.batch_fit)
        fit_buttons_layout.addWidget(self.batch_fit_button)

        self.batch_settings_button = QPushButton("Set")
        self.batch_settings_button.setToolTip("Configure batch fitting parameters")
        self.batch_settings_button.clicked.connect(self.open_batch_settings_dialog)
        fit_buttons_layout.addWidget(self.batch_settings_button)

        self.stop_batch_fit_button = QPushButton("Stop")
        self.stop_batch_fit_button.setToolTip('Stop batch fitting')
        self.stop_batch_fit_button.clicked.connect(self.stop_batch_fit)
        fit_buttons_layout.addWidget(self.stop_batch_fit_button)

        self.visualize_button = QPushButton("Fit Check")
        self.visualize_button.setToolTip('Illustrate fitting at specific locations')
        self.visualize_button.clicked.connect(self.visualize_region3_fits)
        fit_buttons_layout.addWidget(self.visualize_button)

        lower_left_layout.addLayout(fit_buttons_layout)

        # Stored batch-fit inputs (configured via dialog)
        self.box_width_input = QLineEdit("20", self)
        self.box_height_input = QLineEdit("20", self)
        self.step_x_input = QLineEdit("5", self)
        self.step_y_input = QLineEdit("5", self)
        self.interpolation_checkbox = QCheckBox("Inter", self)
        self.interpolation_checkbox.setChecked(True)
        self.min_x_input = QLineEdit("200", self)
        self.max_x_input = QLineEdit("500", self)
        self.min_y_input = QLineEdit("100", self)
        self.max_y_input = QLineEdit("400", self)
        for widget in [
            self.box_width_input,
            self.box_height_input,
            self.step_x_input,
            self.step_y_input,
            self.interpolation_checkbox,
            self.min_x_input,
            self.max_x_input,
            self.min_y_input,
            self.max_y_input,
        ]:
            widget.hide()

        # Shortcuts for plot font adjustments
        shortcut_increase_plot = QShortcut(QKeySequence("Ctrl+Up"), self)
        shortcut_increase_plot.activated.connect(self.increase_plot_font_size)
        shortcut_decrease_plot = QShortcut(QKeySequence("Ctrl+Down"), self)
        shortcut_decrease_plot.activated.connect(self.decrease_plot_font_size)
        self.update_plot_font_size(self.plot_font_size)

        # Progress dialog for batch operations
        self.batch_progress_dialog = QDialog(self)
        self.batch_progress_dialog.setWindowTitle("Batch Progress")
        self.batch_progress_dialog.setModal(False)
        dialog_layout = QVBoxLayout(self.batch_progress_dialog)
        self.batch_progress_label = QLabel("Fitting Progress:")
        self.batch_progress_bar = QProgressBar()
        self.batch_remaining_time_label = QLabel("Remaining:")
        self.batch_stop_button = QPushButton("Stop")
        self.batch_stop_button.clicked.connect(self.stop_batch_fit)
        dialog_layout.addWidget(self.batch_progress_label)
        dialog_layout.addWidget(self.batch_progress_bar)
        dialog_layout.addWidget(self.batch_remaining_time_label)
        dialog_layout.addWidget(self.batch_stop_button)
        self.batch_progress_dialog.hide()
        self.batch_progress_dialog.setWindowModality(Qt.NonModal)

        # Assign left_container (with its layout) into the main splitter
        main_splitter.addWidget(left_container)

        # ============== RIGHT SIDE (plots + text areas) ==============
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_splitter = QSplitter(Qt.Vertical)

        upper_right_widget = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_widget)

        # Plots layout
        plots_layout = QGridLayout()

        self.plot_canvas_a = MplCanvas(self, width=4, height=4, dpi=100)
        self.plot_canvas_b = MplCanvas(self, width=4, height=4, dpi=100)
        self.plot_canvas_c = MplCanvas(self, width=4, height=4, dpi=100)
        self.plot_canvas_d = MplCanvas(self, width=4, height=4, dpi=100)

        self.plot_canvas_a.setMinimumSize(200, 200)
        self.plot_canvas_b.setMinimumSize(200, 200)
        self.plot_canvas_c.setMinimumSize(200, 200)
        self.plot_canvas_d.setMinimumSize(200, 200)


        # Make them all resizable
        self.plot_canvas_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_canvas_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_canvas_c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_canvas_d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plots_layout.addWidget(self.plot_canvas_a, 0, 0)
        plots_layout.addWidget(self.plot_canvas_b, 0, 1)
        plots_layout.addWidget(self.plot_canvas_c, 1, 0)
        plots_layout.addWidget(self.plot_canvas_d, 1, 1)
        self._connect_fit_canvas_interactions()

        upper_right_layout.addLayout(plots_layout)
        upper_right_widget.setMinimumSize(200, 200)

        # Lower right widget for positions + message box
        lower_right_widget = QWidget()
        lower_right_layout = QVBoxLayout(lower_right_widget)

        # Positions + Message area
        positions_message_layout = QHBoxLayout()

        positions_layout = QVBoxLayout()
        position_label = QLabel("Fit Check (x,y):")
        self.positions_input = QTextEdit()
        self.positions_input.setMaximumWidth(200)
        self.positions_input.setPlaceholderText(
            "Enter coordinates as x,y pairs (one per line), then use 'Fit Check' to see fits."
        )
        positions_layout.addWidget(position_label)
        positions_layout.addWidget(self.positions_input)
        positions_message_layout.addLayout(positions_layout)

        message_layout = QVBoxLayout()
        message_label = QLabel("Messages:")
        self.message_box = QTextEdit()
        self.message_box.setReadOnly(True)
        message_layout.addWidget(message_label)
        message_layout.addWidget(self.message_box)
        positions_message_layout.addLayout(message_layout)

        # right_layout.addLayout(positions_message_layout)

        lower_right_layout.addLayout(positions_message_layout)
        lower_right_widget.setMinimumSize(200, 150)

        # Add top/bottom widgets to the right splitter
        right_splitter.addWidget(upper_right_widget)
        right_splitter.addWidget(lower_right_widget)

        # Set stretch factors so the plots area is larger than the bottom message area
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        # Add the right splitter to the right_layout
        right_layout.addWidget(right_splitter)  

        # Finally add the right container to the main splitter
        main_splitter.addWidget(right_container)

        # Set initial splitter sizes (optional: good for first display)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        # Add the main splitter to the main layout
        main_layout.addWidget(main_splitter)

        # Connect wavelength input changes to update plots
        self.min_wavelength_input.editingFinished.connect(self.update_plots)
        self.max_wavelength_input.editingFinished.connect(self.update_plots)

    def refresh_phase_dropdown(self, select=None):
        """Rebuild the phase dropdown with current phase data."""
        target = select or self.phase_dropdown.currentText()

        self.phase_dropdown.blockSignals(True)
        self.phase_dropdown.clear()
        self.phase_dropdown.addItem(getattr(self, "phase_placeholder", "Select Phase"))
        for name in sorted(getattr(self, "phase_data", PHASE_DATA).keys()):
            self.phase_dropdown.addItem(name)
        self.phase_dropdown.blockSignals(False)

        idx = self.phase_dropdown.findText(target)
        if idx >= 0:
            self.phase_dropdown.setCurrentIndex(idx)
        else:
            self.phase_dropdown.setCurrentIndex(0)

        if self.phase_dropdown.currentIndex() != 0:
            self.phase_selection_changed(self.phase_dropdown.currentIndex())
            self.update_bragg_edge_table()

    def open_add_phase_dialog(self):
        """Phase management dialog: add, edit, or delete phases."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Phase Management")
        dialog.resize(700, 400)

        main_layout = QHBoxLayout(dialog)

        # Left panel: list + buttons
        left_layout = QVBoxLayout()
        list_label = QLabel("Phases")
        phase_list = QListWidget()
        phase_list.addItems(sorted(getattr(self, "phase_data", PHASE_DATA).keys()))
        left_layout.addWidget(list_label)
        left_layout.addWidget(phase_list)

        add_button = QPushButton("Add New")
        delete_button = QPushButton("Delete")
        add_delete_layout = QHBoxLayout()
        add_delete_layout.addWidget(add_button)
        add_delete_layout.addWidget(delete_button)
        left_layout.addLayout(add_delete_layout)

        # Right panel: edit form
        right_layout = QVBoxLayout()
        form_layout = QFormLayout()

        name_input = QLineEdit()
        form_layout.addRow("Name", name_input)

        structure_input = QComboBox()
        structure_input.addItems(["fcc", "bcc", "tetragonal", "hexagonal", "orthorhombic"])
        form_layout.addRow("Structure", structure_input)

        lattice_a_input = QLineEdit()
        lattice_b_input = QLineEdit()
        lattice_c_input = QLineEdit()
        lattice_a_input.setPlaceholderText("Required for all structures")
        lattice_b_input.setPlaceholderText("Required for orthorhombic")
        lattice_c_input.setPlaceholderText("Required for tetragonal / hexagonal / orthorhombic")
        form_layout.addRow("a (Å)", lattice_a_input)
        form_layout.addRow("b (Å)", lattice_b_input)
        form_layout.addRow("c (Å)", lattice_c_input)

        hkl_input = QTextEdit()
        hkl_input.setPlaceholderText("e.g. 1,1,1\n2,0,0\n2,2,0")
        hkl_input.setFixedHeight(120)
        form_layout.addRow("hkl list", hkl_input)

        right_layout.addLayout(form_layout)
        right_layout.addWidget(QLabel("Enter one h,k,l triplet per line (or separate with commas/semicolons)."))

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        right_layout.addWidget(button_box)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        def populate_form(phase_name):
            phase = getattr(self, "phase_data", PHASE_DATA).get(phase_name, {})
            name_input.setText(phase_name or "")
            structure = phase.get("structure", "fcc")
            if structure_input.findText(structure) < 0:
                structure_input.addItem(structure)
            structure_input.setCurrentText(structure)
            lattice = phase.get("lattice_params", {})
            lattice_a_input.setText(str(lattice.get("a", "")))
            lattice_b_input.setText(str(lattice.get("b", "")))
            lattice_c_input.setText(str(lattice.get("c", "")))
            hkl_list = phase.get("hkl_list", [])
            hkl_lines = "\n".join(f"{h},{k},{l}" for (h, k, l) in hkl_list)
            hkl_input.setPlainText(hkl_lines)

        def reset_form():
            name_input.clear()
            structure_input.setCurrentIndex(0)
            lattice_a_input.clear()
            lattice_b_input.clear()
            lattice_c_input.clear()
            hkl_input.clear()

        def parse_phase_from_form():
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Missing name", "Please provide a phase name.")
                return None, None
            if name in ("Phase", getattr(self, "phase_placeholder", "Select Phase")):
                QMessageBox.warning(self, "Invalid name", "Please choose a different name.")
                return None, None

            structure = structure_input.currentText()

            try:
                lattice_params = {}
                if lattice_a_input.text().strip():
                    lattice_params["a"] = float(lattice_a_input.text())
                if lattice_b_input.text().strip():
                    lattice_params["b"] = float(lattice_b_input.text())
                if lattice_c_input.text().strip():
                    lattice_params["c"] = float(lattice_c_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid lattice parameter", "Lattice parameters must be numeric values.")
                return None, None

            required = {
                "fcc": ["a"],
                "bcc": ["a"],
                "tetragonal": ["a", "c"],
                "hexagonal": ["a", "c"],
                "orthorhombic": ["a", "b", "c"],
                "cubic": ["a"],  # legacy support
            }
            missing = [p for p in required.get(structure, []) if p not in lattice_params]
            if missing:
                QMessageBox.warning(self, "Missing lattice parameter", f"Provide values for: {', '.join(missing)}")
                return None, None

            hkl_text = hkl_input.toPlainText().strip()
            if not hkl_text:
                QMessageBox.warning(self, "Missing hkl list", "Add at least one hkl entry.")
                return None, None

            hkl_entries = []
            for raw in hkl_text.replace(";", "\n").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                raw = raw.strip("()[]")
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                if len(parts) != 3:
                    QMessageBox.warning(self, "Invalid hkl entry", f"'{raw}' is not in h,k,l format.")
                    return None, None
                try:
                    hkl_entries.append(tuple(int(v) for v in parts))
                except ValueError:
                    QMessageBox.warning(self, "Invalid hkl entry", f"'{raw}' must contain integers.")
                    return None, None

            if not hkl_entries:
                QMessageBox.warning(self, "Invalid hkl list", "Add at least one valid hkl entry.")
                return None, None

            phase_definition = {
                "structure": structure,
                "lattice_params": lattice_params,
                "hkl_list": hkl_entries,
            }
            return name, phase_definition

        def save_current_phase():
            parsed = parse_phase_from_form()
            if not parsed:
                return
            name, definition = parsed
            if name is None:
                return
            if name in getattr(self, "phase_data", {}) and name not in getattr(self, "custom_phases", {}):
                QMessageBox.warning(self, "Duplicate phase", "A built-in phase with this name already exists. Use a different name.")
                return
            if hasattr(self, "register_custom_phase"):
                self.register_custom_phase(name, definition)
            else:
                # Fallback: update local phase_data
                self.phase_data[name] = definition
            if not phase_list.findItems(name, Qt.MatchExactly):
                phase_list.addItem(name)
            self.refresh_phase_dropdown(select=name)
            if hasattr(self, "message_box"):
                self.message_box.append(f"Saved phase '{name}'.")
            dialog.accept()

        def delete_selected_phase():
            item = phase_list.currentItem()
            if not item:
                QMessageBox.information(self, "No selection", "Select a phase to delete.")
                return
            name = item.text()
            if hasattr(self, "delete_phase"):
                if not self.delete_phase(name):
                    return
            else:
                # fallback local removal
                if name in getattr(self, "phase_data", {}):
                    self.phase_data.pop(name, None)
            row = phase_list.row(item)
            phase_list.takeItem(row)
            reset_form()
            self.refresh_phase_dropdown()

        def on_selection_changed():
            item = phase_list.currentItem()
            if not item:
                reset_form()
                return
            populate_form(item.text())

        add_button.clicked.connect(reset_form)
        delete_button.clicked.connect(delete_selected_phase)
        phase_list.currentItemChanged.connect(lambda *_: on_selection_changed())
        button_box.accepted.connect(save_current_phase)
        button_box.rejected.connect(dialog.close)

        if phase_list.count():
            phase_list.setCurrentRow(0)
            populate_form(phase_list.currentItem().text())
        else:
            reset_form()

        dialog.exec_()

    def batch_fit_edges(self):
        """
        Initiates the batch fitting process over the ROI using the 'fit_region' approach
        for each row in the bragg_table. Similar to 'batch_fit' but calls 'fit_region'.
        """
        # Ensure ROI and box dimensions are defined
        try:
            box_width = int(self.box_width_input.text())
            box_height = int(self.box_height_input.text())
            step_x = int(self.step_x_input.text())
            step_y = int(self.step_y_input.text())
            min_x = int(self.min_x_input.text())
            max_x = int(self.max_x_input.text())
            min_y = int(self.min_y_input.text())
            max_y = int(self.max_y_input.text())
        except ValueError:
            self.message_box.append("Please enter valid integers for box size, step size, and ROI coordinates.")
            return

        if box_width <= 0 or box_height <= 0 or step_x <= 0 or step_y <= 0:
            self.message_box.append("Box and step sizes must be positive integers.")
            return

        if not self.images:
            self.message_box.append("No images loaded.")
            return

        # If there's no row in the bragg_table, there's nothing to fit.
        total_rows = self.bragg_table.rowCount()
        if total_rows == 0:
            self.message_box.append("No edges in the Bragg table. Nothing to fit.")
            return

        # Check if a valid ROI is set
        fit_area_width = max_x - min_x
        fit_area_height = max_y - min_y
        if fit_area_width < box_width or fit_area_height < box_height:
            self.message_box.append("ROI is too small relative to the box size.")
            return

        # Warn if the batch box differs from the picked macro-pixel size
        if not self._confirm_batch_box_size(box_width, box_height):
            return

        # Compute total boxes for progress bar
        total_boxes = ((fit_area_height - box_height) // step_y + 1) * ((fit_area_width - box_width) // step_x + 1)

        # Reset and show the progress dialog
        self.batch_progress_bar.setMinimum(0)
        self.batch_progress_bar.setMaximum(100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_label.setText("Batch Edge Progress:")
        self.batch_remaining_time_label.setText("Remaining: Calculating...")
        self.batch_progress_dialog.show()

        # Prepare a place to store results (only Region 3 or all) -- Example: store (row, y, x) for (a, s, t).
        # We'll store them in dictionaries or 4D arrays if you like. For brevity, let's do dictionaries here.
        self.batch_edges_results = {}

        # Start the QTimer to estimate remaining time
        self.fit_start_time = time.time()
        if not hasattr(self, 'remaining_time_timer'):
            self.remaining_time_timer = QTimer()
            self.remaining_time_timer.timeout.connect(self.update_remaining_time)
        self.remaining_time_timer.start(5000)

        if not hasattr(self, 'work_directory'):
            self.message_box.append("Working directory is not set. Please load FITS images first.")
            return
        # Get the state of the fix_s and fix_t checkboxes
        fix_s = self.fix_s_checkbox.isChecked()
        fix_t = self.fix_t_checkbox.isChecked()
        fix_eta = self.fix_eta_checkbox.isChecked()

        # Start the worker
        fit_context = self._build_batch_fit_context()
        self.batch_fit_edges_worker = BatchFitEdgesWorker(
            parent=self,
            images=self.images,
            wavelengths=self.wavelengths,
            fit_context=fit_context,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            box_width=box_width,
            box_height=box_height,
            step_x=step_x,
            step_y=step_y,
            total_boxes=total_boxes,
            work_directory=self.work_directory,
            interpolation_enabled = True,
            fix_s=fix_s,
            fix_t=fix_t,
            fix_eta=fix_eta
        )
        self.batch_fit_edges_worker.progress_updated.connect(self.update_progress_bar)
        self.batch_fit_edges_worker.message.connect(self.append_message)
        self.batch_fit_edges_worker.finished.connect(self.batch_fit_edges_finished)
        self.batch_fit_edges_worker.current_box_changed.connect(self.update_current_box)
        self.batch_fit_edges_worker.start()

        # Start the periodic display updates
        self.update_timer.start()

    def batch_fit_edges_finished(self, filename):
        """
        Called when BatchFitEdgesWorker finishes. 'filename' can be an output CSV or empty if there's an error.
        """
        if filename:
            self.message_box.append(f"Batch fitting (edges) completed. Results saved to {filename}")
        else:
            self.message_box.append("Batch fitting (edges) completed (possibly with errors).")

        self.update_timer.stop()
        self.batch_progress_bar.setValue(100)
        self.batch_remaining_time_label.setText("Remaining: ")
        self.batch_progress_dialog.hide()

    def fit_full_pattern_core(
        self,
        fix_s=False,
        fix_t=False,
        fix_eta=False,
        max_nfev=300,
        curve_fit_maxfev=None,
        fit_context=None,
        wavelengths=None,
        intensities=None,
        apply_lattice_update=True,
    ):
        """Core logic for full-pattern fitting supporting multiple crystal structures 
           and returning parameter uncertainties.
           
           max_nfev: iterations for the global least-squares solver (batch uses this).
           curve_fit_maxfev: optional cap for the region-1/2 curve_fit calls.
        """
        STRUCTURE_CONFIG = {
            "cubic": ["a"],
            "fcc": ["a"],
            "bcc": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"]
        }

        # -------------------------------
        # 1) Validate structure & params
        # -------------------------------
        ctx = fit_context or {}
        structure_type = ctx.get("structure_type", getattr(self, "structure_type", "cubic"))
        if structure_type not in STRUCTURE_CONFIG:
            return None, f"Unsupported structure type: {structure_type}"

        required_params = STRUCTURE_CONFIG[structure_type]
        lattice_params = dict(ctx.get("lattice_params", getattr(self, "lattice_params", {})))
        if not lattice_params:
            return None, "Lattice parameters not initialized"

        missing_params = [p for p in required_params if p not in lattice_params]
        if missing_params:
            return None, f"Missing parameters for {structure_type}: {missing_params}"

        wavelengths_data = self.wavelengths if wavelengths is None else np.asarray(wavelengths)
        intensities_data = self.intensities if intensities is None else np.asarray(intensities)

        # -------------------------------
        # 2) Collect valid Bragg edges w/ Region 3 data
        # -------------------------------
        if fit_context is not None:
            source_rows = list(ctx.get("bragg_rows", []))
            total_edges = len(source_rows)
        else:
            source_rows = None
            total_edges = self.bragg_table.rowCount()
        if total_edges == 0:
            return None, "No Bragg edges to fit"

        bragg_edges = []
        for row in range(total_edges):
            if source_rows is not None:
                row_data = source_rows[row]
                if not row_data.get("valid"):
                    continue
                hkl = row_data.get("hkl")
                regions = row_data.get("regions")
                s_val = row_data.get("s", 0.001)
                t_val = row_data.get("t", 0.01)
                eta_val = row_data.get("eta", 0.5)
                if hkl is None or not regions:
                    continue
            else:
                # Parse hkl
                hkl_item = self.bragg_table.item(row, 0)
                try:
                    hkl_str = hkl_item.text().strip('()') if hkl_item else "0,0,0"
                    h, k, l = map(int, hkl_str.split(','))
                    hkl = (h, k, l)
                except ValueError:
                    return None, f"Invalid hkl format in row {row+1}"

                # Validate region bounds
                regions = []
                for region_num in range(1, 4):
                    min_w_item = self.bragg_table.item(row, (region_num - 1)*2 + 2)
                    max_w_item = self.bragg_table.item(row, (region_num - 1)*2 + 3)
                    try:
                        min_w = float(min_w_item.text())
                        max_w = float(max_w_item.text())
                        if min_w >= max_w:
                            return None, f"Invalid region {region_num} bounds for hkl{hkl}"
                        regions.append({'min_wavelength': min_w, 'max_wavelength': max_w})
                    except (ValueError, AttributeError):
                        return None, f"Invalid region {region_num} data for hkl{hkl}"

            # Extract region 3 data
            r3_min, r3_max = regions[2]['min_wavelength'], regions[2]['max_wavelength']
            mask_r3 = (wavelengths_data >= r3_min) & (wavelengths_data <= r3_max)
            x_r3 = wavelengths_data[mask_r3]
            y_r3 = intensities_data[mask_r3]
            if len(x_r3) == 0:
                # Skip edges that have no Region 3 coverage
                continue

            # Fit Region 1 and 2
            try:
                # Region 1
                mask_r1 = (wavelengths_data >= regions[1]['min_wavelength']) & (wavelengths_data <= regions[1]['max_wavelength'])
                x_r1 = wavelengths_data[mask_r1]
                y_r1 = intensities_data[mask_r1]
                p0 = [0,0]
                lower = [-10, -10]
                upper = [10, 10]

                cf_kwargs = {}
                if curve_fit_maxfev is not None:
                    cf_kwargs["maxfev"] = curve_fit_maxfev

                popt_r1, _ = curve_fit(
                    fitting_function_1,
                    x_r1,
                    y_r1,
                    p0=p0,
                    bounds=(lower, upper),
                    **cf_kwargs,
                )
                a0, b0 = popt_r1

                # Region 2
                mask_r2 = (wavelengths_data >= regions[0]['min_wavelength']) & (wavelengths_data <= regions[0]['max_wavelength'])
                x_r2 = wavelengths_data[mask_r2]
                y_r2 = intensities_data[mask_r2]
                popt_r2, _ = curve_fit(
                    lambda xx, a, b: fitting_function_2(xx, a, b, a0, b0),
                    x_r2,
                    y_r2,
                    p0=[0, 0],
                    **cf_kwargs,
                )
                a_hkl, b_hkl = popt_r2

            except (RuntimeError, ValueError, TypeError, FloatingPointError) as e:
                return None, f"Fitting error for hkl{hkl}: {str(e)}"

            if source_rows is None:
                # Grab s/t from the table
                s_item = self.bragg_table.item(row, 8)
                t_item = self.bragg_table.item(row, 9)
                eta_item = self.bragg_table.item(row, 10)
                try:
                    s_val = float(s_item.text()) if s_item else 0.001
                    t_val = float(t_item.text()) if t_item else 0.01
                    eta_val=float(eta_item.text()) if eta_item else 0.5
                except ValueError:
                    return None, f"Invalid s/t values for hkl{hkl}"

            # Store edge data
            bragg_edges.append({
                'hkl': hkl,
                'a0': a0,
                'b0': b0,
                'a_hkl': a_hkl,
                'b_hkl': b_hkl,
                's': s_val,
                't': t_val,
                'eta': eta_val,
                'x_r3': x_r3,
                'y_r3': y_r3,
                'regions': regions
            })

        if not bragg_edges:
            return None, "No valid edges with Region 3 data"

        # -------------------------------
        # 3) Prepare global fit params
        # -------------------------------
        # Lattice parameters
        lattice_initial = [lattice_params[p] for p in required_params]
        lattice_lower = [v * 0.95 for v in lattice_initial]
        lattice_upper = [v * 1.05 for v in lattice_initial]



        # s/t
        s_initial = [e['s'] for e in bragg_edges]
        t_initial = [e['t'] for e in bragg_edges]
        eta_initial = [e['eta'] for e in bragg_edges]

        initial_guess = lattice_initial.copy()
        lower_bounds = lattice_lower.copy()
        upper_bounds = lattice_upper.copy()

        # ---- (NEW) per-edge a0, b0, a_hkl, b_hkl  ----------------------------
        for edge in bragg_edges:
            for key in ("a0", "b0", "a_hkl", "b_hkl"):
                val = edge[key]
                half = 1 * max(abs(val), 1)     # symmetric ±50 %, works if val < 0
                initial_guess.append(val)
                lower_bounds.append(val - half)
                upper_bounds.append(val + half)
        # ----------------------------------------------------------------------




        if not fix_s:
            for s in s_initial:
                initial_guess.append(0.01)
                lower_bounds.append(5e-4)
                upper_bounds.append(0.01)

        # ---- t (0.0001 … 0.1) -------------------------------------------
        if not fix_t:
            for t in t_initial:
                initial_guess.append(0.02)
                lower_bounds.append(1e-2)
                upper_bounds.append(0.1)

        # ---- eta (0 … 1) ------------------------------------------------
        if not fix_eta:
            for eta in eta_initial:
                initial_guess.append(0.5)
                lower_bounds.append(0.0)
                upper_bounds.append(1.0)

        # Concatenate region 3 data from all edges
        concatenated_x = np.concatenate([e['x_r3'] for e in bragg_edges])
        concatenated_y = np.concatenate([e['y_r3'] for e in bragg_edges])
        edge_indices = np.concatenate([
            np.full_like(e['x_r3'], i, dtype=int) for i, e in enumerate(bragg_edges)
        ])

        # -------------------------------
        # 4)  Residuals   (FIXED INDICES)
        # -------------------------------

        n_lat   = len(required_params)
        n_edge  = len(bragg_edges)
        n_ab    = 4 * n_edge 


        def residuals(params):
            idx = 0

            # lattice ----------------------------------------------------------
            lattice_dict = dict(zip(required_params, params[:n_lat]))
            idx += n_lat

            # per-edge (a0, b0, a_hkl, b_hkl) ----------------------------------
            ab_block = params[idx : idx + n_ab].reshape(n_edge, 4)
            idx += n_ab

            # s parameters -----------------------------------------------------
            if not fix_s:
                s_params = params[idx : idx + n_edge];  idx += n_edge
            else:
                s_params = s_initial

            # t parameters -----------------------------------------------------
            if not fix_t:
                t_params = params[idx : idx + n_edge];  idx += n_edge
            else:
                t_params = t_initial

            # eta parameters ---------------------------------------------------
            if not fix_eta:
                eta_params = params[idx : idx + n_edge]
            else:
                eta_params = eta_initial

            # build model ------------------------------------------------------
            model = np.zeros_like(concatenated_y)
            for edge_idx, edge in enumerate(bragg_edges):
                mask   = edge_indices == edge_idx
                x_loc  = concatenated_x[mask]

                a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_block[edge_idx]
                s_val  = s_params[edge_idx]
                t_val  = t_params[edge_idx]
                eta_val= eta_params[edge_idx]
                r3_min = edge['regions'][2]['min_wavelength']
                r3_max = edge['regions'][2]['max_wavelength']

                model[mask] = fitting_function_3(
                    x_loc,
                    a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_val, t_val, eta_val,
                    [edge['hkl']], r3_min, r3_max,
                    structure_type, lattice_dict
                )

            return concatenated_y - model



        # -------------------------------
        # 5) Perform optimization
        # -------------------------------
        try:
            result = least_squares(
                residuals,
                initial_guess,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=max_nfev,
                verbose=0
            )
        except Exception as e:
            return None, f"Optimization failed: {str(e)}"

        if not result.success:
            return None, f"Fit did not converge: {result.message}"



        # -------------------------------
        # 6)  Solve + compute covariance
        # -------------------------------
        final_params = result.x          # already available
        lattice_fit  = dict(zip(required_params,
                                final_params[:len(required_params)]))

        # ▶ 1 — grab the refined (a0,b0,a_hkl,b_hkl) for every edge
        ab_fit_block = final_params[len(required_params) :
                                    len(required_params) + n_ab].reshape(n_edge, 4)

        # ---- variance & covariance ----
        final_res = residuals(final_params)
        N, M      = len(concatenated_y), len(final_params)
        RSS       = np.sum(final_res ** 2)
        variance  = RSS / max(1, N - M)      # avoid zero‑div

        try:
            J   = result.jac
            cov = np.linalg.inv(J.T @ J) * variance
            param_stderr = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            param_stderr = np.full_like(final_params, np.inf)

        # -------------------------------
        # 7)  Slice parameters correctly
        # -------------------------------
        idx = len(required_params)
        ab_slice        = param_stderr[idx : idx + n_ab].reshape(n_edge, 4)
        idx            += n_ab
        # --- s ---
        if not fix_s:
            fitted_s_vals       = final_params[idx : idx + len(bragg_edges)]
            s_uncertainties_list = param_stderr[idx : idx + len(bragg_edges)]
            idx += len(bragg_edges)
        else:
            fitted_s_vals        = s_initial
            s_uncertainties_list = [0.0] * len(bragg_edges)

        # --- t ---
        if not fix_t:
            fitted_t_vals       = final_params[idx : idx + len(bragg_edges)]
            t_uncertainties_list = param_stderr[idx : idx + len(bragg_edges)]
            idx += len(bragg_edges)
        else:
            fitted_t_vals        = t_initial
            t_uncertainties_list = [0.0] * len(bragg_edges)

        # --- eta ---
        if not fix_eta:
            fitted_eta_vals       = final_params[idx : idx + len(bragg_edges)]
            eta_uncertainties_list = param_stderr[idx : idx + len(bragg_edges)]
        else:
            fitted_eta_vals        = eta_initial
            eta_uncertainties_list = [0.0] * len(bragg_edges)



        # Update our class lattice_params with the newly fitted ones
        if apply_lattice_update and hasattr(self, "lattice_params"):
            self.lattice_params.update(lattice_fit)

        # Lattice param uncertainties
        lattice_uncertainties = {}
        for i, p in enumerate(required_params):
            lattice_uncertainties[p] = param_stderr[i]

        # Build dictionaries from hkl -> s/t uncertainty
        s_unc_dict = {}
        t_unc_dict = {}
        eta_unc_dict = {}
        for edge, s_val, s_unc_val, t_val, t_unc_val, eta_val, eta_unc_val in zip(bragg_edges, fitted_s_vals, s_uncertainties_list, fitted_t_vals, t_uncertainties_list, fitted_eta_vals, eta_uncertainties_list):
            s_unc_dict[edge['hkl']] = s_unc_val
            t_unc_dict[edge['hkl']] = t_unc_val
            eta_unc_dict[edge['hkl']] = eta_unc_val
        ab_fits = {
            edge['hkl']: tuple(ab_fit_block[i])      # ab_fit_block comes from step 1
            for i, edge in enumerate(bragg_edges)
        }

        # -------------------------------
        # 8) Build sorted model data
        # -------------------------------
        # Model = data - residual, so:
        model_vals = concatenated_y - final_res
        order = np.argsort(concatenated_x)
        sorted_x = concatenated_x[order]
        sorted_y = model_vals[order]
        sorted_x_exp = concatenated_x[order]
        sorted_y_exp = concatenated_y[order]

        # We'll also return the entire (x_data, y_data) for plotting
        # x_data => sorted_x, y_data => sorted_y
        # residuals => final_res

        # -------------------------------
        # 9) Construct result dictionary
        # -------------------------------
        result_dict = {
            'ab_fits': ab_fits,
            'bragg_edges': bragg_edges,
            'structure_type': structure_type,
            'lattice_params': lattice_fit,
            'lattice_uncertainties': lattice_uncertainties,  # <--- newly added
            'fitted_s': { e['hkl']: s for e, s in zip(bragg_edges, fitted_s_vals) },
            'fitted_t': { e['hkl']: t for e, t in zip(bragg_edges, fitted_t_vals) },
            'fitted_eta': { e['hkl']: eta for e, eta in zip(bragg_edges, fitted_eta_vals) },
            's_uncertainties': s_unc_dict,                  # <--- newly added
            't_uncertainties': t_unc_dict,                  # <--- newly added
            'eta_uncertainties': eta_unc_dict,                  # <--- newly added
            'x_data': sorted_x,                             # <--- for plotting
            'y_data': sorted_y,
            'x_exp_sorted': sorted_x_exp,
            'y_exp_sorted': sorted_y_exp,
            'residuals': final_res,
            'success': result.success,
            'message': result.message
        }

        edge_heights = {}
        edge_widths = {}

        # For each edge in bragg_edges, we now have final s/t + final lattice params.
        # Compute x_edge from lattice params; height from Region3 plateau (max-min); width from Region3 derivative.
        def d_spacing(h, k, l, structure, lat):
            x_vals = calculate_x_hkl_general(structure, lat, [(h, k, l)])
            if not x_vals or np.isnan(x_vals[0]):
                return np.nan
            return x_vals[0] / 2.0  # lambda = 2d => d = lambda/2

        for i, edge in enumerate(bragg_edges):
            (h, k, l) = edge['hkl']
            s_val = fitted_s_vals[i]
            t_val = fitted_t_vals[i]
            eta_val = fitted_eta_vals[i]
            a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_fit_block[i]

            d_hkl = d_spacing(h, k, l, structure_type, lattice_fit)
            if np.isnan(d_hkl) or d_hkl <= 0:
                edge_heights[edge['hkl']] = np.nan
                edge_widths[edge['hkl']] = np.nan
                continue

            x_edge = 2.0 * d_hkl

            # Height: max-min of fitted Region3 curve over its span
            try:
                r3_min = edge["regions"][2]["min_wavelength"]
                r3_max = edge["regions"][2]["max_wavelength"]
                xx_h = np.linspace(r3_min, r3_max, 4000)
                yy_h = fitting_function_3(
                    xx_h, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_val, t_val, eta_val, [edge['hkl']],
                    r3_min, r3_max,
                    structure_type, lattice_fit
                )
                edge_height = yy_h.max() - yy_h.min() if yy_h.size else np.nan
            except (ValueError, FloatingPointError, TypeError):
                edge_height = np.nan

            # Width: FWHM of derivative around x_edge using Region3 model
            try:
                r3_min = edge["regions"][2]["min_wavelength"]
                r3_max = edge["regions"][2]["max_wavelength"]
                span = max(0.2, 0.1 * (r3_max - r3_min))
                start = max(r3_min, x_edge - span)
                end = min(r3_max, x_edge + span)
                if end <= start:
                    raise ValueError("Invalid span for width computation")
                xx = np.linspace(start, end, 4000)
                yy = fitting_function_3(
                    xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_val, t_val, eta_val, [edge['hkl']],
                    start, end,
                    structure_type, lattice_fit
                )
                if yy.size < 5:
                    raise ValueError("Insufficient points for width computation")
                dy = np.gradient(yy, xx)
                dy_max = dy.max()
                if dy_max <= 0:
                    raise ValueError("Non-positive derivative peak")
                half = dy_max / 2.0
                indices = np.where(dy >= half)[0]
                if indices.size == 0:
                    raise ValueError("No points above half maximum")
                edge_width = xx[indices[-1]] - xx[indices[0]]
            except (ValueError, FloatingPointError, TypeError):
                edge_width = np.nan

            edge_widths[edge['hkl']] = edge_width
            edge_heights[edge['hkl']] = edge_height



        # Store in the result dictionary
        result_dict["edge_heights"] = edge_heights  
        result_dict["edge_widths"] = edge_widths


        return result_dict, None

    def fit_full_pattern(self, skip_plot=False):
        """High-level method that calls fit_full_pattern_core and updates GUI elements."""
        # 1) Check if user wants to fix_s/fix_t
        fix_s = self.fix_s_checkbox.isChecked()
        fix_t = self.fix_t_checkbox.isChecked()
        fix_eta = self.fix_eta_checkbox.isChecked()

        # 2) Call core fitting
        result_dict, error_msg = self.fit_full_pattern_core(fix_s=fix_s, fix_t=fix_t, fix_eta=fix_eta)

        # 3) Handle min/max wavelength for theoretical lines, etc.
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
        except ValueError:
            min_wavelength = self.wavelengths[0]
            max_wavelength = self.wavelengths[-1]

        if error_msg:
            self.message_box.append(f"Fit failed: {error_msg}")
            self.fit_full_pattern_button.setEnabled(True)
            self.fit_all_regions_button.setEnabled(True)
            return

        # 4) If we got results, display success
        self.message_box.append("---------- Start pattern fitting ----------")

        # 5) Display lattice parameters + uncertainties
        structure_type = result_dict['structure_type']
        lattice_params = result_dict['lattice_params']
        lattice_unc = result_dict['lattice_uncertainties']

        # For convenience in printing:
        structure_map = {
            "cubic": ["a"],
            "fcc": ["a"],
            "bcc": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"]
        }
        param_names = structure_map.get(structure_type, [])

        for param in param_names:
            val = lattice_params.get(param, np.nan)
            unc = lattice_unc.get(param, np.nan)
            self.message_box.append(
                f"Fitted {param.lower()} = {val:.6f} ± {unc:.6f}"
            )

        # 6) Store fitted s/t in class variables if desired
        self.fitted_s_values = result_dict['fitted_s']
        self.fitted_t_values = result_dict['fitted_t']
        self.fitted_eta_values = result_dict['fitted_eta']
        self.lattice_params.update(lattice_params)  # Update local lattice param store

        # 7) Update table with fitted s,t + show uncertainties
        s_unc = result_dict['s_uncertainties']
        t_unc = result_dict['t_uncertainties']
        eta_unc = result_dict['eta_uncertainties']

        for row, edge in enumerate(result_dict['bragg_edges']):
            hkl = edge['hkl']
            s_val = result_dict['fitted_s'].get(hkl, np.nan)
            t_val = result_dict['fitted_t'].get(hkl, np.nan)
            eta_val = result_dict['fitted_eta'].get(hkl, np.nan)
            s_err = s_unc.get(hkl, np.nan)
            t_err = t_unc.get(hkl, np.nan)
            eta_err = eta_unc.get(hkl, np.nan)

            if row < self.bragg_table.rowCount():
                self.bragg_table.item(row, 8).setText(f"{s_val:.6f}")
                self.bragg_table.item(row, 9).setText(f"{t_val:.6f}")
                self.bragg_table.item(row, 10).setText(f"{eta_val:.3f}")

            # Also display in the message box
            self.message_box.append(
                f"Bragg Edge {row+1} (hkl{hkl}): "
                f"s = {s_val:.6f} ± {s_err:.6f}, "
                f"t = {t_val:.6f} ± {t_err:.6f},"
                f"eta = {eta_val:.6f} ± {eta_err:.3f}"
            )

        # 4) If we got results, display success
        self.message_box.append("---------- Pattern fitting completed ----------")


        # 8) Plot if not skipping
        if not skip_plot:
            ax_main_a, _ = self._initialize_fit_canvas(self.plot_canvas_a)
            # Original data for the "selected region"
            # (Assuming you have self.selected_region_x, etc.)
            ax_main_a.plot(
                self.selected_region_x,
                self.selected_region_y,
                'o',
                markersize=getattr(self, "symbol_size", 4),
                markerfacecolor='blue',
                markeredgecolor='none',
                # label='Experimental Data'
            )
            # Fitted model
            x_fit = result_dict['x_data']
            y_fit = result_dict['y_data']
            self._plot_line(
                ax_main_a,
                x_fit,
                y_fit,
                split_on_gaps=True,
                color='red',
                linestyle='-',
            )

            # Plot theoretical edges in [min_wavelength, max_wavelength]
            if self.show_theoretical_checkbox.isChecked():
                edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength)
                for (hkl, x_hkl) in edges_in_range:
                    ax_main_a.axvline(x=x_hkl, color='red', linestyle='--')
                    y_max = ax_main_a.get_ylim()[1]
                    ax_main_a.text(
                        x_hkl * 1.02,
                        y_max * 0.95,
                        f'hkl{hkl}',
                        rotation=90,
                        verticalalignment='top',
                        color='red',
                        fontsize=getattr(self, "plot_font_size", 12),
                    )

            # Keep top-left x-range bound to the current global wavelength range,
            # independent of prior pan/zoom or wide region-3 model extents.
            if np.isfinite(min_wavelength) and np.isfinite(max_wavelength) and min_wavelength < max_wavelength:
                ax_main_a.set_xlim(min_wavelength, max_wavelength)

            x_obs = result_dict.get("x_exp_sorted", x_fit)
            y_obs = result_dict.get("y_exp_sorted")
            if y_obs is None:
                y_obs = np.interp(x_fit, self.selected_region_x, self.selected_region_y)
            self._plot_residual_line(self.plot_canvas_a, x_obs, y_obs, y_fit, split_on_gaps=True)
            self.plot_canvas_a.draw()

        # 9) Re-enable UI elements
        self.fit_full_pattern_button.setEnabled(True)
        self.fit_all_regions_button.setEnabled(True)

        # 10b) Display edge heights
        if "edge_heights" in result_dict:
            edge_heights = result_dict["edge_heights"]
            self.message_box.append("Edge Heights (Region1 - Region2):")
            for row, edge_info in enumerate(result_dict["bragg_edges"]):
                hkl = edge_info["hkl"]
                height_val = edge_heights.get(hkl, np.nan)
                self.message_box.append(
                    f"  hkl{hkl}: {height_val:.6f}"
                )

        # 10b) Display edge heights
        if "edge_widths" in result_dict:
            edge_widths = result_dict["edge_widths"]
            self.message_box.append("Edge widths:")
            for row, edge_info in enumerate(result_dict["bragg_edges"]):
                hkl = edge_info["hkl"]
                width_val = edge_widths.get(hkl, np.nan)
                self.message_box.append(
                    f"  hkl{hkl}: {width_val:.6f}"
                )

    def apply_selected_area(self):

        try:
            self.min_x = int(self.min_x_input.text())
            self.max_x = int(self.max_x_input.text())
            self.min_y = int(self.min_y_input.text())
            self.max_y = int(self.max_y_input.text())
        except ValueError:
            self.message_box.append("Please enter valid integers for the region.")
            return

        # Ensure the min and max values are within the image bounds
        if self.min_x < 0 or self.max_x > self.images[0].shape[1] or self.min_y < 0 or self.max_y > self.images[0].shape[0]:
            self.message_box.append("Selected area is out of bounds.")
            return

        self.selected_area = (self.min_x, self.max_x, self.min_y, self.max_y)
        self.message_box.append(
            f"Selected area: x[{self.min_x}, {self.max_x}), y[{self.min_y}, {self.max_y})"
        )

        # Now update the image to show the selected area with a rectangle
        self.display_image()

    def handle_fits_run_loaded(self, folder_path, run_dict):
        """
        Handle the images loaded by ImageLoadWorker.
        Perform intensity check, scaling, set up UI elements,
        and attempt to load spectra file. If not found,
        prompt user for manual selection.
        """
        if run_dict:
            # Sort the suffixes to ensure images are in correct order
            sorted_suffixes = sorted(run_dict.keys())
            # Clear existing images if any
            self.images = []
            for suffix in sorted_suffixes:
                self.images.append(run_dict[suffix])
            run_number = len(run_dict)
            self.message_box.append(f"Successfully loaded {run_number} FITS images from the selected folder.")

            # Calculate the total intensity of the first image
            total_intensity = np.sum(self.images[0])
            if total_intensity < 10:
                # Scale all images by 1,000,000
                self.images = [image.astype(np.float64) * 1e6 for image in self.images]
                self.message_box.append(
                    "Total intensity of the first image is less than 1. "
                    "All images have been scaled by 1,000,000."
                )
            else:
                self.message_box.append("Loaded images without scaling.")

            # Enable the image slider and set its range based on the number of images
            self.image_slider.setEnabled(True)
            self.image_slider.setRange(0, len(self.images) - 1)
            self.image_slider.setValue(0)  # Display the first image initially

            # Attempt to find a spectra file ending with "_Spectra"
            spectra_files = glob.glob(os.path.join(self.folder_path, "*_Spectra*"))

            if not spectra_files:
                # No spectra file detected automatically, prompt user
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No Spectra File Found")
                msg.setText("No spectra file with suffix '_Spectra' found. Would you like to select one manually?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                choice = msg.exec_()

                if choice == QMessageBox.Yes:
                    # Let user pick the spectra file
                    spectra_file, _ = QFileDialog.getOpenFileName(
                        self, 
                        "Select Spectra File",
                        self.folder_path,
                        "Text Files (*.txt);;All Files (*)"
                    )
                    if spectra_file:
                        spectra_files = [spectra_file]
                    else:
                        self.message_box.append("No spectra file selected. Proceeding without wavelength data.")
                else:
                    self.message_box.append("No spectra file selected. Proceeding without wavelength data.")

            # If we have a spectra file either from detection or selection
            if spectra_files:
                spectra_file = spectra_files[0]
                try:
                    self.tof_array = np.loadtxt(spectra_file, usecols=0)
                    self.set_manual_wavelength_mode(False)
                    self.update_wavelengths()
                except Exception as e:
                    self.message_box.append(f"Error reading spectra file: {e}")
                    self.set_manual_wavelength_mode(True)
            else:
                self.set_manual_wavelength_mode(True)

            # Display the first image with auto-adjusted contrast and brightness
            self.display_image()
        else:
            self.message_box.append("No valid FITS images were loaded from the selected folder.")

    def change_flight_path(self):
        """
        Opens a dialog to allow the user to change the flight path value.
        Then updates the wavelength calculation if tof_array is available.
        """
        new_flight_path, ok = QInputDialog.getDouble(
            self,
            "Change Flight Path",
            "Enter new flight path value (default: 56.4):",
            self.flight_path,
            0.0,   # Minimum value
            1000.0, # Maximum value
            3       # Decimal places
        )
        if ok:
            self.flight_path = new_flight_path
            self.message_box.append(f"Flight path updated to {self.flight_path}")

            # If we already have a tof_array loaded, recalculate wavelengths
            if self.tof_array is not None and len(self.tof_array) > 0:
                self.update_wavelengths()
                self.message_box.append("Click 'Pick' to update the plot")
            else:
                self.message_box.append("No ToF data available to update wavelengths.")
        else:
            self.message_box.append("Flight path change cancelled.")

    def update_wavelengths(self):
        """
        Recalculate the wavelength array based on the current tof_array and flight_path.
        Update start/end wavelength accordingly.
        """
        if getattr(self, "manual_wavelength_mode", False):
            self.update_manual_wavelengths()
            return

        if self.tof_array is not None and len(self.tof_array) > 0:
            adjusted_tof = self.tof_array + getattr(self, 'delay', 0.0)
            self.wavelengths = (adjusted_tof * 3.956) / self.flight_path * 1000

            if len(self.wavelengths) != len(self.images):
                self.message_box.append(
                    f"Warning: Number of wavelengths ({len(self.wavelengths)}) "
                    f"does not match number of images ({len(self.images)})."
                )

            self.start_wavelength = self.wavelengths[0]
            self.end_wavelength = self.wavelengths[-1]
            self.message_box.append(
                f"Updated start/end wavelengths: {self.start_wavelength:.6f} / {self.end_wavelength:.6f}"
            )
        else:
            self.wavelengths = np.array([])
            self.message_box.append("No valid ToF data to compute wavelengths.")

    def set_manual_wavelength_mode(self, enabled: bool):
        """Enable or disable manual wavelength interpolation mode."""
        if getattr(self, "manual_wavelength_mode", False) == enabled:
            if enabled:
                self.update_manual_wavelengths()
            return
        self.manual_wavelength_mode = enabled
        if enabled:
            self.tof_array = None
            self.message_box.append(
                "Manual mode enabled. Set anchors (wavelength or ToF) in the Config dialog."
            )
            self.update_manual_wavelengths()
        else:
            self.message_box.append("Spectra file detected. Manual wavelength controls disabled.")

    def _format_image_suffix(self, image_index: int) -> str:
        """Convert a 1-based image index into the zero-padded file suffix."""
        if image_index <= 0:
            return "--"
        return f"_{image_index - 1:05d}"

    def update_manual_wavelengths(self):
        """Compute wavelengths via interpolation when no spectra file is available."""
        if not getattr(self, "images", []):
            self.message_box.append("Load images before setting manual wavelength bounds.")
            self.wavelengths = np.array([])
            return

        count = len(self.images)

        raw_anchors = getattr(self, "manual_wavelength_anchors", [])
        anchor_mode = getattr(self, "manual_anchor_mode", "wavelength")
        flight_path = getattr(self, "flight_path", 0.0) or 0.0
        delay = getattr(self, "delay", 0.0)
        if anchor_mode == "tof" and flight_path == 0:
            self.message_box.append("Flight path must be > 0 to convert time-of-flight anchors.")
            self.wavelengths = np.array([])
            return
        anchor_map = {}
        for entry in raw_anchors:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index", 0))
                raw_val = float(entry.get("value", entry.get("wavelength")))
            except (TypeError, ValueError):
                continue
            if idx <= 0:
                continue
            if idx > count:
                self.message_box.append(f"Anchor at image {idx} exceeds available images ({count}); clamped to last image.")
                idx = count
            if anchor_mode == "tof":
                wl = ((raw_val + delay) * 3.956) / flight_path * 1000
            else:
                wl = raw_val
            anchor_map[idx - 1] = wl

        user_anchor_count = len(anchor_map)
        anchors = sorted(anchor_map.items())
        if not anchors:
            self.message_box.append("Provide at least two anchors in Config to compute manual wavelengths.")
            self.wavelengths = np.array([])
            return

        if len(anchors) == 1:
            wl_value = anchors[0][1]
            self.wavelengths = np.full(count, wl_value)
            self.start_wavelength = self.end_wavelength = wl_value
            self.message_box.append(
                f"Only one anchor provided; assigned constant wavelength {wl_value:.6f} Å to all images."
            )
            return

        # Ensure coverage from first to last image by extending anchors to edges
        if anchors[0][0] > 0:
            anchors.insert(0, (0, anchors[0][1]))
        if anchors[-1][0] < count - 1:
            anchors.append((count - 1, anchors[-1][1]))

        self.wavelengths = np.empty(count)
        for i in range(len(anchors) - 1):
            start_idx, start_wl = anchors[i]
            end_idx, end_wl = anchors[i + 1]
            if end_idx <= start_idx:
                self.wavelengths[start_idx] = start_wl
                continue
            segment = np.linspace(start_wl, end_wl, end_idx - start_idx + 1)
            self.wavelengths[start_idx:end_idx + 1] = segment

        self.start_wavelength = float(self.wavelengths[0])
        self.end_wavelength = float(self.wavelengths[-1])
        self.message_box.append(
            f"Manual wavelengths set from {self.start_wavelength:.6f} to {self.end_wavelength:.6f} Å "
            f"using {user_anchor_count} user anchors ({len(anchors)} points including edges)."
        )

    def set_delay(self):
        """Open dialog to set time delay correction"""
        new_delay, ok = QInputDialog.getDouble(
            self,
            "Set Time Delay",
            "Enter time delay correction:",
            getattr(self, 'delay', 0.0),  
            -1.0,  
            1.0,   
            3         
        )

        if ok:
            self.delay = new_delay
            self.message_box.append(f"Time delay set to {self.delay} ")


            if self.tof_array is not None:
                self.update_wavelengths()
                self.update_plots()

    def _ensure_load_progress_dialog(self):
        """Create the modal progress dialog for FITS loading if needed."""
        if getattr(self, "_fits_progress_dialog", None) is None:
            self._fits_progress_dialog = QProgressDialog(
                "Loading FITS images...", "Cancel", 0, 100, self
            )
            self._fits_progress_dialog.setWindowTitle("Loading Images")
            self._fits_progress_dialog.setAutoClose(False)
            self._fits_progress_dialog.setAutoReset(False)
            self._fits_progress_dialog.canceled.connect(self._cancel_fits_loading)
        return self._fits_progress_dialog

    def _cancel_fits_loading(self):
        """Stop the worker if the user cancels loading."""
        worker = getattr(self, "fits_image_load_worker", None)
        if worker is not None:
            worker.requestInterruption()
            worker.stop()
            worker.wait(3000)
            if worker.isRunning():
                self.message_box.append("Stopping image loader...")
                return
            self.fits_image_load_worker = None
            self.images = []
            self.image_slider.setEnabled(False)
            self.load_button.setEnabled(True)
            self.display_image()
            self.message_box.append("Image loading cancelled.")
        if getattr(self, "_fits_progress_dialog", None) is not None:
            self._fits_progress_dialog.hide()

    def update_fits_load_progress(self, value):
        """
        Update the FITS Viewer image loading progress dialog.
        """
        dialog = self._ensure_load_progress_dialog()
        dialog.setValue(value)
        if value >= 100:
            dialog.hide()
            dialog.close()

    def fits_image_loading_finished(self):
        """
        Handle the completion of the image loading process.
        Reset the progress bar and re-enable the load button.
        """
        worker = getattr(self, "fits_image_load_worker", None)
        if worker is not None:
            if worker.isRunning():
                worker.wait(3000)
            if worker.isRunning():
                self.message_box.append("Image loader is still running; cleanup deferred.")
                return
            self.fits_image_load_worker = None
        if getattr(self, "_fits_progress_dialog", None) is not None:
            self._fits_progress_dialog.hide()
            self._fits_progress_dialog.close()
        self.load_button.setEnabled(True)        # Re-enable the load button

    def open_batch_settings_dialog(self):
        """Dialog to edit batch fitting parameters and ROI."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Fit Settings")
        dialog_layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        width_edit = QLineEdit(self.box_width_input.text())
        height_edit = QLineEdit(self.box_height_input.text())
        step_x_edit = QLineEdit(self.step_x_input.text())
        step_y_edit = QLineEdit(self.step_y_input.text())

        form_layout.addRow("Box Width:", width_edit)
        form_layout.addRow("Box Height:", height_edit)
        form_layout.addRow("Step X:", step_x_edit)
        form_layout.addRow("Step Y:", step_y_edit)
        dialog_layout.addLayout(form_layout)

        interpolation_checkbox = QCheckBox("Enable interpolation")
        interpolation_checkbox.setChecked(self.interpolation_checkbox.isChecked())
        dialog_layout.addWidget(interpolation_checkbox)

        roi_layout = QFormLayout()
        min_x_edit = QLineEdit(self.min_x_input.text())
        max_x_edit = QLineEdit(self.max_x_input.text())
        min_y_edit = QLineEdit(self.min_y_input.text())
        max_y_edit = QLineEdit(self.max_y_input.text())
        batch_roi_tooltip = self._roi_bounds_tooltip(
            "Batch fitting ROI",
            "x[10, 12), y[20, 23) includes x pixels 10-11 and y pixels 20-22.",
        )
        for roi_edit in (min_x_edit, max_x_edit, min_y_edit, max_y_edit):
            roi_edit.setToolTip(batch_roi_tooltip)
        roi_layout.addRow("ROI X Min:", min_x_edit)
        roi_layout.addRow("ROI X Max:", max_x_edit)
        roi_layout.addRow("ROI Y Min:", min_y_edit)
        roi_layout.addRow("ROI Y Max:", max_y_edit)
        dialog_layout.addLayout(roi_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            try:
                box_width = int(width_edit.text())
                box_height = int(height_edit.text())
                step_x = int(step_x_edit.text())
                step_y = int(step_y_edit.text())
                min_x = int(min_x_edit.text())
                max_x = int(max_x_edit.text())
                min_y = int(min_y_edit.text())
                max_y = int(max_y_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "All values must be integers.")
                return

            self.box_width_input.setText(str(box_width))
            self.box_height_input.setText(str(box_height))
            self.step_x_input.setText(str(step_x))
            self.step_y_input.setText(str(step_y))
            self.min_x_input.setText(str(min_x))
            self.max_x_input.setText(str(max_x))
            self.min_y_input.setText(str(min_y))
            self.max_y_input.setText(str(max_y))
            self.interpolation_checkbox.setChecked(interpolation_checkbox.isChecked())

            # Do NOT overwrite the current picked ROI (selected_area); batch settings
            # should not change the spectra used for single-pixel fits.

    def open_general_settings_dialog(self):
        """Dialog for auto/manual adjustments and flight/delay settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configuration")
        layout = QVBoxLayout(dialog)

        adjust_layout = QHBoxLayout()
        auto_btn = QPushButton("Auto Adjust")
        auto_btn.clicked.connect(self.auto_adjust)
        adjust_layout.addWidget(auto_btn)

        manual_btn = QPushButton("Manual Adjust")
        manual_btn.clicked.connect(self.open_adjustments_dialog)
        adjust_layout.addWidget(manual_btn)
        layout.addLayout(adjust_layout)

        form_layout = QFormLayout()
        flight_spin = QDoubleSpinBox()
        flight_spin.setRange(0.0, 1000.0)
        flight_spin.setDecimals(3)
        flight_spin.setValue(self.flight_path)
        form_layout.addRow("Flight Path (m):", flight_spin)

        delay_spin = QDoubleSpinBox()
        delay_spin.setRange(-1.0, 1.0)
        delay_spin.setDecimals(4)
        delay_spin.setSingleStep(0.001)
        delay_spin.setValue(getattr(self, "delay", 0.0))
        form_layout.addRow("Time Delay (ms):", delay_spin)

        symbol_spin = QSpinBox()
        symbol_spin.setRange(1, 20)
        symbol_spin.setValue(getattr(self, "symbol_size", 4))
        form_layout.addRow("Symbol Size:", symbol_spin)
        layout.addLayout(form_layout)

        anchor_mode_combo = QComboBox()
        anchor_mode_combo.addItems(["Wavelength (Å)", "Time of Flight (ms)"])
        anchor_mode_combo.setToolTip("Choose whether anchor values are wavelengths or time-of-flight.")
        anchor_mode_combo.setCurrentIndex(1 if getattr(self, "manual_anchor_mode", "wavelength") == "tof" else 0)
        layout.addWidget(anchor_mode_combo)

        anchor_group = QGroupBox("Manual anchors (leave index as 'Unused' to skip)")
        anchor_layout = QGridLayout(anchor_group)
        anchor_layout.addWidget(QLabel("Image #"), 0, 0)
        anchor_layout.addWidget(QLabel("Suffix"), 0, 1)
        value_header = QLabel("Wavelength (Å)")
        anchor_layout.addWidget(value_header, 0, 2)

        max_index = max(len(getattr(self, "images", [])), 5000)
        existing_anchors = getattr(self, "manual_wavelength_anchors", [])
        anchor_rows = []

        def _update_anchor_header(idx: int):
            if idx == 1:
                value_header.setText("ToF (ms)")
                for _, val_spin, _ in anchor_rows:
                    val_spin.setSuffix("")
            else:
                value_header.setText("Wavelength (Å)")
                for _, val_spin, _ in anchor_rows:
                    val_spin.setSuffix("")

        def add_anchor_row(preset_index=0, preset_val=1.0):
            row = len(anchor_rows) + 1  # +1 for header
            idx_spin = QSpinBox()
            idx_spin.setRange(0, max_index)
            idx_spin.setSpecialValueText("Unused")
            idx_spin.setValue(preset_index)
            idx_spin.setEnabled(getattr(self, "manual_wavelength_mode", False))

            suffix_label = QLabel(self._format_image_suffix(idx_spin.value()))
            idx_spin.valueChanged.connect(lambda val, lbl=suffix_label: lbl.setText(self._format_image_suffix(val)))

            val_spin = QDoubleSpinBox()
            val_spin.setRange(-10.0, 100.0)
            val_spin.setDecimals(6)
            val_spin.setValue(preset_val)
            val_spin.setEnabled(getattr(self, "manual_wavelength_mode", False))

            anchor_layout.addWidget(idx_spin, row, 0)
            anchor_layout.addWidget(suffix_label, row, 1)
            anchor_layout.addWidget(val_spin, row, 2)
            anchor_rows.append((idx_spin, val_spin, suffix_label))
            _update_anchor_header(anchor_mode_combo.currentIndex())

        base_rows = max(10, len(existing_anchors))
        for i in range(base_rows):
            preset_index = 0
            preset_val = 1.0
            if i < len(existing_anchors):
                try:
                    preset_index = int(existing_anchors[i].get("index", 0))
                    preset_val = float(
                        existing_anchors[i].get(
                            "value",
                            existing_anchors[i].get("wavelength", preset_val)
                        )
                    )
                except (TypeError, ValueError, AttributeError):
                    pass
            add_anchor_row(preset_index, preset_val)

        _update_anchor_header(anchor_mode_combo.currentIndex())
        anchor_mode_combo.currentIndexChanged.connect(_update_anchor_header)

        layout.addWidget(anchor_group)

        add_row_btn = QPushButton("Add more lines")
        add_row_btn.setToolTip("Append another anchor row for manual wavelength/ToF input.")
        add_row_btn.clicked.connect(lambda: add_anchor_row())
        layout.addWidget(add_row_btn)

        load_cfg_btn = QPushButton("Load configuration")
        load_cfg_btn.setToolTip("Load metadata from a result CSV to reuse settings.")
        layout.addWidget(load_cfg_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        def _load_config_from_csv():
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select result CSV with metadata",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not file_name:
                return
            metadata = {}
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    first = f.readline().strip()
                    if not first.startswith("Metadata Name"):
                        QMessageBox.warning(dialog, "Invalid file", "Selected CSV does not contain metadata header.")
                        return
                    for line in f:
                        line = line.strip()
                        if not line:
                            break
                        if "," in line:
                            k, v = line.split(",", 1)
                            metadata[k] = v
            except Exception as exc:
                QMessageBox.warning(dialog, "Load failed", f"Could not read metadata: {exc}")
                return

            def _set_float(spin, key):
                try:
                    spin.setValue(float(metadata[key]))
                except (KeyError, TypeError, ValueError):
                    pass
            def _set_line_float(line_edit, key):
                try:
                    line_edit.setText(str(float(metadata[key])))
                except (KeyError, TypeError, ValueError):
                    pass
            def _set_int(line_edit, key):
                try:
                    line_edit.setText(str(int(float(metadata[key]))))
                except (KeyError, TypeError, ValueError):
                    pass
            def _set_bool(checkbox, key):
                try:
                    checkbox.setChecked(str(metadata[key]).strip().lower() in ("1","true","yes","y","on"))
                except (KeyError, TypeError, ValueError):
                    pass

            _set_float(flight_spin, "flight_path")
            for key in ("time_delay_ms", "delay"):
                if key in metadata:
                    _set_float(delay_spin, key)
                    break
            _set_line_float(self.min_wavelength_input, "min_wavelength")
            _set_line_float(self.max_wavelength_input, "max_wavelength")
            try:
                if "symbol_size" in metadata:
                    symbol_spin.setValue(int(float(metadata["symbol_size"])))
            except (TypeError, ValueError):
                pass

            # Batch ROI and grid parameters
            _set_int(self.box_width_input, "box_width")
            _set_int(self.box_height_input, "box_height")
            _set_int(self.step_x_input, "step_x")
            _set_int(self.step_y_input, "step_y")
            _set_int(self.min_x_input, "roi_x_min")
            _set_int(self.max_x_input, "roi_x_max")
            _set_int(self.min_y_input, "roi_y_min")
            _set_int(self.max_y_input, "roi_y_max")

            # Fix flags
            _set_bool(self.fix_s_checkbox, "fix_s")
            _set_bool(self.fix_t_checkbox, "fix_t")
            _set_bool(self.fix_eta_checkbox, "fix_eta")

            # Selected phase
            phase_name = metadata.get("selected_phase")
            if phase_name:
                idx = self.phase_dropdown.findText(phase_name)
                if idx == -1:
                    self.phase_dropdown.addItem(phase_name)
                    idx = self.phase_dropdown.findText(phase_name)
                if idx >= 0:
                    self.phase_dropdown.setCurrentIndex(idx)

            # Restore Bragg table rows if present
            bragg_rows = [(k, v) for k, v in metadata.items() if k.lower().startswith("bragg_table_row_")]
            if bragg_rows:
                bragg_rows.sort(key=lambda kv: int(''.join(filter(str.isdigit, kv[0])) or 0))
                self.bragg_table.setRowCount(len(bragg_rows))
                for row_idx, (_, row_text) in enumerate(bragg_rows):
                    parts = row_text.split("|")
                    if parts:
                        parts[0] = parts[0].replace(";", ",")
                    for col_idx in range(self.bragg_table.columnCount()):
                        val = parts[col_idx] if col_idx < len(parts) else ""
                        item = QTableWidgetItem(val)
                        self.bragg_table.setItem(row_idx, col_idx, item)
                self._bragg_table_ready = True

            QMessageBox.information(dialog, "Configuration loaded", "Metadata applied where available.")

        load_cfg_btn.clicked.connect(_load_config_from_csv)

        if dialog.exec_() == QDialog.Accepted:
            self.flight_path = flight_spin.value()
            self.delay = delay_spin.value()
            self.message_box.append(f"Flight path set to {self.flight_path:.3f} m.")
            self.message_box.append(f"Time delay set to {self.delay:.4f} ms.")

            old_symbol_size = getattr(self, "symbol_size", 4)
            self.symbol_size = symbol_spin.value()
            if self.tof_array is not None:
                self.update_wavelengths()
                self.update_plots()

            if (
                self.symbol_size != old_symbol_size
                and getattr(self, "intensities", None) is not None
                and len(self.intensities) > 0
            ):
                self.update_plots()
            self.apply_symbol_size()

            anchors = []
            self.manual_anchor_mode = "tof" if anchor_mode_combo.currentIndex() == 1 else "wavelength"
            anchors = []
            for idx_spin, val_spin, _ in anchor_rows:
                idx_val = idx_spin.value()
                if idx_val <= 0:
                    continue
                anchors.append({"index": idx_val, "value": val_spin.value()})
            self.manual_wavelength_anchors = anchors
            if getattr(self, "manual_wavelength_mode", False):
                self.update_manual_wavelengths()
                if getattr(self, "selected_area", None):
                    self.update_plots()

    def open_adjustments_dialog(self):
        # (Re-open the existing dialog if it is already up)
        if getattr(self, "adjustments_dialog", None) and self.adjustments_dialog.isVisible():
            self.adjustments_dialog.raise_()
            self.adjustments_dialog.activateWindow()
            return

        # Otherwise create it and remember a reference
        self.adjustments_dialog = AdjustmentsDialog(self)
        # Expose the two spin-boxes so other methods can use them
        self.min_spinbox = self.adjustments_dialog.min_spinbox
        self.max_spinbox = self.adjustments_dialog.max_spinbox
        self.adjustments_dialog.show()

    def load_fits_images(self):
        """
        Load FITS images by selecting a folder. Only images with suffixes from _00000 to _02924 are loaded.
        After loading, perform intensity check and scaling if necessary.
        """
        # Open a folder dialog to select a directory containing FITS images
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing FITS Images", ""
        )
        if folder_path:
            existing_worker = getattr(self, "fits_image_load_worker", None)
            if existing_worker is not None and existing_worker.isRunning():
                self.message_box.append("Image loading is already in progress.")
                return
            self.fits_image_load_worker = None

            self.folder_path = folder_path
            # Set the working directory to the parent directory of the selected folder
            self.work_directory = os.path.dirname(folder_path)

            # Disable the load button to prevent multiple concurrent loads
            self.load_button.setEnabled(False)

            # Start image loading in a separate thread using ImageLoadWorker
            self.fits_image_load_worker = ImageLoadWorker(folder_path)
            self.fits_image_load_worker.progress_updated.connect(self.update_fits_load_progress)
            self.fits_image_load_worker.message.connect(self.message_box.append)
            self.fits_image_load_worker.run_loaded.connect(self.handle_fits_run_loaded)
            self.fits_image_load_worker.finished.connect(self.fits_image_loading_finished)
            self.fits_image_load_worker.start()

    def auto_adjust(self):
        """
        Robust auto-window: 5th–95th percentiles, brightness=0, contrast=1.0.
        """
        if not self.images:
            QMessageBox.warning(self, "Auto Adjust", "Load an image first.")
            return

        idx = self.image_slider.value()
        img = self.images[idx]

        # --- recompute limits -------------------------------------------------
        self.auto_vmin, self.auto_vmax = np.percentile(img, [5, 95])
        self.current_vmin = self.auto_vmin
        self.current_vmax = self.auto_vmax
        self.min_slider_value = self.auto_vmin
        self.max_slider_value = self.auto_vmax
        self.contrast_slider_value   = 100   # == 1.0
        self.brightness_slider_value = 0

        # --- update the Adjust Image dialog if it is open ---------------------
        dlg = getattr(self, "adjustments_dialog", None)
        if dlg and dlg.isVisible():
            # Block signals while pushing new values
            for w in (dlg.contrast_slider, dlg.contrast_spinbox,
                      dlg.brightness_slider, dlg.brightness_spinbox,
                      dlg.min_spinbox, dlg.max_spinbox):
                w.blockSignals(True)

            dlg.contrast_slider.setValue(100)
            dlg.contrast_spinbox.setValue(1.0)
            dlg.brightness_slider.setValue(0)
            dlg.brightness_spinbox.setValue(0)

            rng = self.auto_vmax - self.auto_vmin
            dlg.min_spinbox.setRange(self.auto_vmin - 0.5*rng, self.auto_vmax)
            dlg.max_spinbox.setRange(self.auto_vmin, self.auto_vmax + 0.5*rng)
            dlg.min_spinbox.setValue(self.auto_vmin)
            dlg.max_spinbox.setValue(self.auto_vmax)

            for w in (dlg.contrast_slider, dlg.contrast_spinbox,
                      dlg.brightness_slider, dlg.brightness_spinbox,
                      dlg.min_spinbox, dlg.max_spinbox):
                w.blockSignals(False)

        # --- keep viewer-level spin-boxes (if they exist) in sync --------------
        if hasattr(self, "min_spinbox") and hasattr(self, "max_spinbox"):
            self.min_spinbox.blockSignals(True)
            self.max_spinbox.blockSignals(True)
            self.min_spinbox.setValue(self.current_vmin)
            self.max_spinbox.setValue(self.current_vmax)
            self.min_spinbox.blockSignals(False)
            self.max_spinbox.blockSignals(False)

        # finally redraw
        self.display_image()

    def display_image(self):
        if not self.images:
            return

        # Get the selected image based on the slider position
        self.current_image_index = self.image_slider.value()
        current_image = self.images[self.current_image_index]

        # Retrieve contrast and brightness adjustments from variables
        contrast_scale = self.contrast_slider_value / 100.0
        brightness_offset = self.brightness_slider_value

        if self.auto_vmin is None or self.auto_vmax is None:
            # Default display range if auto-adjust not set
            self.auto_vmin, self.auto_vmax = np.percentile(current_image, [5, 95])
            self.current_vmin, self.current_vmax = self.auto_vmin, self.auto_vmax

        # Calculate new display range for ImageJ-style adjustment
        mid = (self.auto_vmin + self.auto_vmax) / 2.0
        range_half = (self.auto_vmax - self.auto_vmin) * contrast_scale / 2.0
        new_vmin = mid - range_half + brightness_offset
        new_vmax = mid + range_half + brightness_offset

        # Ensure vmin and vmax are within the bounds set by min/max values
        self.current_vmin = max(new_vmin, self.min_slider_value)
        self.current_vmax = min(new_vmax, self.max_slider_value)

        # Additional validation to ensure vmin is less than vmax
        # if self.current_vmin >= self.current_vmax:
        #     # Reset to auto-adjusted values if clamping leads to invalid range
        #     self.current_vmin, self.current_vmax = self.auto_vmin, self.auto_vmax
        #     self.min_spinbox.setValue(self.current_vmin)
        #     self.max_spinbox.setValue(self.current_vmax)
        #     QMessageBox.warning(self, "Adjustment Error", "Minimum value exceeded Maximum value. Resetting to auto-adjusted range.")

        if self.current_vmin >= self.current_vmax:
            self.current_vmin, self.current_vmax = self.auto_vmin, self.auto_vmax
            if hasattr(self, "min_spinbox"):
                self.min_spinbox.setValue(self.current_vmin)
            if hasattr(self, "max_spinbox"):
                self.max_spinbox.setValue(self.current_vmax)
            QMessageBox.warning(
                self, "Adjustment Error",
                "Minimum value exceeded Maximum value. Resetting to auto-adjusted range."
            )


        # Preserve current zoom/pan view across redraws.
        saved_xlim = None
        saved_ylim = None
        if self.canvas.axes.has_data():
            saved_xlim = self.canvas.axes.get_xlim()
            saved_ylim = self.canvas.axes.get_ylim()

        # Display the adjusted image on the Matplotlib canvas
        self.canvas.axes.clear()
        self.canvas.axes.imshow(current_image, cmap='gray', vmin=self.current_vmin, vmax=self.current_vmax)

        # Draw the selected area, if available (initial fitting ROI)
        if self.selected_area:
            xmin, xmax, ymin, ymax = self.selected_area
            xy, width, height = self._slice_box_patch_args(xmin, xmax, ymin, ymax)
            rect = Rectangle(xy, width, height, edgecolor='yellow', facecolor='none', lw=1)
            self.canvas.axes.add_patch(rect)

        # Draw the static batch ROI from min/max inputs (orange)
        try:
            roi_min_x = int(self.min_x_input.text())
            roi_max_x = int(self.max_x_input.text())
            roi_min_y = int(self.min_y_input.text())
            roi_max_y = int(self.max_y_input.text())
            if roi_min_x < roi_max_x and roi_min_y < roi_max_y:
                xy, width, height = self._slice_box_patch_args(
                    roi_min_x,
                    roi_max_x,
                    roi_min_y,
                    roi_max_y,
                )
                roi_rect = Rectangle(
                    xy,
                    width,
                    height,
                    edgecolor="orange",
                    facecolor="none",
                    lw=1,
                )
                self.canvas.axes.add_patch(roi_rect)
        except ValueError:
            pass

        # # Draw the batch fitting moving box, if available
        # if self.current_batch_box:
        #     xmin, xmax, ymin, ymax = self.current_batch_box
        #     if self.batch_box_patch:
        #         self.batch_box_patch.remove()
        #     self.batch_box_patch = Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, edgecolor='red', facecolor='none', lw=1)
        #     self.canvas.axes.add_patch(self.batch_box_patch)

        # self.canvas.axes.set_title(f"Image {self.current_image_index + 1}/{len(self.images)}")
        # self.canvas.draw()

        # Draw the batch fitting moving box, if available
        if self.current_batch_box:
            row_min, row_max, col_min, col_max = self.current_batch_box
            xy, width, height = self._slice_box_patch_args(col_min, col_max, row_min, row_max)

            # If a rectangle already exists, update it
            if self.batch_box_patch is not None and self.batch_box_patch.axes is not None:
                try:
                    # Update position and size instead of removing/re-adding
                    self.batch_box_patch.set_xy(xy)
                    self.batch_box_patch.set_width(width)
                    self.batch_box_patch.set_height(height)
                except (ValueError, RuntimeError, AttributeError):
                    # Fallback: recreate it cleanly
                    self.batch_box_patch.remove()
                    self.batch_box_patch = Rectangle(
                        xy,
                        width,
                        height,
                        edgecolor="red",
                        facecolor="none",
                        lw=1,
                    )
                    self.canvas.axes.add_patch(self.batch_box_patch)
            else:
                # No existing patch – create a new one
                self.batch_box_patch = Rectangle(
                    xy,
                    width,
                    height,
                    edgecolor="red",
                    facecolor="none",
                    lw=1,
                )
                self.canvas.axes.add_patch(self.batch_box_patch)

        self.canvas.axes.set_title(
            f"Image {self.current_image_index + 1}/{len(self.images)}",
            fontsize=getattr(self, "plot_font_size", 12),
        )
        if saved_xlim is not None and saved_ylim is not None:
            self.canvas.axes.set_xlim(saved_xlim)
            self.canvas.axes.set_ylim(saved_ylim)
        self.canvas.draw_idle()

    def update_display(self):
        # Update the displayed image to show the moving batch box
        self.display_image()

    def update_current_box(self, xmin, xmax, ymin, ymax):
        # Update the current batch box coordinates (separate from pick ROI)
        self.current_batch_box = (xmin, xmax, ymin, ymax)
        self.display_image()

    def select_area(self):
        # Get coordinates from inputs and draw the rectangle for initial fitting
        try:
            xmin = int(self.xmin_input.text())
            xmax = int(self.xmax_input.text())
            ymin = int(self.ymin_input.text())
            ymax = int(self.ymax_input.text())
            if xmin >= xmax or ymin >= ymax:
                QMessageBox.warning(self, "Invalid Coordinates", "Minimum values must be less than maximum values.")
                return
            self.selected_area = (xmin, xmax, ymin, ymax)
            # Now that a region is picked, allow/populate the Bragg edge table
            self._bragg_table_ready = True
            self.update_bragg_edge_table()
            self.display_image()  # Refresh display to show the selected area
            self.update_plots()  # Update all plots
        except ValueError:
            self.message_box.append("Please enter valid integers for the coordinates.")

    # ──────────────────────────────────────────────────────────────
    # Interactive ROI dragging on the canvas (yellow box)
    # ──────────────────────────────────────────────────────────────
    def _on_canvas_press(self, event):
        if event.button != 1 or event.inaxes != self.canvas.axes:
            return
        if not self.images or event.xdata is None or event.ydata is None:
            return

        resize_requested = self._is_resize_modifier_pressed(event)

        # Priority: yellow macro-pixel box. Corner resize requires Ctrl to avoid
        # conflicting with move gestures on very small boxes.
        if self.selected_area:
            corner = (
                self._hit_corner(self.selected_area, event.xdata, event.ydata)
                if resize_requested else None
            )
            if corner is not None:
                self._dragging_roi = True
                self._dragging_roi_mode = "resize"
                self._roi_active_corner = corner
                return

            if self._point_in_slice_box(self.selected_area, event.xdata, event.ydata):
                xmin, xmax, ymin, ymax = self.selected_area
                left, top, _, _ = self._slice_box_edges(xmin, xmax, ymin, ymax)
                self._dragging_roi = True
                self._dragging_roi_mode = "move"
                self._roi_drag_offset = (event.xdata - left, event.ydata - top)
                return

        # Otherwise check orange batch ROI, with the same Ctrl+corner resize rule.
        batch_roi = self._get_batch_roi()
        if batch_roi:
            corner = (
                self._hit_corner(batch_roi, event.xdata, event.ydata)
                if resize_requested else None
            )
            if corner is not None:
                self._dragging_batch_roi = True
                self._dragging_batch_mode = "resize"
                self._batch_active_corner = corner
                return

            if self._point_in_slice_box(batch_roi, event.xdata, event.ydata):
                bxmin, bxmax, bymin, bymax = batch_roi
                left, top, _, _ = self._slice_box_edges(bxmin, bxmax, bymin, bymax)
                self._dragging_batch_roi = True
                self._dragging_batch_mode = "move"
                self._batch_drag_offset = (event.xdata - left, event.ydata - top)
                return

        if self._is_fitting_navigation_active():
            return

        self._dragging_image_pan = True
        self._image_pan_start_xpixel = getattr(event, "x", None)
        self._image_pan_start_ypixel = getattr(event, "y", None)
        self._image_pan_start_xlim = self.canvas.axes.get_xlim()
        self._image_pan_start_ylim = self.canvas.axes.get_ylim()

    def _on_canvas_motion(self, event):
        if event.inaxes != self.canvas.axes or event.xdata is None or event.ydata is None or not self.images:
            return

        img_h, img_w = self.images[0].shape

        if self._dragging_roi and self.selected_area:
            if self._dragging_roi_mode == "move":
                xmin, xmax, ymin, ymax = self.selected_area
                box_w = xmax - xmin
                box_h = ymax - ymin
                offset_x, offset_y = self._roi_drag_offset

                new_xmin = self._pixel_edge_to_index(event.xdata - offset_x)
                new_ymin = self._pixel_edge_to_index(event.ydata - offset_y)

                new_xmin = max(0, min(new_xmin, img_w - box_w))
                new_ymin = max(0, min(new_ymin, img_h - box_h))

                new_xmax = new_xmin + box_w
                new_ymax = new_ymin + box_h

                self.selected_area = (new_xmin, new_xmax, new_ymin, new_ymax)
            elif self._dragging_roi_mode == "resize":
                self.selected_area = self._resize_box(
                    self.selected_area,
                    self._roi_active_corner,
                    event.xdata,
                    event.ydata,
                    img_w,
                    img_h,
                )
            self._sync_selected_area_inputs()
            self.display_image()
            return

        if self._dragging_batch_roi:
            batch_roi = self._get_batch_roi()
            if not batch_roi:
                return
            if self._dragging_batch_mode == "move":
                bxmin, bxmax, bymin, bymax = batch_roi
                box_w = bxmax - bxmin
                box_h = bymax - bymin
                offset_x, offset_y = self._batch_drag_offset

                new_xmin = self._pixel_edge_to_index(event.xdata - offset_x)
                new_ymin = self._pixel_edge_to_index(event.ydata - offset_y)

                new_xmin = max(0, min(new_xmin, img_w - box_w))
                new_ymin = max(0, min(new_ymin, img_h - box_h))

                new_xmax = new_xmin + box_w
                new_ymax = new_ymin + box_h
            else:  # resize
                new_xmin, new_xmax, new_ymin, new_ymax = self._resize_box(
                    batch_roi,
                    self._batch_active_corner,
                    event.xdata,
                    event.ydata,
                    img_w,
                    img_h,
                )

            # Update the batch ROI inputs (orange box)
            self.min_x_input.setText(str(new_xmin))
            self.max_x_input.setText(str(new_xmax))
            self.min_y_input.setText(str(new_ymin))
            self.max_y_input.setText(str(new_ymax))
            self.display_image()
            return

        if self._dragging_image_pan:
            self._pan_image_view(event, img_w, img_h)

    def _on_canvas_release(self, event):
        if self._dragging_roi:
            self._dragging_roi = False
            self._roi_drag_offset = (0.0, 0.0)
            self._dragging_roi_mode = "move"
            self._roi_active_corner = None
            # Recompute plots with the new ROI
            self.update_plots()
        if self._dragging_batch_roi:
            self._dragging_batch_roi = False
            self._batch_drag_offset = (0.0, 0.0)
            self._dragging_batch_mode = "move"
            self._batch_active_corner = None
        if self._dragging_image_pan:
            self._reset_image_pan_state()

    def _pan_image_view(self, event, img_w, img_h):
        """Pan the image viewer by shifting axis limits from the drag start."""
        if (
            self._image_pan_start_xpixel is None
            or self._image_pan_start_ypixel is None
            or self._image_pan_start_xlim is None
            or self._image_pan_start_ylim is None
        ):
            return

        ax = self.canvas.axes
        bbox = ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return

        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return

        dx_pixels = event_x - self._image_pan_start_xpixel
        dy_pixels = event_y - self._image_pan_start_ypixel
        x0, x1 = self._image_pan_start_xlim
        y0, y1 = self._image_pan_start_ylim
        dx = dx_pixels * ((x1 - x0) / bbox.width)
        dy = dy_pixels * ((y1 - y0) / bbox.height)
        x_bounds = (-0.5, img_w - 0.5)
        y_bounds = (-0.5, img_h - 0.5)

        new_xlim = self._shift_image_axis_limits(
            self._image_pan_start_xlim,
            dx,
            x_bounds,
        )
        new_ylim = self._shift_image_axis_limits(
            self._image_pan_start_ylim,
            dy,
            y_bounds,
        )
        self.canvas.axes.set_xlim(*new_xlim)
        self.canvas.axes.set_ylim(*new_ylim)
        self.canvas.draw_idle()

    def _reset_image_pan_state(self):
        """Clear image viewer drag-pan state."""
        self._dragging_image_pan = False
        self._image_pan_start_xpixel = None
        self._image_pan_start_ypixel = None
        self._image_pan_start_xlim = None
        self._image_pan_start_ylim = None

    @staticmethod
    def _shift_image_axis_limits(limits, delta, bounds):
        """Shift axis limits by a drag delta while preserving direction and bounds."""
        a0, a1 = limits
        inverted = a0 > a1
        lo, hi = (a1, a0) if inverted else (a0, a1)
        span = hi - lo
        full_span = bounds[1] - bounds[0]
        if span <= 0:
            return limits
        if span >= full_span:
            new_lo, new_hi = bounds
        else:
            new_lo = lo - delta
            new_hi = hi - delta
            if new_lo < bounds[0]:
                new_hi += bounds[0] - new_lo
                new_lo = bounds[0]
            if new_hi > bounds[1]:
                new_lo -= new_hi - bounds[1]
                new_hi = bounds[1]
            new_lo = max(new_lo, bounds[0])
            new_hi = min(new_hi, bounds[1])
        return (new_hi, new_lo) if inverted else (new_lo, new_hi)

    def _is_fitting_navigation_active(self):
        """Return True when matplotlib toolbar pan/zoom mode is active."""
        mode = str(getattr(self.toolbar, "mode", "")).strip().lower()
        if mode and mode not in ("none",):
            return True
        active = getattr(self.toolbar, "_active", None)
        return bool(str(active).strip()) if active is not None else False

    def _on_canvas_scroll(self, event):
        """Zoom in/out around cursor with mouse wheel."""
        if not self.images or event.inaxes != self.canvas.axes:
            return
        if self._is_fitting_navigation_active():
            return
        if self._dragging_roi or self._dragging_batch_roi or self._dragging_image_pan:
            return
        if event.button not in ("up", "down"):
            return

        ax = self.canvas.axes
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        y_center = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        scale = 0.9 if event.button == "up" else 1.1

        image_h, image_w = self.images[0].shape
        x_bounds = (-0.5, image_w - 0.5)
        y_bounds = (-0.5, image_h - 0.5)

        def _zoom_axis(a0, a1, center, bounds):
            inverted = a0 > a1
            lo, hi = (a1, a0) if inverted else (a0, a1)
            if hi <= lo:
                return a0, a1

            center = min(max(center, bounds[0]), bounds[1])
            rel = (center - lo) / (hi - lo)
            rel = min(max(rel, 0.0), 1.0)
            span = (hi - lo) * scale
            full_span = bounds[1] - bounds[0]

            if span >= full_span:
                new_lo, new_hi = bounds
            else:
                new_lo = center - span * rel
                new_hi = center + span * (1.0 - rel)
                if new_lo < bounds[0]:
                    new_hi += (bounds[0] - new_lo)
                    new_lo = bounds[0]
                if new_hi > bounds[1]:
                    new_lo -= (new_hi - bounds[1])
                    new_hi = bounds[1]
                new_lo = max(new_lo, bounds[0])
                new_hi = min(new_hi, bounds[1])

            return (new_hi, new_lo) if inverted else (new_lo, new_hi)

        new_x0, new_x1 = _zoom_axis(x0, x1, x_center, x_bounds)
        new_y0, new_y1 = _zoom_axis(y0, y1, y_center, y_bounds)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _connect_fit_canvas_interactions(self):
        """Attach wheel-zoom and drag-pan handlers to fitting result canvases."""
        for canvas in (
            getattr(self, "plot_canvas_a", None),
            getattr(self, "plot_canvas_b", None),
            getattr(self, "plot_canvas_c", None),
            getattr(self, "plot_canvas_d", None),
        ):
            if canvas is None or getattr(canvas, "_fit_nav_connected", False):
                continue
            self._fit_canvas_nav_state[id(canvas)] = {
                "panning": False,
                "active_axis": None,
                "start_xdata": None,
                "start_ydata": None,
                "start_xlim": None,
                "start_main_ylim": None,
                "start_residual_ylim": None,
            }
            canvas.mpl_connect(
                "button_press_event",
                lambda event, target=canvas: self._on_fit_canvas_press(target, event),
            )
            canvas.mpl_connect(
                "motion_notify_event",
                lambda event, target=canvas: self._on_fit_canvas_motion(target, event),
            )
            canvas.mpl_connect(
                "button_release_event",
                lambda event, target=canvas: self._on_fit_canvas_release(target, event),
            )
            canvas.mpl_connect(
                "scroll_event",
                lambda event, target=canvas: self._on_fit_canvas_scroll(target, event),
            )
            canvas._fit_nav_connected = True

    def _fit_canvas_state(self, canvas):
        """Return mutable navigation state for one fitting canvas."""
        if not hasattr(self, "_fit_canvas_nav_state"):
            self._fit_canvas_nav_state = {}
        return self._fit_canvas_nav_state.setdefault(
            id(canvas),
            {
                "panning": False,
                "active_axis": None,
                "start_xdata": None,
                "start_ydata": None,
                "start_xlim": None,
                "start_main_ylim": None,
                "start_residual_ylim": None,
            },
        )

    def _on_fit_canvas_press(self, canvas, event):
        """Start drag-pan on a fitting canvas."""
        if event.button != 1:
            return
        main_ax, residual_ax = self._ensure_fit_canvas_axes(canvas)
        if event.inaxes not in (main_ax, residual_ax):
            return
        if event.xdata is None or event.ydata is None:
            return
        state = self._fit_canvas_state(canvas)
        state["panning"] = True
        state["active_axis"] = "main" if event.inaxes is main_ax else "residual"
        state["start_xdata"] = event.xdata
        state["start_ydata"] = event.ydata
        state["start_xlim"] = main_ax.get_xlim()
        state["start_main_ylim"] = main_ax.get_ylim()
        state["start_residual_ylim"] = residual_ax.get_ylim()

    def _on_fit_canvas_motion(self, canvas, event):
        """Apply drag-pan to fitting canvas (x is shared between main and residual)."""
        state = self._fit_canvas_state(canvas)
        if not state.get("panning"):
            return
        main_ax, residual_ax = self._ensure_fit_canvas_axes(canvas)
        if event.inaxes not in (main_ax, residual_ax):
            return
        if event.xdata is None or event.ydata is None:
            return
        if state["start_xdata"] is None or state["start_xlim"] is None:
            return

        dx = event.xdata - state["start_xdata"]
        x0, x1 = state["start_xlim"]
        main_ax.set_xlim(x0 - dx, x1 - dx)

        dy = event.ydata - state["start_ydata"]
        if state.get("active_axis") == "main" and state.get("start_main_ylim") is not None:
            y0, y1 = state["start_main_ylim"]
            main_ax.set_ylim(y0 - dy, y1 - dy)
        elif state.get("active_axis") == "residual" and state.get("start_residual_ylim") is not None:
            y0, y1 = state["start_residual_ylim"]
            residual_ax.set_ylim(y0 - dy, y1 - dy)

        canvas.draw_idle()

    def _on_fit_canvas_release(self, canvas, _event):
        """End drag-pan on a fitting canvas."""
        state = self._fit_canvas_state(canvas)
        state["panning"] = False
        state["active_axis"] = None
        state["start_xdata"] = None
        state["start_ydata"] = None
        state["start_xlim"] = None
        state["start_main_ylim"] = None
        state["start_residual_ylim"] = None

    @staticmethod
    def _zoom_axis_limits(a0, a1, center, scale):
        """Return zoomed limits around center while preserving axis direction."""
        inverted = a0 > a1
        lo, hi = (a1, a0) if inverted else (a0, a1)
        span = hi - lo
        if span <= 0:
            return a0, a1

        if center is None:
            center = (lo + hi) / 2.0
        rel = (center - lo) / span
        rel = min(max(rel, 0.0), 1.0)

        new_span = span * scale
        new_lo = center - new_span * rel
        new_hi = center + new_span * (1.0 - rel)

        if abs(new_hi - new_lo) < 1e-12:
            return a0, a1
        return (new_hi, new_lo) if inverted else (new_lo, new_hi)

    def _on_fit_canvas_scroll(self, canvas, event):
        """Zoom a fitting canvas with mouse wheel. X zoom is shared with residual."""
        if event.button not in ("up", "down"):
            return
        main_ax, residual_ax = self._ensure_fit_canvas_axes(canvas)
        if event.inaxes not in (main_ax, residual_ax):
            return

        state = self._fit_canvas_state(canvas)
        if state.get("panning"):
            return

        scale = 0.9 if event.button == "up" else 1.1
        x0, x1 = main_ax.get_xlim()
        x_center = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        new_x0, new_x1 = self._zoom_axis_limits(x0, x1, x_center, scale)
        main_ax.set_xlim(new_x0, new_x1)

        target_ax = main_ax if event.inaxes is main_ax else residual_ax
        y0, y1 = target_ax.get_ylim()
        y_center = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        new_y0, new_y1 = self._zoom_axis_limits(y0, y1, y_center, scale)
        target_ax.set_ylim(new_y0, new_y1)
        canvas.draw_idle()

    def _sync_selected_area_inputs(self):
        """Keep coordinate text boxes in sync with the current ROI."""
        if not self.selected_area:
            return
        xmin, xmax, ymin, ymax = self.selected_area
        self.xmin_input.setText(str(xmin))
        self.xmax_input.setText(str(xmax))
        self.ymin_input.setText(str(ymin))
        self.ymax_input.setText(str(ymax))

    def _get_macro_pixel_size(self):
        """
        Return the width/height of the picked macro-pixel ROI if available.
        Falls back to the ROI coordinate inputs when no selected_area is set.
        """
        if getattr(self, "selected_area", None):
            xmin, xmax, ymin, ymax = self.selected_area
            if xmin < xmax and ymin < ymax:
                return xmax - xmin, ymax - ymin
        try:
            xmin = int(self.xmin_input.text())
            xmax = int(self.xmax_input.text())
            ymin = int(self.ymin_input.text())
            ymax = int(self.ymax_input.text())
            if xmin < xmax and ymin < ymax:
                return xmax - xmin, ymax - ymin
        except ValueError:
            pass
        return None, None

    def _confirm_batch_box_size(self, box_width, box_height):
        """
        If the batch box size differs from the picked macro-pixel size, prompt the user.
        Returns True to proceed, False to cancel.
        """
        macro_w, macro_h = self._get_macro_pixel_size()
        if macro_w is None or macro_h is None:
            return True
        if box_width == macro_w and box_height == macro_h:
            return True

        reply = QMessageBox.question(
            self,
            "Batch Box Size Mismatch",
            (
                f"Batch box size ({box_width} x {box_height}) differs from the picked "
                f"macro-pixel size ({macro_w} x {macro_h}).\n"
                f"Start batch fitting using the current batch box size ({box_width} x {box_height})?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            return True

        self.message_box.append("Batch fitting cancelled: box size differs from macro-pixel size.")
        return False

    def _get_batch_roi(self):
        """Return current batch ROI (orange) as (xmin, xmax, ymin, ymax) or None if invalid."""
        try:
            xmin = int(self.min_x_input.text())
            xmax = int(self.max_x_input.text())
            ymin = int(self.min_y_input.text())
            ymax = int(self.max_y_input.text())
        except ValueError:
            return None
        if xmin >= xmax or ymin >= ymax:
            return None
        return (xmin, xmax, ymin, ymax)

    @staticmethod
    def _roi_bounds_tooltip(label, example):
        return (
            f"{label} bounds use half-open pixel ranges: "
            "Min is included, Max is excluded. "
            "Pixels included are x_min <= x < x_max and y_min <= y < y_max. "
            f"{example}"
        )

    def _is_resize_modifier_pressed(self, event):
        """Return True when the Ctrl modifier is active for an ROI resize gesture."""
        key = str(getattr(event, "key", "") or "").lower()
        if "control" in key or "ctrl" in key:
            return True
        try:
            modifiers = QApplication.keyboardModifiers()
        except RuntimeError:
            return False
        return bool(modifiers & Qt.ControlModifier)

    @staticmethod
    def _pixel_edge_to_index(value):
        """Convert a displayed pixel edge coordinate back to an integer slice bound."""
        return int(np.floor(value + 0.5))

    @staticmethod
    def _slice_box_edges(xmin, xmax, ymin, ymax):
        """Return displayed pixel-edge bounds for a half-open ROI slice."""
        return xmin - 0.5, ymin - 0.5, xmax - 0.5, ymax - 0.5

    @classmethod
    def _slice_box_patch_args(cls, xmin, xmax, ymin, ymax):
        """Return Rectangle args that outline the pixels covered by a slice ROI."""
        left, top, right, bottom = cls._slice_box_edges(xmin, xmax, ymin, ymax)
        return (left, top), right - left, bottom - top

    def _point_in_slice_box(self, box, x, y):
        """Return True when a point is inside the displayed bounds of a slice ROI."""
        if x is None or y is None or not box:
            return False
        left, top, right, bottom = self._slice_box_edges(*box)
        return left <= x <= right and top <= y <= bottom

    def _hit_corner(self, box, x, y, threshold=8):
        """Return corner label if (x,y) is near a corner of the box."""
        if x is None or y is None or not box:
            return None
        left, top, right, bottom = self._slice_box_edges(*box)
        corners = {
            "tl": (left, top),
            "tr": (right, top),
            "bl": (left, bottom),
            "br": (right, bottom),
        }
        for label, (cx, cy) in corners.items():
            if abs(x - cx) <= threshold and abs(y - cy) <= threshold:
                return label
        return None

    def _resize_box(self, box, corner, x, y, img_w, img_h, min_w=2, min_h=2):
        """Resize box based on dragged corner, clamped to image bounds."""
        xmin, xmax, ymin, ymax = box
        if corner is None:
            return box
        if "l" in corner:
            xmin = max(0, min(self._pixel_edge_to_index(x), xmax - min_w))
        if "r" in corner:
            xmax = min(img_w, max(self._pixel_edge_to_index(x), xmin + min_w))
        if "t" in corner:
            ymin = max(0, min(self._pixel_edge_to_index(y), ymax - min_h))
        if "b" in corner:
            ymax = min(img_h, max(self._pixel_edge_to_index(y), ymin + min_h))
        return (xmin, xmax, ymin, ymax)

    def _ensure_fit_canvas_axes(self, canvas):
        """Ensure a fitting canvas has a main axis and a residual axis."""
        fig = canvas.fig
        main_ax = getattr(canvas, "axes", None)
        residual_ax = getattr(canvas, "residual_axes", None)

        valid_axes = (
            main_ax is not None
            and residual_ax is not None
            and main_ax in fig.axes
            and residual_ax in fig.axes
        )
        if not valid_axes:
            fig.clf()
            grid = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
            main_ax = fig.add_subplot(grid[0, 0])
            residual_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
            canvas.axes = main_ax
            canvas.residual_axes = residual_ax

        residual_ax.set_xlabel(self._fit_canvas_xlabel(canvas))
        return main_ax, residual_ax

    def _fit_canvas_xlabel(self, canvas):
        """Return residual x-axis label text for a fit canvas based on its 2x2 position."""
        top_row = (
            getattr(self, "plot_canvas_a", None),
            getattr(self, "plot_canvas_b", None),
        )
        if canvas in top_row:
            return ""
        return "Wavelength (A)"

    @staticmethod
    def _valid_region_bounds(r1_min, r1_max, r2_min, r2_max, r3_min, r3_max):
        """Validate that region bounds are finite and strictly increasing."""
        values = (r1_min, r1_max, r2_min, r2_max, r3_min, r3_max)
        if any(v is None for v in values):
            return False
        try:
            vals = [float(v) for v in values]
        except (TypeError, ValueError):
            return False
        if any(not np.isfinite(v) for v in vals):
            return False
        return vals[0] < vals[1] and vals[2] < vals[3] and vals[4] < vals[5]

    def _active_region_bounds(self):
        """
        Return active region bounds as (r1_min, r1_max, r2_min, r2_max, r3_min, r3_max).
        Prefer cached current_* values, then fall back to the selected/first table row.
        """
        cached = (
            getattr(self, "current_r1_min", None),
            getattr(self, "current_r1_max", None),
            getattr(self, "current_r2_min", None),
            getattr(self, "current_r2_max", None),
            getattr(self, "current_r3_min", None),
            getattr(self, "current_r3_max", None),
        )
        if self._valid_region_bounds(*cached):
            return tuple(float(v) for v in cached)

        table = getattr(self, "bragg_table", None)
        if table is None or table.rowCount() == 0:
            return None

        row = table.currentRow()
        if row < 0:
            row = 0

        try:
            bounds = (
                float(table.item(row, 4).text()),
                float(table.item(row, 5).text()),
                float(table.item(row, 2).text()),
                float(table.item(row, 3).text()),
                float(table.item(row, 6).text()),
                float(table.item(row, 7).text()),
            )
        except (AttributeError, TypeError, ValueError):
            return None

        if not self._valid_region_bounds(*bounds):
            return None

        self.current_r1_min, self.current_r1_max = bounds[0], bounds[1]
        self.current_r2_min, self.current_r2_max = bounds[2], bounds[3]
        self.current_r3_min, self.current_r3_max = bounds[4], bounds[5]
        return bounds

    @staticmethod
    def _xlim_with_margin(xmin, xmax, fraction=0.02):
        """Return x-limits padded by a fraction of the span on both sides."""
        try:
            xmin = float(xmin)
            xmax = float(xmax)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(xmin) and np.isfinite(xmax) and xmin < xmax):
            return None

        span = xmax - xmin
        # Keep a tiny absolute fallback for near-zero spans.
        pad = max(span * fraction, 1e-6)
        return xmin - pad, xmax + pad

    def _initialize_fit_canvas(self, canvas, title=None):
        """Clear and initialize a fit canvas with an empty residual panel."""
        main_ax, residual_ax = self._ensure_fit_canvas_axes(canvas)
        main_ax.clear()
        residual_ax.clear()
        # Clear removes artists but we also force autoscale back on after manual pan/zoom.
        main_ax.set_autoscalex_on(True)
        main_ax.set_autoscaley_on(True)
        residual_ax.set_autoscalex_on(True)
        residual_ax.set_autoscaley_on(True)

        font_size = getattr(self, "plot_font_size", 12)
        residual_font_size = font_size

        if title:
            main_ax.set_title(title, fontsize=font_size)
        main_ax.set_ylabel("")
        main_ax.tick_params(axis="both", labelsize=font_size, direction="in")
        main_ax.tick_params(axis="x", labelbottom=False, direction="in")
        main_ax.xaxis.label.set_size(font_size)
        main_ax.yaxis.label.set_size(font_size)

        zero_line = residual_ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        zero_line.set_gid("residual_zero")
        residual_ax.set_ylabel("")
        residual_ax.set_xlabel(self._fit_canvas_xlabel(canvas), fontsize=residual_font_size)
        residual_ax.tick_params(axis="both", labelsize=residual_font_size, direction="in")
        residual_ax.xaxis.label.set_size(residual_font_size)
        residual_ax.yaxis.label.set_size(residual_font_size)

        # Reserve enough figure margins for axis labels/titles in 2x2 plot layout.
        left_margin = min(0.28, max(0.18, 0.12 + 0.006 * font_size))
        bottom_margin = min(0.30, max(0.16, 0.10 + 0.006 * font_size))
        canvas.fig.subplots_adjust(
            left=left_margin,
            right=0.98,
            top=0.96,
            bottom=bottom_margin,
            hspace=0.0,
        )

        return main_ax, residual_ax

    @staticmethod
    def _segment_bounds_from_x(x_arr, gap_factor=8.0):
        """Return (start, end) segment bounds, splitting where x has large gaps."""
        if x_arr.size < 2:
            return [(0, x_arr.size)]

        dx = np.diff(x_arr)
        if np.any(dx < 0):
            return [(0, x_arr.size)]

        positive_dx = dx[(dx > 0) & np.isfinite(dx)]
        if positive_dx.size == 0:
            return [(0, x_arr.size)]

        base_step = np.median(positive_dx)
        if not np.isfinite(base_step) or base_step <= 0:
            return [(0, x_arr.size)]

        gap_threshold = base_step * gap_factor
        gap_indices = np.where(dx > gap_threshold)[0]
        if gap_indices.size == 0:
            return [(0, x_arr.size)]

        bounds = []
        start = 0
        for idx in gap_indices:
            end = idx + 1
            if end > start:
                bounds.append((start, end))
            start = end
        if start < x_arr.size:
            bounds.append((start, x_arr.size))
        return bounds

    def _plot_line(self, ax, x_data, y_data, split_on_gaps=False, **plot_kwargs):
        """Plot line data, optionally breaking segments across large wavelength gaps."""
        x_arr = np.asarray(x_data)
        y_arr = np.asarray(y_data)
        n_points = min(x_arr.size, y_arr.size)
        if n_points == 0:
            return

        x_arr = x_arr[:n_points]
        y_arr = y_arr[:n_points]
        if not split_on_gaps:
            ax.plot(x_arr, y_arr, **plot_kwargs)
            return

        for start, end in self._segment_bounds_from_x(x_arr):
            if end - start < 1:
                continue
            ax.plot(x_arr[start:end], y_arr[start:end], **plot_kwargs)

    def _plot_residual_line(self, canvas, x_data, y_data, y_fit, split_on_gaps=False):
        """Plot residual line (data - fit) on the residual axis of a fit canvas."""
        _, residual_ax = self._ensure_fit_canvas_axes(canvas)
        x_arr = np.asarray(x_data)
        y_arr = np.asarray(y_data)
        y_fit_arr = np.asarray(y_fit)
        if x_arr.size == 0 or y_arr.size == 0 or y_fit_arr.size == 0:
            return

        n_points = min(x_arr.size, y_arr.size, y_fit_arr.size)
        if n_points == 0:
            return

        x_arr = x_arr[:n_points]
        y_arr = y_arr[:n_points]
        y_fit_arr = y_fit_arr[:n_points]

        has_zero_line = any(line.get_gid() == "residual_zero" for line in residual_ax.lines)
        if not has_zero_line:
            zero_line = residual_ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            zero_line.set_gid("residual_zero")

        residual = y_arr - y_fit_arr
        self._plot_line(
            residual_ax,
            x_arr,
            residual,
            split_on_gaps=split_on_gaps,
            color="#444444",
            linewidth=0.5,
            alpha=0.95,
        )

    def update_plots(self):
        if not self.images or not self.selected_area:
            return

        # Sum intensity in the selected area for each image
        intensities = []
        for img in self.images:
            xmin, xmax, ymin, ymax = self.selected_area
            selected_area = img[ymin:ymax, xmin:xmax]
            intensities.append(np.sum(selected_area))

        if not intensities:
            return

        self.intensities = np.array(intensities)/[(xmax-xmin)*(ymax-ymin)]

        # Apply smoothing if checkbox is checked
        if self.smooth_checkbox.isChecked() and len(self.intensities) > 2:
            self.intensities = np.convolve(self.intensities, np.ones(3)/3, mode='same')

        # Clear previous fitted parameters
        self.params_region1 = None
        self.params_region2 = None
        self.params_region3 = {}

        if not hasattr(self, "wavelengths") or self.wavelengths is None or len(self.wavelengths) != len(self.intensities):
            self.message_box.append("Cannot plot because wavelength data does not match image count.")
            return

        # Get the wavelength range from inputs, if provided
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
        except ValueError:
            # If inputs are empty or invalid, use full range
            min_wavelength = self.wavelengths[0]
            max_wavelength = self.wavelengths[-1]

        if self.wavelengths is None or len(self.wavelengths) != len(self.intensities):
            self.message_box.append("Wavelength data not aligned with intensities.")
            return

        try:
            # Filter the wavelengths and intensities based on the range
            mask = (self.wavelengths >= min_wavelength) & (self.wavelengths <= max_wavelength)
            if mask.sum() == 0:
                self.message_box.append("No wavelengths in the selected range.")
                return
            self.selected_region_x = self.wavelengths[mask]  # Store as instance variable
            self.selected_region_y = self.intensities[mask]  # Store as instance variable
        except (TypeError, ValueError, AttributeError):
            self.message_box.append("Select a Bragg Edge to display the regions")

        # Plot (a): Intensity vs Wavelength
        ax_a, _ = self._initialize_fit_canvas(self.plot_canvas_a)
        ax_a.plot(
            self.selected_region_x,
            self.selected_region_y,
            'o',
            markersize=getattr(self, "symbol_size", 4),
            markerfacecolor='blue',
            markeredgecolor='none',
            # label='Experimental Data'
        )

        # **Calculate and Plot Theoretical Bragg Edges within the Range**
        edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength) if self.show_theoretical_checkbox.isChecked() else []
        for (hkl, x_hkl) in edges_in_range:
            # Plot vertical dashed line for the theoretical Bragg edge
            ax_a.axvline(
                x=x_hkl,
                color='red',
                linestyle='--',
                # label='Theoretical Bragg Edge'
            )
            # Annotate the (h, k, l) index near the top of the plot
            y_max = ax_a.get_ylim()[1]
            ax_a.text(
                x_hkl * 1.02,
                y_max * 0.95,  # Position text slightly below the top
                f'hkl{hkl}',
                rotation=90,
                verticalalignment='top',
                color='red',
                fontsize=getattr(self, "plot_font_size", 12),
            )

        # Keep top-left x-range tied to the current global wavelength window.
        if np.isfinite(min_wavelength) and np.isfinite(max_wavelength) and min_wavelength < max_wavelength:
            ax_a.set_xlim(min_wavelength, max_wavelength)

        # **Handle Legend to Avoid Duplicate Labels**
        handles, labels = ax_a.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        # self.plot_canvas_a.axes.legend(unique_handles, unique_labels)

        self.plot_canvas_a.draw()

        active_bounds = self._active_region_bounds()
        if active_bounds is None:
            for name, canvas in (
                ("Region 1", self.plot_canvas_b),
                ("Region 2", self.plot_canvas_c),
                ("Region 3", self.plot_canvas_d),
            ):
                ax_main, _ = self._initialize_fit_canvas(canvas)
                ax_main.text(
                    0.98,
                    0.98,
                    name,
                    transform=ax_main.transAxes,
                    ha="right",
                    va="top",
                    fontsize=getattr(self, "plot_font_size", 12),
                )
                canvas.draw()
            return

        r1_min, r1_max, r2_min, r2_max, r3_min, r3_max = active_bounds
        regions = [
            ("Region 1", r2_min, r2_max, self.plot_canvas_b),
            ("Region 2", r1_min, r1_max, self.plot_canvas_c),
            ("Region 3", r3_min, r3_max, self.plot_canvas_d),
        ]


        for name, region_min_wavelength, region_max_wavelength, canvas in regions:
            ax_main, _ = self._initialize_fit_canvas(canvas)
            ax_main.text(
                0.98,
                0.98,
                name,
                transform=ax_main.transAxes,
                ha="right",
                va="top",
                fontsize=getattr(self, "plot_font_size", 12),
            )

            try:
                # Filter data for this region
                region_mask = (self.wavelengths >= region_min_wavelength) & (self.wavelengths <= region_max_wavelength)
                region_x = self.wavelengths[region_mask]
                region_y = self.intensities[region_mask]


                ax_main.plot(
                    region_x,
                    region_y,
                    'o',
                    markersize=getattr(self, "symbol_size", 4),
                    # label=name
                )
                # Force region x-range to follow the active table bounds.
                if (
                    np.isfinite(region_min_wavelength)
                    and np.isfinite(region_max_wavelength)
                    and region_min_wavelength < region_max_wavelength
                ):
                    xlim = self._xlim_with_margin(region_min_wavelength, region_max_wavelength)
                    if xlim is not None:
                        ax_main.set_xlim(*xlim)
                # canvas.axes.legend()
                canvas.draw()
            except Exception:
                # self.message_box.append("Select a Bragg Edge to display the regions")
                canvas.draw()
                continue

    def export_data(self):
        """
        This function exports the 'wavelength' and 'summed intensity' data to a CSV file.
        """
        if not hasattr(self, 'selected_region_x') or not hasattr(self, 'selected_region_y'):
            QMessageBox.warning(self, "No Data", "No data available to export. Please select an area and update the plots first.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                data = np.column_stack((self.selected_region_x, self.selected_region_y))
                header = "Wavelength,Summed Intensity"
                np.savetxt(file_name, data, delimiter=',', header=header, comments='')
                QMessageBox.information(self, "Save Data", f"Data saved successfully to:\n{file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Data", f"Failed to save data:\n{e}")

    def fit_all_regions(self):
        """
        Performs the Region 1, 2, 3 fit for each row in the bragg_table, but only
        for rows that have valid (non-empty) data in all required columns.
        """
        self.message_box.append("---------- Starting individual edges fitting ----------")

        def _range_from_rows(min_col, max_col, row_limit):
            xmins = []
            xmaxs = []
            for idx in range(row_limit):
                if not self.is_row_complete(idx):
                    continue
                try:
                    lo = float(self.bragg_table.item(idx, min_col).text())
                    hi = float(self.bragg_table.item(idx, max_col).text())
                except (AttributeError, TypeError, ValueError):
                    continue
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    xmins.append(lo)
                    xmaxs.append(hi)
            if not xmins or not xmaxs:
                return None
            return min(xmins), max(xmaxs)

        # Start each run from clean region canvases so previous fits do not accumulate.
        max_rows = min(5, self.bragg_table.rowCount())
        region_ranges = {
            "Region 1": _range_from_rows(2, 3, max_rows),  # plotted on canvas_b
            "Region 2": _range_from_rows(4, 5, max_rows),  # plotted on canvas_c
            "Region 3": _range_from_rows(6, 7, max_rows),  # plotted on canvas_d
        }
        for region_name, canvas in (
            ("Region 1", self.plot_canvas_b),
            ("Region 2", self.plot_canvas_c),
            ("Region 3", self.plot_canvas_d),
        ):
            ax_main, _ = self._initialize_fit_canvas(canvas)
            x_range = region_ranges.get(region_name)
            if x_range is not None:
                xlim = self._xlim_with_margin(x_range[0], x_range[1])
                if xlim is not None:
                    ax_main.set_xlim(*xlim)
            ax_main.text(
                0.98,
                0.98,
                region_name,
                transform=ax_main.transAxes,
                ha="right",
                va="top",
                fontsize=getattr(self, "plot_font_size", 12),
            )
            canvas.draw()

        for row in range(max_rows):
            # Check whether this row has all required inputs
            if not self.is_row_complete(row):
                # self.message_box.append(f"Skipping Row {row + 1}: incomplete or missing data.")
                continue

            # If row is valid, do the fit
            self.message_box.append(f"Fitting for Edge {row + 1}...")
            self.fit_region(row)

        self.message_box.append("---------- Individual edges fitting completed ----------")   
        self.fit_all_regions_button.setEnabled(True)

    def is_row_complete(self, row_index):
        """
        Returns True if the specified row has non-empty data in all required columns.
        Otherwise, returns False.

        Customize the 'required_columns' list to match your actual bragg_table layout.
        E.g., columns might be:
            0 => hkl
            1 => d (or N/A)
            2 => R1_min, 3 => R1_max
            4 => R2_min, 5 => R2_max
            6 => R3_min, 7 => R3_max
            8 => s, 9 => t, 10 => eta
        """
        # Example: columns 0..9 are all relevant. 
        # If you want to skip 'd' or 'hkl' checks, remove them from required_columns. 
        required_columns = [0,1,2,3,4,5,6,7,8,9,10]

        for col in required_columns:
            item = self.bragg_table.item(row_index, col)
            if not item or not item.text().strip():
                # Either no item or empty text => incomplete
                return False

        return True

    def _update_cell(self, row: int, col: int, text: str):
        """Create QTableWidgetItem if missing, then write text."""
        item = self.bragg_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self.bragg_table.setItem(row, col, item)
        item.setText(text)

    def fit_region(
        self,
        row_number,
        skip_ui_updates=False,
        row_data=None,
        wavelengths=None,
        intensities=None,
        fit_flags=None,
        selected_phase=None,
        structure_type=None,
        lattice_params=None,
    ):

        if fit_flags is None:
            fix_s = self.fix_s_checkbox.isChecked()
            fix_t = self.fix_t_checkbox.isChecked()
            fix_eta = self.fix_eta_checkbox.isChecked()
        else:
            fix_s, fix_t, fix_eta = fit_flags
        selected_phase = selected_phase if selected_phase is not None else self.phase_dropdown.currentText()
        is_known_phase = selected_phase != "Unknown_Phase"
        wavelengths_data = self.wavelengths if wavelengths is None else np.asarray(wavelengths)
        intensities_data = self.intensities if intensities is None else np.asarray(intensities)
        lattice_params_data = dict(lattice_params if lattice_params is not None else getattr(self, "lattice_params", {}))
        structure_type_data = structure_type if structure_type is not None else getattr(self, "structure_type", "fcc" if is_known_phase else "unknown")
        params_store = {}
        if not skip_ui_updates:
            if not hasattr(self, "params_unknown"):
                self.params_unknown = {}
            params_store = self.params_unknown

        try:
            # Validate structure type
            # if is_known_phase:
            #     structure_type = self.structure_type
            #     if structure_type != "cubic":
            #         raise ValueError(f"Non-cubic structure '{structure_type}' not supported. "
            #                           "Only cubic structures allowed.")

            # Get lattice parameter (cubic-specific logic)
            if is_known_phase:
                # Cubic structure requires only 'a'
                if "a" not in lattice_params_data:
                    raise ValueError("Missing cubic lattice parameter 'a'")
                a_guess = lattice_params_data["a"]
            else:
                # Unknown phase: use 'd' value from table as 'a' estimate
                if row_data is not None:
                    d_val = row_data.get("d")
                    if d_val is None:
                        raise ValueError("Missing 'd' value for unknown phase")
                    a_guess = float(d_val) / 2.0
                else:
                    d_item = self.bragg_table.item(row_number, 1)
                    if not d_item or not d_item.text().strip():
                        raise ValueError("Missing 'd' value for unknown phase")
                    a_guess = float(d_item.text())/2

            # Get region boundaries and parameters
            if row_data is not None:
                regions = row_data.get("regions") or []
                if len(regions) != 3:
                    raise ValueError("Invalid region bounds for selected edge")
                r1_min, r1_max = regions[1]["min_wavelength"], regions[1]["max_wavelength"]
                r2_min, r2_max = regions[0]["min_wavelength"], regions[0]["max_wavelength"]
                r3_min, r3_max = regions[2]["min_wavelength"], regions[2]["max_wavelength"]
                s_val = float(row_data.get("s"))
                t_val = float(row_data.get("t"))
                eta_val = float(row_data.get("eta"))
            else:
                r1_min = float(self.bragg_table.item(row_number, 4).text())
                r1_max = float(self.bragg_table.item(row_number, 5).text())
                r2_min = float(self.bragg_table.item(row_number, 2).text())
                r2_max = float(self.bragg_table.item(row_number, 3).text())
                r3_min = float(self.bragg_table.item(row_number, 6).text())
                r3_max = float(self.bragg_table.item(row_number, 7).text())
                s_val = float(self.bragg_table.item(row_number, 8).text())
                t_val = float(self.bragg_table.item(row_number, 9).text())
                eta_val = float(self.bragg_table.item(row_number, 10).text())

        except Exception as e:
            if not skip_ui_updates:
                self.message_box.append(f"Edge {row_number+1} Error: {str(e)}")
            return
        # -------------------------------------------------
        #  Region 1 Fitting
        # -------------------------------------------------
        try:
            mask_r1 = (wavelengths_data >= r1_min) & (wavelengths_data <= r1_max)
            x_r1 = wavelengths_data[mask_r1]
            y_r1 = intensities_data[mask_r1]

            if x_r1.size == 0:
                if not skip_ui_updates:
                    self.message_box.append(
                        f"Edge {row_number + 1} - Region 1: No data in the specified range.")
                return

            if not skip_ui_updates:
                ax_r1_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_c)
                ax_r1_main.plot(
                    x_r1,
                    y_r1,
                    'o',
                    markersize=getattr(self, "symbol_size", 4),
                    markerfacecolor='blue',
                    markeredgecolor='none',
                )
            p0 = [0,0]
            lower = [-10, -10]
            upper = [10, 10]
            popt_r1, _ = curve_fit(fitting_function_1, x_r1, y_r1, p0=p0, bounds=(lower, upper))
            a0, b0 = popt_r1
            params_store[row_number, 1] = popt_r1  # (a0, b0)
            fit_y1 = fitting_function_1(x_r1, *popt_r1)

            if not skip_ui_updates:
                ax_r1_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_c)
                ax_r1_main.plot(x_r1, fit_y1, 'r-', 
                                             # label=f"Edge {row_number + 1} Fit R1"
                                             )
                self._plot_residual_line(self.plot_canvas_c, x_r1, y_r1, fit_y1)
                # self.plot_canvas_c.axes.legend()
                self.plot_canvas_c.draw()
                self.message_box.append(
                    f"Edge {row_number + 1} - Fitted Region 1: "
                    f"a0={popt_r1[0]:.6f}, b0={popt_r1[1]:.6f}")

        except Exception as e:
            if not skip_ui_updates:
                self.message_box.append(
                    f"Edge {row_number + 1} - Error fitting Region 1: {e}")
            return

        # -------------------------------------------------
        #  Region 2 Fitting
        # -------------------------------------------------
        try:
            if (row_number, 1) not in params_store:
                # Region 1 must be fitted first
                return
            a0, b0 = params_store[row_number, 1]

            mask_r2 = (wavelengths_data >= r2_min) & (wavelengths_data <= r2_max)
            x_r2 = wavelengths_data[mask_r2]
            y_r2 = intensities_data[mask_r2]

            if x_r2.size == 0:
                if not skip_ui_updates:
                    self.message_box.append(
                        f"Edge {row_number + 1} - Region 2: No data in range.")
                return

            if not skip_ui_updates:
                ax_r2_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_b)
                ax_r2_main.plot(
                    x_r2,
                    y_r2,
                    'o',
                    markersize=getattr(self, "symbol_size", 4),
                    markerfacecolor='blue',
                    markeredgecolor='none',
                )

            popt_r2, _ = curve_fit(
                lambda xx, a_hkl, b_hkl: fitting_function_2(xx, a_hkl, b_hkl, a0, b0),
                x_r2, y_r2, p0=[0, 0]
            )
            a_hkl, b_hkl = popt_r2
            params_store[row_number, 2] = popt_r2  # (a_hkl, b_hkl)
            fit_y2 = fitting_function_2(x_r2, *popt_r2, a0, b0)

            if not skip_ui_updates:
                ax_r2_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_b)
                ax_r2_main.plot(
                    x_r2, fit_y2, 'r-', 
                    # label=f"Edge {row_number + 1} Fit R2"
                    )
                self._plot_residual_line(self.plot_canvas_b, x_r2, y_r2, fit_y2)
                # self.plot_canvas_b.axes.legend()
                self.plot_canvas_b.draw()
                self.message_box.append(
                    f"Edge {row_number + 1} - Fitted Region 2: "
                    f"a_hkl={popt_r2[0]:.6f}, b_hkl={popt_r2[1]:.6f}")

        except Exception as e:
            if not skip_ui_updates:
                self.message_box.append(
                    f"Edge {row_number + 1} - Error fitting Region 2: {e}")
            return

        # -------------------------------------------------
        #  Region 3 Fitting
        # -------------------------------------------------
        try:
            def span50(x):
                """Return (lower, upper) for a ±50 % interval around x."""
                d = 1 * max(abs(x), 1)        # half-width
                return x - d, x + d       # works for positives and negatives

            # 1. pull stage-1/2 estimates -----------------------------------------
            a0_hat,  b0_hat  = params_store[row_number, 1]
            a_hkl_hat, b_hkl_hat = params_store[row_number, 2]

            # 2. first four bounds ( ±50 % each ) ---------------------------------
            lb4, ub4 = zip(*(span50(p) for p in (a0_hat, b0_hat, a_hkl_hat, b_hkl_hat)))

            # ---------- hkl tuple ---------------------------------
            if is_known_phase:
                if row_data is not None and row_data.get("hkl") is not None:
                    h, k, l = row_data["hkl"]
                else:
                    hkl_item = self.bragg_table.item(row_number, 0)
                    h, k, l = map(int, hkl_item.text().strip("()").split(","))
                hkl = (h, k, l)
            else:
                hkl = f"edge{row_number+1}"        # label for unknown

            # ---------- masks & data ------------------------------
            mask_r3 = (wavelengths_data >= r3_min) & (wavelengths_data <= r3_max)
            x_r3    = wavelengths_data[mask_r3]
            y_r3    = intensities_data[mask_r3]

            if not skip_ui_updates:
                ax_r3_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_d)
                ax_r3_main.plot(
                    x_r3,
                    y_r3,
                    "o",
                    markersize=getattr(self, "symbol_size", 4),
                    markerfacecolor='blue',
                    markeredgecolor='none',
                )

            # ---------- build p0 / bounds -------------------------
            p0 = [a0_hat,  b0_hat,  a_hkl_hat,  b_hkl_hat,  a_guess]                    \
                 + ([] if fix_s  else [0.01]) \
                 + ([] if fix_t  else [0.1])  \
                 + ([] if fix_eta else [0.5])

            lb = list(lb4) + [a_guess*0.95]             \
                 + ([] if fix_s  else [0.0001])\
                 + ([] if fix_t  else [0.01])\
                 + ([] if fix_eta else [0])

            ub = list(ub4) + [a_guess*1.05]              \
                 + ([] if fix_s  else [0.01])   \
                 + ([] if fix_t  else [0.1])   \
                 + ([] if fix_eta else [1])

            # ----------------- inside Region-3: helper -----------------
            def func_r3(x, *params):
                a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = params[:4]
                idx = 4                      # start right after a_fit

                a_fit  = params[idx]; idx += 1
                s_fit  = s_val   if fix_s  else params[idx]; idx += (0 if fix_s  else 1)
                t_fit  = t_val   if fix_t  else params[idx]; idx += (0 if fix_t  else 1)
                eta_fit = eta_val if fix_eta else params[idx]

                return fitting_function_3(
                    x, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_fit, t_fit, eta_fit,
                    [hkl] if is_known_phase else [],
                    r3_min, r3_max,
                    "cubic", {"a": a_fit}
                )


            # ---------- run curve_fit ----------------------------
            popt_3, pcov_3 = curve_fit(
                func_r3, x_r3, y_r3, p0=p0,
                bounds=(lb, ub), maxfev=300
            )

            y3_fit   = func_r3(x_r3, *popt_3)
            resid_3  = y_r3 - y3_fit
            rms_3    = np.sqrt(np.mean(resid_3**2))

            # ----- 4. unpack results (now 8+ parameters) --
            a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = popt_3[:4]
            idx = 4
            a_fit  = popt_3[idx]*2;  a_unc = np.sqrt(pcov_3[idx, idx])*2; idx += 1
            if not fix_s:   s_fit  = popt_3[idx]; s_unc  = np.sqrt(pcov_3[idx, idx]); idx += 1
            else:           s_fit, s_unc = s_val, np.nan
            if not fix_t:   t_fit  = popt_3[idx]; t_unc  = np.sqrt(pcov_3[idx, idx]); idx += 1
            else:           t_fit, t_unc = t_val, np.nan
            if not fix_eta: eta_fit = popt_3[idx]; eta_unc = np.sqrt(pcov_3[idx, idx])
            else:           eta_fit, eta_unc = eta_val, np.nan


            # ----- convert a → d (for known phase) ---------------
            if is_known_phase:
                denom    = np.sqrt(h**2 + k**2 + l**2)
                d_fit    = a_fit / denom
                d_unc    = a_unc / denom
                x_edge   = d_fit
            else:
                d_fit    = a_fit               # use lattice‑like value
                d_unc    = a_unc
                x_edge   = a_fit

            # # ----- edge height (same formula) --------------------
            # f1 = np.exp(-(a0_fit + b0_fit * x_edge))
            # f2 = f1 * np.exp(-(a_hkl_fit + b_hkl_fit * x_edge))
            # edge_height = f1 - f2
            # self.params_unknown[row_number, 4] = edge_height


            # Use the same approach as pattern fitting for height/width
            # Height: max-min of fitted Region3 curve over its span
            # Width : FWHM of derivative of Region3 model around the edge
            try:
                if is_known_phase:
                    # Known phase: derive d-spacing from lattice + hkl.
                    structure = structure_type_data
                    lat = lattice_params_data or {"a": a_fit}
                    d_vals = calculate_x_hkl_general(structure, lat, [(h, k, l)])
                    d_hkl = d_vals[0] / 2.0 if d_vals and not np.isnan(d_vals[0]) else np.nan
                else:
                    # Unknown phase: a_fit is lambda_hkl, so d = lambda/2.
                    structure = "cubic"
                    d_hkl = a_fit / 2.0
                    lat = {"a": d_hkl}
            except (TypeError, ValueError, KeyError):
                d_hkl = np.nan

            if np.isnan(d_hkl) or d_hkl <= 0:
                edge_height = np.nan
                edge_width = np.nan
            else:
                x_edge = 2.0 * d_hkl

                # Height: plateau difference over Region 3
                try:
                    xx_h = np.linspace(r3_min, r3_max, 4000)
                    yy_h = fitting_function_3(
                        xx_h, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                        s_fit, t_fit, eta_fit, [hkl] if is_known_phase else [],
                        r3_min, r3_max,
                        structure, lat
                    )
                    edge_height = yy_h.max() - yy_h.min() if yy_h.size else np.nan
                except (ValueError, FloatingPointError):
                    edge_height = np.nan

                try:
                    span = max(0.2, 0.1 * (r3_max - r3_min))
                    start = max(r3_min, x_edge - span)
                    end = min(r3_max, x_edge + span)
                    if end <= start:
                        raise ValueError("Invalid span for width computation")
                    xx = np.linspace(start, end, 4000)
                    yy = fitting_function_3(
                        xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                        s_fit, t_fit, eta_fit, [hkl] if is_known_phase else [],
                        start, end,
                        structure, lat
                    )
                    if yy.size < 5:
                        raise ValueError("Insufficient points for width computation")
                    dy = np.gradient(yy, xx)
                    dy_max = dy.max()
                    if dy_max <= 0:
                        raise ValueError("Non-positive derivative peak")
                    half = dy_max / 2.0
                    indices = np.where(dy >= half)[0]
                    if indices.size == 0:
                        raise ValueError("No points above half maximum")
                    edge_width = xx[indices[-1]] - xx[indices[0]]
                except (ValueError, FloatingPointError):
                    edge_width = np.nan

            params_store[row_number, 4] = edge_height
            params_store[row_number, 5] = edge_width

            # <<< END NEW ----------------------------------------

            # ----- store everything for later use ----------------
            params_store[row_number, 3] = (
                d_fit, s_fit, t_fit, eta_fit,
                d_unc, s_unc, t_unc, eta_unc
            )

            # ----- return dict when skip_ui_updates --------------
            if skip_ui_updates:
                return {
                    "hkl":      hkl,
                    "x":        x_r3,
                    "fit":      func_r3(x_r3, *popt_3),

                    #  numbers for info box
                    "d_fit":    d_fit,   "d_unc":  d_unc,
                    "s_fit":    s_fit,   "s_unc":  s_unc,
                    "t_fit":    t_fit,   "t_unc":  t_unc,
                    "eta_fit":  eta_fit, "eta_unc":eta_unc,
                    "fit_params": (d_fit, s_fit, t_fit, eta_fit, d_unc, s_unc, t_unc, eta_unc),
                    "edge_height": edge_height,
                    "edge_width": edge_width,
                }

            # ---------- live plotting / messages (GUI) -----------
            if not skip_ui_updates:
                ax_r3_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_d)
                ax_r3_main.plot(
                    x_r3, func_r3(x_r3, *popt_3), "r-",
                    # label=f"Edge {hkl} Fit R3"
                )
                self._plot_residual_line(self.plot_canvas_d, x_r3, y_r3, y3_fit)
                # self.plot_canvas_d.axes.legend()
                self.plot_canvas_d.draw()
                self.message_box.append(
                    f"Edge {row_number + 1} - Fitted Region 1: "
                    f"a0_fit={a0_fit:.6f}, b0={b0_fit:.6f}")
                self.message_box.append(
                    f"Edge {row_number + 1} - Fitted Region 2: "
                    f"a_hkl_fit={a_hkl_fit:.6f}, b_hkl={b_hkl_fit:.6f}")

                msg = [f"d{hkl}: {d_fit:.6f} ± {d_unc:.6f}",
                       f"Edge Height = {edge_height:.6f}",
                       f"FWHM = {edge_width:.6f} Å",
                       f"RMS (Region-3) = {rms_3:.6f}",
                       # f"half = {half:.6f} Å"
                       ]
                if not fix_s:
                    msg.append(f"s: {s_fit:.6f} ± {s_unc:.6f}")
                if not fix_t:
                    msg.append(f"t: {t_fit:.6f} ± {t_unc:.6f}")
                if not fix_eta:
                    msg.append(f"η: {eta_fit:.3f} ± {eta_unc:.3f}")
                self.message_box.append(
                    f"Edge {hkl} – Fit Results:\n" + "\n".join(msg)
                )

                # update table cells
                self._update_cell(row_number, 8,  f"{s_fit:.6f}")
                self._update_cell(row_number, 9,  f"{t_fit:.6f}")
                self._update_cell(row_number, 10, f"{eta_fit:.3f}")

        except Exception as e:
            if not skip_ui_updates:
                self.message_box.append(
                    f"Edge {row_number+1} – Region‑3 Fit Error: {e}"
                )

    def batch_fit(self):
        """
        Initiates the batch fitting process over the ROI by calling the fit_full_pattern
        function for each box in the defined grid.
        """
        # Ensure ROI and box dimensions are defined
        try:
            box_width = int(self.box_width_input.text())
            box_height = int(self.box_height_input.text())
            step_x = int(self.step_x_input.text())
            step_y = int(self.step_y_input.text())
            min_x = int(self.min_x_input.text())
            max_x = int(self.max_x_input.text())
            min_y = int(self.min_y_input.text())
            max_y = int(self.max_y_input.text())
        except ValueError:
            self.message_box.append("Please enter valid integers for box size, step size, and ROI coordinates.")
            return

        if box_width <= 0 or box_height <= 0 or step_x <= 0 or step_y <= 0:
            self.message_box.append("Box and step sizes must be positive integers.")
            return

        if not self.images:
            self.message_box.append("No images loaded.")
            return

        # Check if full pattern fitting has been performed
        if not hasattr(self, 'fitted_s_values') or not self.fitted_s_values or not hasattr(self, 'fitted_t_values') or not self.fitted_t_values:
            QMessageBox.warning(
                self,
                "Initial Fit Required",
                "Please perform an initial full pattern fitting before starting batch fitting."
            )
            return

        if hasattr(self, 'batch_fit_worker') and self.batch_fit_worker.isRunning():
            self.message_box.append("Batch fitting is already in progress.")
            return

        # Warn if the batch box differs from the picked macro-pixel size
        if not self._confirm_batch_box_size(box_width, box_height):
            return

        # Compute the total number of boxes for progress bar
        fit_area_width = max_x - min_x
        fit_area_height = max_y - min_y
        total_boxes = ((fit_area_height - box_height) // step_y + 1) * ((fit_area_width - box_width) // step_x + 1)

        self.batch_progress_bar.setMinimum(0)
        self.batch_progress_bar.setMaximum(100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_label.setText("Batch Pattern Progress:")
        self.batch_remaining_time_label.setText("Remaining: Calculating...")
        self.batch_progress_dialog.show()

        interpolation_enabled = self.interpolation_checkbox.isChecked()
        self.fit_start_time = time.time()

        # Initialize remaining_time_timer if not already
        if not hasattr(self, 'remaining_time_timer'):
            self.remaining_time_timer = QTimer()
            self.remaining_time_timer.timeout.connect(self.update_remaining_time)

        self.remaining_time_timer.start(5000)

        if not hasattr(self, 'work_directory'):
            self.message_box.append("Working directory is not set. Please load FITS images first.")
            return

        # Get the state of the fix_s and fix_t checkboxes
        fix_s = self.fix_s_checkbox.isChecked()
        fix_t = self.fix_t_checkbox.isChecked()
        fix_eta = self.fix_eta_checkbox.isChecked()
        fit_context = self._build_batch_fit_context()

        # Start the batch fitting worker
        self.batch_fit_worker = BatchFitWorker(
            parent=self,
            images=self.images,
            wavelengths=self.wavelengths,
            fit_context=fit_context,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            box_width=box_width,
            box_height=box_height,
            step_x=step_x,
            step_y=step_y,
            total_boxes=total_boxes,
            interpolation_enabled=interpolation_enabled,
            work_directory=self.work_directory,
            fix_s=fix_s,
            fix_t=fix_t,
            fix_eta=fix_eta
        )
        self.batch_fit_worker.progress_updated.connect(self.update_progress_bar)
        self.batch_fit_worker.message.connect(self.append_message)
        self.batch_fit_worker.finished.connect(self.batch_fit_finished)
        self.batch_fit_worker.current_box_changed.connect(self.update_current_box)
        self.batch_fit_worker.start()

        # Start a timer for periodic display updates
        self.update_timer.start()

        # Start a timer for estimating remaining time
        self.fit_start_time = time.time()
        self.remaining_time_timer.start(5000)

    def stop_batch_fit(self):

        stopped_any = False

        # Stop the standard batch-fit worker
        if hasattr(self, 'batch_fit_worker') and self.batch_fit_worker.isRunning():
            self.batch_fit_worker.stop()
            self.message_box.append("Stop requested for BatchFitWorker. Please wait...")
            stopped_any = True

        # Stop the edges batch-fit worker
        if hasattr(self, 'batch_fit_edges_worker') and self.batch_fit_edges_worker.isRunning():
            self.batch_fit_edges_worker.stop()
            self.message_box.append("Stop requested for BatchFitEdgesWorker. Please wait...")
            stopped_any = True

        # If neither was running, inform the user
        if not stopped_any:
            self.message_box.append("No batch fitting is currently running.")


        # Reset remaining time label
        self.batch_remaining_time_label.setText("Remaining: ")
        self.batch_progress_dialog.hide()

    def update_progress_bar(self, value):
        self.batch_progress_bar.setValue(value)

    def update_remaining_time(self):
        # Get the current progress value
        value = self.batch_progress_bar.value()

        if value > 0:
            current_time = time.time()
            elapsed_time = current_time - self.fit_start_time
            estimated_total_time = elapsed_time / (value / 100.0)
            remaining_time = estimated_total_time - elapsed_time

            # Format the remaining time
            remaining_time_str = self.format_time(remaining_time)
            self.batch_remaining_time_label.setText(f"Remaining: {remaining_time_str}")
        else:
            self.batch_remaining_time_label.setText("Remaining: Calculating...")

    def format_time(self, seconds):
        # Convert seconds to hours, minutes, seconds
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def append_message(self, text):
        self.message_box.append(text)

    def batch_fit_finished(self, filename):
        if filename:
            self.message_box.append("Batch fitting completed.")
        else:
            self.message_box.append("Batch fitting completed with errors.")
        self.update_timer.stop()
        self.batch_progress_bar.setValue(100)
        self.batch_remaining_time_label.setText("Remaining: ")
        self.batch_progress_dialog.hide()

    def visualize_region3_fits(self):
        """
        Visualize the full-pattern fit (formerly "Region 3 fit") at user-specified positions.
        This takes bounding boxes around each (x_center, y_center) and calls fit_full_pattern_core.
        """
        selected_phase = self.phase_dropdown.currentText()
        is_known_phase = selected_phase != "Unknown_Phase"
        if not self.images:
            self.message_box.append("No images loaded.")
            return

        positions_text = self.positions_input.toPlainText().strip()
        if not positions_text:
            self.message_box.append("Please enter positions to visualize, e.g.\n100,200\n300,400")
            return

        # Parse positions
        positions = []
        for line in positions_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                x_str, y_str = line.split(',')
                x_pos = int(x_str.strip())
                y_pos = int(y_str.strip())
                positions.append((x_pos, y_pos))
            except ValueError:
                self.message_box.append(f"Invalid position format: '{line}'. Expected 'x,y'")
                continue

        if not positions:
            self.message_box.append("No valid positions to visualize.")
            return

        # Read box size
        try:
            box_width = int(self.box_width_input.text())
            box_height = int(self.box_height_input.text())
        except ValueError:
            self.message_box.append("Please enter valid integers for box width and height.")
            return

        # Read wavelength range
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
        except ValueError:
            self.message_box.append("Please enter valid min and max wavelengths.")
            return

        fix_s = self.fix_s_checkbox.isChecked()
        fix_t = self.fix_t_checkbox.isChecked()
        fix_eta = self.fix_eta_checkbox.isChecked()

        image_height, image_width = self.images[0].shape

        for (x_center, y_center) in positions:
            # Define bounding box around (x_center, y_center)
            x_tl = x_center - box_width // 2
            y_tl = y_center - box_height // 2

            # Check bounds
            if x_tl < 0 or (x_tl + box_width) > image_width or \
               y_tl < 0 or (y_tl + box_height) > image_height:
                self.message_box.append(
                    f"Position ({x_center}, {y_center}) with box size ({box_width}x{box_height}) exceeds image bounds."
                )
                continue

            # Sum intensities in the bounding box across all images
            intensities = []
            for img in self.images:
                sub_img = img[y_tl:y_tl + box_height, x_tl:x_tl + box_width]
                intensities.append(np.sum(sub_img))
            self.intensities = np.array(intensities)

            # Filter wavelengths to [min_wavelength, max_wavelength]
            mask = (self.wavelengths >= min_wavelength) & (self.wavelengths <= max_wavelength)
            x_exp = self.wavelengths[mask]
            y_exp = self.intensities[mask]

            # If no data in that range, skip
            if x_exp.size == 0:
                self.message_box.append(
                    f"No data in wavelength range [{min_wavelength}, {max_wavelength}] at position ({x_center}, {y_center})."
                )
                continue

            # Temporarily store in self so fit_full_pattern_core uses them
            self.selected_region_x = x_exp
            self.selected_region_y = y_exp

            # Attempt the full pattern fit
            if is_known_phase:
                result_dict, error_msg = self.fit_full_pattern_core(fix_s=fix_s, fix_t=fix_t, fix_eta=fix_eta)
                # ---------------------------------------------
                # 2A)  grab per‑edge Region‑3 fits (new)
                # ---------------------------------------------
                edge_fits = []
                for row_idx in range(self.bragg_table.rowCount()):
                    edge_fit = self.fit_region(row_idx, skip_ui_updates=True)
                    if edge_fit:                       # None means the fit failed
                        edge_fits.append(edge_fit)

                if error_msg:
                    self.message_box.append(
                        f"Fitting error at ({x_center},{y_center}): {error_msg}"
                    )
                    continue

                if not result_dict or not result_dict.get('success', False):
                    self.message_box.append(
                        f"Fitting at ({x_center},{y_center}) did not converge or returned no result."
                    )
                    continue

                # Extract final model data
                model_x = result_dict['x_data']     # sorted x
                model_y = result_dict['y_data']     # model
                residuals = result_dict['residuals']  # residual array

                # Build parameters to pass to dialog
                parameters = {
                    'lattice_params':     result_dict['lattice_params'],
                    'lattice_uncertainties': result_dict['lattice_uncertainties'],
                    'fitted_s':           result_dict['fitted_s'],
                    'fitted_t':           result_dict['fitted_t'],
                    'fitted_eta':           result_dict['fitted_eta'],
                    's_uncertainties':    result_dict['s_uncertainties'],
                    't_uncertainties':    result_dict['t_uncertainties'],
                    'eta_uncertainties':    result_dict['eta_uncertainties'],
                    'model_x':            model_x,
                    'model_y':            model_y,
                    'residuals':          residuals,
                    # Experimental data
                    'x_exp': x_exp,
                    'y_exp': y_exp,
                    'edge_fits': edge_fits,
                }
            else:
                edge_fits = []
                for row_idx in range(self.bragg_table.rowCount()):
                    edge_fit = self.fit_region(row_idx, skip_ui_updates=True)
                    if edge_fit:                       # None means the fit failed
                        edge_fits.append(edge_fit)


                # Build parameters to pass to dialog
                parameters = {
                    "lattice_params"        : {},
                    "lattice_uncertainties" : {},
                    "fitted_d"              : {},
                    "d_uncertainties"       : {},
                    "fitted_s"              : {},
                    "fitted_t"              : {},
                    "fitted_eta"            : {},
                    "s_uncertainties"       : {},
                    "t_uncertainties"       : {},
                    "eta_uncertainties"     : {},
                    "model_x"               : np.array([]),
                    "model_y"               : np.array([]),
                    "x_exp"                 : x_exp,
                    "y_exp"                 : y_exp,
                    "edge_fits"             : edge_fits,
                }


            dialog = FitVisualizationDialog(
                x_center, y_center,
                box_width, box_height,
                parameters,
                parent=self
            )
            dialog.show()

    def update_image(self):
        # Update the displayed image based on selected image adjustments
        self.display_image()

    def update_plot_font_size(self, value):
        """Update matplotlib font sizes across all canvases."""
        self.plot_font_size = int(value)
        font_size = self.plot_font_size
        mpl.rcParams['font.size'] = font_size
        mpl.rcParams['axes.labelsize'] = font_size
        mpl.rcParams['xtick.labelsize'] = font_size
        mpl.rcParams['ytick.labelsize'] = font_size
        mpl.rcParams['legend.fontsize'] = font_size

        for canvas in self._iter_plot_canvases():
            main_ax = getattr(canvas, "axes", None)
            if getattr(canvas, "residual_axes", None) is not None:
                left_margin = min(0.28, max(0.18, 0.12 + 0.006 * font_size))
                bottom_margin = min(0.30, max(0.16, 0.10 + 0.006 * font_size))
                canvas.fig.subplots_adjust(
                    left=left_margin,
                    right=0.98,
                    top=0.96,
                    bottom=bottom_margin,
                    hspace=0.0,
                )
            for ax in self._iter_axes_for_canvas(canvas):
                axis_font_size = font_size
                tick_kwargs = {"labelsize": axis_font_size}
                if getattr(canvas, "residual_axes", None) is not None:
                    tick_kwargs["direction"] = "in"
                ax.tick_params(**tick_kwargs)
                ax.xaxis.label.set_size(axis_font_size)
                ax.yaxis.label.set_size(axis_font_size)
                ax.title.set_fontsize(font_size)
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(axis_font_size)
                for text in ax.texts:
                    text.set_fontsize(font_size)
            canvas.draw_idle()

    def _iter_plot_canvases(self):
        """Yield the matplotlib canvases that should respond to font changes."""
        for name in ("canvas", "plot_canvas_a", "plot_canvas_b", "plot_canvas_c", "plot_canvas_d"):
            canvas = getattr(self, name, None)
            if canvas is not None:
                yield canvas

    def _iter_axes_for_canvas(self, canvas):
        """Yield all axes attached to a canvas (main + optional residual)."""
        main_ax = getattr(canvas, "axes", None)
        residual_ax = getattr(canvas, "residual_axes", None)
        if main_ax is not None:
            yield main_ax
        if residual_ax is not None and residual_ax is not main_ax:
            yield residual_ax

    def apply_symbol_size(self):
        """Apply the current symbol size to all existing plot markers."""
        marker_size = getattr(self, "symbol_size", 4)
        for canvas in self._iter_plot_canvases():
            changed = False
            for axes in self._iter_axes_for_canvas(canvas):
                for line in axes.lines:
                    marker = line.get_marker()
                    if marker not in (None, "", "None"):
                        line.set_markersize(marker_size)
                        changed = True
            if changed:
                canvas.draw_idle()

    def increase_plot_font_size(self):
        """Shortcut handler to increase plot font."""
        if self.plot_font_size < 48:
            self.plot_font_size += 1
            self.update_plot_font_size(self.plot_font_size)

    def decrease_plot_font_size(self):
        """Shortcut handler to decrease plot font."""
        if self.plot_font_size > 6:
            self.plot_font_size -= 1
            self.update_plot_font_size(self.plot_font_size)

