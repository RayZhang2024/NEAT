"""Fitting tab functionality."""

import gc
import glob
import csv
import os
import time
from html import escape

import matplotlib as mpl
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygon
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
    QDialog,
    QDialogButtonBox,
    QButtonGroup,
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
    NexusImageStackLoadWorker,
    RadenTiffStackLoadWorker,
    get_nexus_image_stack_info,
    get_raden_tiff_stack_info,
)
from ..dialogs import (
    AdjustmentsDialog,
    FitVisualizationDialog,
    MplCanvas,
    OpenBeamPlotDialog,
)


class CheckableBraggHeader(QHeaderView):
    """Horizontal table header with checkbox states for selected columns."""

    state_changed = pyqtSignal(int, bool)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._checkable_columns = {}
        self._checkbox_widgets = {}
        self.setSectionsClickable(True)
        self.setDefaultAlignment(Qt.AlignCenter)
        self.sectionResized.connect(self._update_checkbox_positions)
        self.sectionMoved.connect(self._update_checkbox_positions)
        self.geometriesChanged.connect(self._update_checkbox_positions)

    def set_checkable_column(self, column, checked=True):
        self._checkable_columns[column] = bool(checked)
        checkbox = self._checkbox_widgets.get(column)
        if checkbox is None:
            checkbox = QCheckBox(self.viewport())
            checkbox.setFocusPolicy(Qt.NoFocus)
            checkbox.setStyleSheet("QCheckBox::indicator { width: 10px; height: 10px; }")
            checkbox.setToolTip("Checked = fixed during fitting; unchecked = fitted.")
            checkbox.toggled.connect(
                lambda state, col=column: self._on_checkbox_toggled(col, state)
            )
            self._checkbox_widgets[column] = checkbox
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(checked))
        checkbox.blockSignals(False)
        checkbox.show()
        self._update_checkbox_positions()
        self.viewport().update()

    def set_column_checked(self, column, checked):
        if column not in self._checkable_columns:
            return
        self._checkable_columns[column] = bool(checked)
        checkbox = self._checkbox_widgets.get(column)
        if checkbox is not None:
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
            checkbox.blockSignals(False)
        self.viewport().update()

    def is_column_checked(self, column):
        checkbox = self._checkbox_widgets.get(column)
        if checkbox is not None:
            return checkbox.isChecked()
        return bool(self._checkable_columns.get(column, False))

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(max(size.height(), 30))
        return size

    def _on_checkbox_toggled(self, column, checked):
        self._checkable_columns[column] = bool(checked)
        self.state_changed.emit(column, bool(checked))
        self.viewport().update()

    def paintSection(self, painter, rect, logicalIndex):
        super().paintSection(painter, rect, logicalIndex)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_checkbox_positions)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_checkbox_positions)

    def _update_checkbox_positions(self, *args):
        for column, checkbox in self._checkbox_widgets.items():
            if self.isSectionHidden(column):
                checkbox.hide()
                continue
            section_x = self.sectionViewportPosition(column)
            section_width = self.sectionSize(column)
            label = str(self.model().headerData(column, self.orientation(), Qt.DisplayRole)).strip()
            label_width = self.fontMetrics().horizontalAdvance(label)
            indicator_size = 12
            gap = 4
            label_left = section_x + max(0, (section_width - label_width) // 2) + self._label_x_shift()
            checkbox_x = max(section_x + 3, label_left - indicator_size - gap)
            checkbox.setGeometry(
                checkbox_x,
                max(1, (self.height() - indicator_size) // 2),
                indicator_size,
                indicator_size,
            )
            checkbox.show()

    @staticmethod
    def _label_x_shift():
        return 8

    def mousePressEvent(self, event):
        column = self.logicalIndexAt(event.pos())
        if column in self._checkable_columns and event.button() == Qt.LeftButton:
            checkbox = self._checkbox_widgets.get(column)
            if checkbox is not None:
                checkbox.setChecked(not checkbox.isChecked())
            else:
                checked = not self._checkable_columns[column]
                self._checkable_columns[column] = checked
                self.state_changed.emit(column, checked)
                self.viewport().update()
            return
        super().mousePressEvent(event)


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
            "flight_path_source": getattr(self, "flight_path_source", "App setting"),
            "data_source": getattr(self, "fitting_data_source", "images"),
            "input_file": getattr(self, "current_fitting_input", ""),
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
        self.show_edge_line = getattr(self, "show_edge_line", True)
        self.fitting_data_source = getattr(self, "fitting_data_source", "images")
        self.fitting_plot_layout_mode = self._normalize_fitting_plot_layout_mode(
            getattr(self, "fitting_plot_layout_mode", "single")
        )
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

        self.image_slider = QSlider(Qt.Vertical)
        self.image_slider.setEnabled(False)  # Disabled until images are loaded
        self.image_slider.setInvertedAppearance(True)
        self.image_slider.setInvertedControls(True)
        self.image_slider.setToolTip("Select image in the loaded stack")
        self.image_slider.setFixedWidth(12)
        self.image_slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.image_slider.setStyleSheet(
            """
            QSlider::groove:vertical {
                width: 4px;
                background: #d9d9d9;
                border: 1px solid #cfcfcf;
                border-radius: 2px;
            }
            QSlider::handle:vertical {
                background: #1976d2;
                width: 10px;
                height: 6px;
                margin: 0 -3px;
                border-radius: 1px;
            }
            """
        )
        self.image_slider.valueChanged.connect(self.update_image)

        image_view_layout = QHBoxLayout()
        image_view_layout.setContentsMargins(0, 0, 0, 0)
        image_view_layout.setSpacing(4)
        image_view_layout.addWidget(self.canvas, 1)
        image_view_layout.addWidget(self.image_slider)
        upper_left_layout.addLayout(image_view_layout)
        upper_left_widget.setMinimumSize(200, 200)

        # Enable dragging of the yellow macro‑pixel ROI directly on the canvas
        self._dragging_roi = False
        self._roi_drag_offset = (0.0, 0.0)
        self._dragging_roi_mode = "move"
        self._roi_active_corner = None
        # Enable dragging of the orange batch ROI as well
        self._dragging_batch_roi = False
        self.batch_roi_visible = False
        self._batch_drag_offset = (0.0, 0.0)
        self._dragging_batch_mode = "move"
        self._batch_active_corner = None
        self._dragging_image_pan = False
        self._image_pan_start_xpixel = None
        self._image_pan_start_ypixel = None
        self._image_pan_start_xlim = None
        self._image_pan_start_ylim = None
        self.live_fit_enabled = False
        self.live_fit_mode = None
        self.live_fit_preview_enabled = getattr(self, "live_fit_preview_enabled", True)
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
        self.phase_dropdown = QComboBox()
        self.phase_placeholder = "Select Phase"
        self.phase_dropdown.addItem(self.phase_placeholder)  # Placeholder/default item
        self.phase_dropdown.setToolTip('Select a phase from the drop down list, or select "unknown_phase" if the material is unknown')
        self.phase_dropdown.addItems(sorted(getattr(self, "phase_data", PHASE_DATA).keys()))  # Dynamically add phase names
        self.phase_dropdown.currentIndexChanged.connect(self.phase_selection_changed)  # Connect to handler
        button_layout.addWidget(self.phase_dropdown)

        lower_left_layout.addLayout(button_layout)

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
        for roi_input in (self.xmin_input, self.xmax_input, self.ymin_input, self.ymax_input):
            roi_input.editingFinished.connect(self.apply_selected_area_from_inputs)

        selection_layout.addWidget(QLabel("X Min:"), 0, 0)
        selection_layout.addWidget(self.xmin_input, 0, 1)
        selection_layout.addWidget(QLabel("X Max:"), 0, 2)
        selection_layout.addWidget(self.xmax_input, 0, 3)
        selection_layout.addWidget(QLabel("Y Min:"), 0, 4)
        selection_layout.addWidget(self.ymin_input, 0, 5)
        selection_layout.addWidget(QLabel("Y Max:"), 0, 6)
        selection_layout.addWidget(self.ymax_input, 0, 7)

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

        self.fix_s = getattr(self, "fix_s", True)
        self.fix_t = getattr(self, "fix_t", True)
        self.fix_eta = getattr(self, "fix_eta", True)
        self.bragg_header = CheckableBraggHeader(Qt.Horizontal, self.bragg_table)
        self.bragg_table.setHorizontalHeader(self.bragg_header)
        self.bragg_table.setHorizontalHeaderLabels([
            "hkl", "d",
            "1 Min", "1 Max",
            "2 Min", "2 Max",
            "3 Min", "3 Max",
            "  s", "  t", "  eta"
        ])
        self.bragg_header.setSectionResizeMode(QHeaderView.Stretch)
        self.bragg_header.set_checkable_column(8, self.fix_s)
        self.bragg_header.set_checkable_column(9, self.fix_t)
        self.bragg_header.set_checkable_column(10, self.fix_eta)
        self.bragg_header.state_changed.connect(self.on_bragg_header_fix_state_changed)
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
            "Initial/fixed value for s, the edge broadening parameter. Header checkbox checked = fixed, unchecked = fitted.",
            "Initial/fixed value for t, the moderator decay parameter. Header checkbox checked = fixed, unchecked = fitted.",
            "Initial/fixed value for eta, the pseudo-Voigt mixing parameter. Header checkbox checked = fixed, unchecked = fitted; eta=0 is Gaussian-like and eta=1 is Lorentzian-like."
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
        fit_buttons_layout.setSpacing(4)
        self.fit_mode_button_group = QButtonGroup(self)
        self.fit_mode_button_group.setExclusive(True)
        self.individual_edges_mode_button = self.create_fitting_mode_button("edge", "Individual Edges")
        self.individual_edges_mode_button.setCheckable(True)
        self.individual_edges_mode_button.setChecked(True)
        self.individual_edges_mode_button.setToolTip("Fit each Bragg edge independently.")
        self.edge_pattern_mode_button = self.create_fitting_mode_button("edge_pattern", "Edge Pattern")
        self.edge_pattern_mode_button.setCheckable(True)
        self.edge_pattern_mode_button.setToolTip("Fit multiple edges together with shared lattice parameters.")
        self.fit_mode_button_group.addButton(self.individual_edges_mode_button, 0)
        self.fit_mode_button_group.addButton(self.edge_pattern_mode_button, 1)
        self.fit_mode_button_group.idClicked.connect(self.set_fitting_mode)
        self.fitting_mode = "individual"
        self.update_fitting_mode_button_styles()

        fit_buttons_layout.addWidget(self.individual_edges_mode_button)
        fit_buttons_layout.addWidget(self.edge_pattern_mode_button)
        fit_buttons_layout.addSpacing(12)

        self.fit_roi_button = self.create_fitting_icon_button("play", "Fit ROI")
        self.fit_roi_button.setToolTip("Fit the current yellow ROI using the selected fitting mode.")
        self.fit_roi_button.clicked.connect(self.run_selected_fit)
        fit_buttons_layout.addWidget(self.fit_roi_button)

        self.batch_map_button = self.create_fitting_icon_button("batch_play", "Batch Map")
        self.batch_map_button.setToolTip("Map the orange ROI using the selected fitting mode.")
        self.batch_map_button.clicked.connect(self.run_selected_batch_fit)
        fit_buttons_layout.addWidget(self.batch_map_button)

        self.batch_settings_button = self.create_fitting_icon_button("gear", "Set")
        self.batch_settings_button.setToolTip("Configure batch fitting parameters")
        self.batch_settings_button.clicked.connect(self.open_batch_settings_dialog)
        fit_buttons_layout.addWidget(self.batch_settings_button)
        self._set_mapping_controls_enabled(False)

        self.stop_batch_fit_button = self.create_fitting_icon_button("stop", "Stop")
        self.stop_batch_fit_button.setToolTip('Stop batch fitting')
        self.stop_batch_fit_button.clicked.connect(self.stop_batch_fit)
        fit_buttons_layout.addWidget(self.stop_batch_fit_button)

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
        plots_layout.setContentsMargins(2, 2, 2, 2)
        plots_layout.setSpacing(6)
        plots_layout.setColumnStretch(0, 1)
        plots_layout.setColumnStretch(1, 1)
        plots_layout.setRowStretch(0, 1)
        plots_layout.setRowStretch(1, 1)
        self.fit_plots_layout = plots_layout

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

        self._apply_fitting_plot_layout_mode()
        self._connect_fit_canvas_interactions()

        upper_right_layout.addLayout(plots_layout)
        upper_right_widget.setMinimumSize(200, 200)

        # Lower right widget for message box
        lower_right_widget = QWidget()
        lower_right_layout = QVBoxLayout(lower_right_widget)

        message_layout = QVBoxLayout()
        message_label = QLabel("Messages:")
        self.message_box = QTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setFont(QFont("Arial", getattr(self, "gui_font_size", 8)))
        self.message_box.setStyleSheet(
            "QTextEdit { font-family: Arial; font-size: %dpt; }"
            % int(getattr(self, "gui_font_size", 8))
        )
        message_layout.addWidget(message_label)
        message_layout.addWidget(self.message_box)

        lower_right_layout.addLayout(message_layout)
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

    def _normalize_fitting_plot_layout_mode(self, mode):
        """Return the canonical fitting plot layout mode."""
        mode_text = str(mode or "").strip().lower()
        if mode_text in ("quad", "four", "four_canvas", "4", "4-canvas"):
            return "quad"
        return "single"

    def is_single_fit_canvas_mode(self):
        return self._normalize_fitting_plot_layout_mode(
            getattr(self, "fitting_plot_layout_mode", "single")
        ) == "single"

    def set_fitting_plot_layout_mode(self, mode, refresh_plots=True):
        """Switch fitting result plots between one-canvas and four-canvas layouts."""
        self.fitting_plot_layout_mode = self._normalize_fitting_plot_layout_mode(mode)
        self._apply_fitting_plot_layout_mode()
        if hasattr(self, "sync_fitting_plot_layout_controls"):
            self.sync_fitting_plot_layout_controls()
        if refresh_plots and hasattr(self, "update_plots"):
            self.update_plots()

    def _final_fit_canvas(self):
        """Return the canvas used for final fit results in the active layout."""
        if self.is_single_fit_canvas_mode():
            return getattr(self, "plot_canvas_a", None)
        return getattr(self, "plot_canvas_d", None)

    def _apply_fitting_plot_layout_mode(self):
        """Apply widget visibility and grid placement for the active plot layout."""
        layout = getattr(self, "fit_plots_layout", None)
        canvases = [
            getattr(self, "plot_canvas_a", None),
            getattr(self, "plot_canvas_b", None),
            getattr(self, "plot_canvas_c", None),
            getattr(self, "plot_canvas_d", None),
        ]
        if layout is None or any(canvas is None for canvas in canvases):
            return

        for canvas in canvases:
            layout.removeWidget(canvas)

        for index in range(2):
            layout.setColumnMinimumWidth(index, 0)
            layout.setRowMinimumHeight(index, 0)

        if self.is_single_fit_canvas_mode():
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 0)
            layout.setRowStretch(0, 1)
            layout.setRowStretch(1, 0)
            layout.addWidget(self.plot_canvas_a, 0, 0)
            self.plot_canvas_a.show()
            for canvas in (self.plot_canvas_b, self.plot_canvas_c, self.plot_canvas_d):
                canvas.hide()
        else:
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 1)
            layout.setRowStretch(0, 1)
            layout.setRowStretch(1, 1)
            layout.addWidget(self.plot_canvas_a, 0, 0)
            layout.addWidget(self.plot_canvas_b, 0, 1)
            layout.addWidget(self.plot_canvas_c, 1, 0)
            layout.addWidget(self.plot_canvas_d, 1, 1)
            for canvas in canvases:
                canvas.show()

        self._clear_fitting_canvases()

    def create_fitting_icon_button(self, icon_name, accessible_name):
        button = QPushButton()
        button.setAccessibleName(accessible_name)
        button.setIcon(self.create_fitting_action_icon(icon_name))
        button.setIconSize(QSize(24, 18))
        button.setFixedSize(30, 21)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return button

    def create_fitting_mode_button(self, icon_name, accessible_name):
        button = QPushButton()
        button.setAccessibleName(accessible_name)
        button.setIcon(self.create_fitting_action_icon(icon_name))
        button.setIconSize(QSize(24, 18))
        button.setFixedSize(30, 21)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return button

    def create_fitting_action_icon(self, icon_name):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(45, 45, 45), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(45, 45, 45)))

        if icon_name == "edge":
            painter.setBrush(Qt.NoBrush)
            self.draw_independent_edge_mode_icon(painter)
        elif icon_name == "edge_pattern":
            painter.setBrush(Qt.NoBrush)
            self.draw_multi_edge_mode_icon(painter)
        elif icon_name == "play":
            painter.drawPolygon(QPolygon([QPoint(11, 8), QPoint(24, 16), QPoint(11, 24)]))
        elif icon_name == "batch_play":
            painter.drawPolygon(QPolygon([QPoint(7, 8), QPoint(18, 16), QPoint(7, 24)]))
            painter.drawPolygon(QPolygon([QPoint(16, 8), QPoint(27, 16), QPoint(16, 24)]))
        elif icon_name == "stop":
            painter.drawRect(10, 10, 12, 12)
        elif icon_name == "gear":
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(11, 11, 10, 10)
            painter.drawEllipse(14, 14, 4, 4)
            for start, end in (
                ((16, 5), (16, 9)),
                ((16, 23), (16, 27)),
                ((5, 16), (9, 16)),
                ((23, 16), (27, 16)),
                ((8, 8), (11, 11)),
                ((24, 8), (21, 11)),
                ((8, 24), (11, 21)),
                ((24, 24), (21, 21)),
            ):
                painter.drawLine(QPoint(*start), QPoint(*end))
        elif icon_name == "load":
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(6, 11, 20, 13)
            painter.drawLine(QPoint(6, 11), QPoint(12, 7))
            painter.drawLine(QPoint(12, 7), QPoint(18, 7))
            painter.drawLine(QPoint(18, 7), QPoint(21, 11))
        elif icon_name == "clear":
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(QPoint(10, 10), QPoint(22, 22))
            painter.drawLine(QPoint(22, 10), QPoint(10, 22))
        elif icon_name == "pick":
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(9, 9, 14, 14)
            painter.drawLine(QPoint(16, 6), QPoint(16, 12))
            painter.drawLine(QPoint(16, 20), QPoint(16, 26))
            painter.drawLine(QPoint(6, 16), QPoint(12, 16))
            painter.drawLine(QPoint(20, 16), QPoint(26, 16))
        elif icon_name == "export":
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(7, 18, 18, 6)
            painter.drawLine(QPoint(16, 7), QPoint(16, 17))
            painter.drawLine(QPoint(16, 7), QPoint(11, 12))
            painter.drawLine(QPoint(16, 7), QPoint(21, 12))
        elif icon_name == "import":
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(7, 18, 18, 6)
            painter.drawLine(QPoint(16, 7), QPoint(16, 17))
            painter.drawLine(QPoint(16, 17), QPoint(11, 12))
            painter.drawLine(QPoint(16, 17), QPoint(21, 12))

        painter.end()
        return QIcon(pixmap)

    def draw_edge_icon_path(self, painter, x, baseline_y):
        painter.drawLine(QPoint(x, baseline_y), QPoint(x + 5, baseline_y))
        painter.drawLine(QPoint(x + 5, baseline_y), QPoint(x + 8, baseline_y - 18))
        painter.drawLine(QPoint(x + 8, baseline_y - 18), QPoint(x + 16, baseline_y - 17))

    def draw_independent_edge_mode_icon(self, painter):
        pen = QPen(QColor(0, 0, 0))
        pen.setWidthF(2.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        path = QPainterPath()
        path.moveTo(3, 13)
        path.lineTo(13, 22)
        path.cubicTo(15, 24, 16, 23, 17, 19)
        path.cubicTo(18, 14, 18, 8, 21, 7)
        path.cubicTo(23, 6, 25, 8, 27, 9)
        path.lineTo(30, 12)
        painter.drawPath(path)

    def draw_multi_edge_mode_icon(self, painter):
        pen = QPen(QColor(0, 0, 0))
        pen.setWidthF(2.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        path = QPainterPath()
        path.moveTo(2, 14)
        path.lineTo(7, 20)
        path.cubicTo(8, 21, 9, 20, 9, 17)
        path.cubicTo(10, 12, 10, 9, 12, 8)
        path.cubicTo(14, 8, 15, 10, 16, 12)
        path.lineTo(21, 22)
        path.cubicTo(22, 24, 23, 23, 24, 18)
        path.cubicTo(24, 12, 25, 6, 27, 5)
        path.cubicTo(29, 5, 30, 7, 31, 8)
        painter.drawPath(path)

    def _set_fits_image_button_states(self, is_loading=False):
        has_images = bool(getattr(self, "images", []))
        load_button = getattr(self, "load_button", None)
        if load_button is not None:
            load_button.setEnabled(not is_loading)
        clear_button = getattr(self, "clear_load_button", None)
        if clear_button is not None:
            clear_button.setEnabled((not is_loading) and has_images)
        self._set_mapping_controls_enabled(has_images and not is_loading)

    def _set_mapping_controls_enabled(self, enabled):
        """Enable image-stack mapping controls only when spatial image data exists."""
        for button_name in ("batch_map_button", "batch_settings_button"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(bool(enabled))

    def update_fitting_mode_button_styles(self):
        selected_style = (
            "QPushButton { background-color: #b9ddf6; border: 1px solid #2b7dbc; "
            "border-radius: 2px; padding: 0px; }"
            "QPushButton:hover { background-color: #c8e6fb; }"
        )
        normal_style = (
            "QPushButton { background-color: #f7f7f7; border: 1px solid #b8b8b8; "
            "border-radius: 2px; padding: 0px; }"
            "QPushButton:hover { background-color: #eeeeee; }"
        )
        for button in (
            getattr(self, "individual_edges_mode_button", None),
            getattr(self, "edge_pattern_mode_button", None),
        ):
            if button is not None:
                button.setStyleSheet(selected_style if button.isChecked() else normal_style)

    def set_fitting_mode(self, mode_id):
        self.fitting_mode = "pattern" if mode_id == 1 else "individual"
        self.update_fitting_mode_button_styles()
        if hasattr(self, "fit_roi_button"):
            self.fit_roi_button.setToolTip(
                "Fit the current yellow ROI using edge pattern fitting."
                if self.fitting_mode == "pattern"
                else "Fit the current yellow ROI using individual edge fitting."
            )
        if hasattr(self, "batch_map_button"):
            self.batch_map_button.setToolTip(
                "Map the orange ROI using edge pattern fitting."
                if self.fitting_mode == "pattern"
                else "Map the orange ROI using individual edge fitting."
            )

    def run_selected_fit(self):
        if getattr(self, "fitting_mode", "individual") == "pattern":
            fitted = self.fit_full_pattern()
            mode = "pattern"
        else:
            fitted = self.fit_all_regions()
            mode = "individual"
        if fitted:
            self.live_fit_enabled = True
            self.live_fit_mode = mode

    def run_selected_batch_fit(self):
        if getattr(self, "fitting_data_source", "images") == "profile":
            self.message_box.append("Mapping is not available for imported intensity profiles.")
            return
        if getattr(self, "fitting_mode", "individual") == "pattern":
            self.batch_fit()
        else:
            self.batch_fit_edges()

    def run_live_fit_preview(self):
        if not getattr(self, "live_fit_preview_enabled", True):
            return
        if not getattr(self, "live_fit_enabled", False):
            return
        if not getattr(self, "selected_area", None):
            return
        mode = getattr(self, "live_fit_mode", None)
        try:
            if mode == "individual":
                fit_results = self.fit_all_regions(
                    emit_messages=False,
                    update_table=True,
                    only_update_unfixed=True,
                )
                self.annotate_live_individual_fit(fit_results)
            elif mode == "pattern":
                self.run_live_pattern_fit_preview()
        except Exception:
            # Live preview should never interrupt ROI movement.
            return

    def run_live_pattern_fit_preview(self):
        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()
        result_dict, error_msg = self.fit_full_pattern_core(
            fix_s=fix_s,
            fix_t=fix_t,
            fix_eta=fix_eta,
            apply_lattice_update=False,
        )
        if error_msg or not result_dict:
            return

        self._update_pattern_fit_parameter_cells(result_dict, only_update_unfixed=True)
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
        except ValueError:
            min_wavelength = self.wavelengths[0]
            max_wavelength = self.wavelengths[-1]

        ax_main_a, _ = self._initialize_fit_canvas(self.plot_canvas_a)
        ax_main_a.plot(
            self.selected_region_x,
            self.selected_region_y,
            "o",
            markersize=getattr(self, "symbol_size", 4),
            markerfacecolor="blue",
            markeredgecolor="none",
        )
        x_fit = result_dict["x_data"]
        y_fit = result_dict["y_data"]
        self._plot_line(
            ax_main_a,
            x_fit,
            y_fit,
            split_on_gaps=True,
            color="red",
            linestyle="-",
        )
        if self.show_edge_lines_enabled():
            for hkl, x_hkl in self.get_edges_in_range(min_wavelength, max_wavelength):
                ax_main_a.axvline(x=x_hkl, color="red", linestyle="--")
                y_max = ax_main_a.get_ylim()[1]
                ax_main_a.text(
                    x_hkl * 1.02,
                    y_max * 0.95,
                    f"hkl{hkl}",
                    rotation=90,
                    verticalalignment="top",
                    color="red",
                    fontsize=getattr(self, "plot_font_size", 12),
                )
        if np.isfinite(min_wavelength) and np.isfinite(max_wavelength) and min_wavelength < max_wavelength:
            ax_main_a.set_xlim(min_wavelength, max_wavelength)
        self.annotate_live_pattern_fit(ax_main_a, result_dict, x_fit, y_fit)
        x_obs = result_dict.get("x_exp_sorted", x_fit)
        y_obs = result_dict.get("y_exp_sorted")
        if y_obs is None:
            y_obs = np.interp(x_fit, self.selected_region_x, self.selected_region_y)
        self._plot_residual_line(self.plot_canvas_a, x_obs, y_obs, y_fit, split_on_gaps=True)
        self.plot_canvas_a.draw()

    def annotate_live_pattern_fit(self, ax, result_dict, x_fit=None, y_fit=None):
        structure_map = {
            "cubic": ["a"],
            "fcc": ["a"],
            "bcc": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"],
        }
        names = structure_map.get(result_dict.get("structure_type"), [])
        lattice = result_dict.get("lattice_params", {})
        parts = []
        for name in names:
            value = lattice.get(name)
            if value is not None and np.isfinite(value):
                parts.append(f"{name}={value:.6f} A")
        if parts:
            self._add_live_fit_annotation(
                ax,
                "\n".join(parts),
                self.selected_region_x,
                self.selected_region_y,
                fit_x=x_fit,
                fit_y=y_fit,
            )

    def annotate_live_individual_fit(self, fit_results):
        if not fit_results:
            return
        lines = []
        x_values = []
        y_values = []
        fit_x_values = []
        fit_y_values = []
        for result in fit_results:
            try:
                hkl = result["hkl"]
                d_fit = float(result["fit_params"][0])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if np.isfinite(d_fit):
                lines.append(f"d{hkl}={d_fit:.6f} A")
            x = np.asarray(result.get("x", []), dtype=float)
            y = np.asarray(result.get("y", []), dtype=float)
            y_fit = np.asarray(result.get("fit", []), dtype=float)
            if x.size and y.size:
                n = min(x.size, y.size)
                x_values.append(x[:n])
                y_values.append(y[:n])
            if x.size and y_fit.size:
                n = min(x.size, y_fit.size)
                fit_x_values.append(x[:n])
                fit_y_values.append(y_fit[:n])
        if lines:
            final_canvas = self._final_fit_canvas()
            ax_main, _ = self._ensure_fit_canvas_axes(final_canvas)
            data_x = np.concatenate(x_values) if x_values else None
            data_y = np.concatenate(y_values) if y_values else None
            fit_x = np.concatenate(fit_x_values) if fit_x_values else None
            fit_y = np.concatenate(fit_y_values) if fit_y_values else None
            self._add_live_fit_annotation(
                ax_main,
                "\n".join(lines),
                data_x,
                data_y,
                fit_x=fit_x,
                fit_y=fit_y,
            )
            final_canvas.draw()

    def set_fit_action_buttons_enabled(self, enabled):
        fit_button = getattr(self, "fit_roi_button", None)
        if fit_button is not None:
            fit_button.setEnabled(enabled)
        self._set_mapping_controls_enabled(
            bool(enabled)
            and getattr(self, "fitting_data_source", "images") != "profile"
            and bool(getattr(self, "images", []))
        )

    def on_bragg_header_fix_state_changed(self, column, checked):
        if column == 8:
            self.fix_s = bool(checked)
        elif column == 9:
            self.fix_t = bool(checked)
        elif column == 10:
            self.fix_eta = bool(checked)
        if hasattr(self, "save_user_settings"):
            self.save_user_settings()

    def set_fix_parameter_state(self, name, checked):
        attr_map = {"s": ("fix_s", 8), "t": ("fix_t", 9), "eta": ("fix_eta", 10)}
        attr, column = attr_map[name]
        setattr(self, attr, bool(checked))
        header = getattr(self, "bragg_header", None)
        if header is not None:
            header.set_column_checked(column, checked)

    def fix_s_enabled(self):
        header = getattr(self, "bragg_header", None)
        return header.is_column_checked(8) if header is not None else bool(getattr(self, "fix_s", True))

    def fix_t_enabled(self):
        header = getattr(self, "bragg_header", None)
        return header.is_column_checked(9) if header is not None else bool(getattr(self, "fix_t", True))

    def fix_eta_enabled(self):
        header = getattr(self, "bragg_header", None)
        return header.is_column_checked(10) if header is not None else bool(getattr(self, "fix_eta", True))

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
        if getattr(self, "fitting_data_source", "images") == "profile":
            self.message_box.append("Mapping is not available for imported intensity profiles.")
            return
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
            self.message_box.append("Working directory is not set. Please load images first.")
            return
        # Get the state of the fix flags from the Bragg table header.
        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()
        interpolation_enabled = self.interpolation_checkbox.isChecked()

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
            interpolation_enabled=interpolation_enabled,
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
            self._append_fit_message(f"[Batch edges] Completed\n  Results saved to {filename}")
        else:
            self._append_fit_message("[Batch edges] Completed with errors")

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
        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()

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
            self._append_fit_message(f"[Pattern fit] Failed\n  {error_msg}")
            self.set_fit_action_buttons_enabled(True)
            return False

        # 4) If we got results, display success
        # Successful pattern fit output is appended once after plotting.

        # 5) Display lattice parameters + uncertainties
        structure_type = result_dict['structure_type']
        lattice_params = result_dict['lattice_params']
        lattice_unc = result_dict['lattice_uncertainties']

        # Detailed pattern output is summarized after plotting.
        structure_map = {
            "cubic": ["a"],
            "fcc": ["a"],
            "bcc": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"]
        }
        param_names = []

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
        self._update_pattern_fit_parameter_cells(result_dict)

        for row, edge in enumerate([]):
            hkl = edge['hkl']
            s_val = result_dict['fitted_s'].get(hkl, np.nan)
            t_val = result_dict['fitted_t'].get(hkl, np.nan)
            eta_val = result_dict['fitted_eta'].get(hkl, np.nan)
            s_err = s_unc.get(hkl, np.nan)
            t_err = t_unc.get(hkl, np.nan)
            eta_err = eta_unc.get(hkl, np.nan)

            # Also display in the message box
            self.message_box.append(
                f"Bragg Edge {row+1} (hkl{hkl}): "
                f"s = {s_val:.6f} ± {s_err:.6f}, "
                f"t = {t_val:.6f} ± {t_err:.6f},"
                f"eta = {eta_val:.6f} ± {eta_err:.3f}"
            )

        # 4) If we got results, display success
        # Summary is appended after plots and table updates are complete.


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
            if self.show_edge_lines_enabled():
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
        self.set_fit_action_buttons_enabled(True)

        self._append_pattern_fit_summary(result_dict, min_wavelength, max_wavelength)
        return True

    def _update_pattern_fit_parameter_cells(self, result_dict, only_update_unfixed=False):
        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()
        for row, edge in enumerate(result_dict.get("bragg_edges", [])):
            if row >= self.bragg_table.rowCount():
                continue
            hkl = edge["hkl"]
            if not only_update_unfixed or not fix_s:
                self.bragg_table.item(row, 8).setText(f"{result_dict['fitted_s'].get(hkl, np.nan):.4f}")
            if not only_update_unfixed or not fix_t:
                self.bragg_table.item(row, 9).setText(f"{result_dict['fitted_t'].get(hkl, np.nan):.4f}")
            if not only_update_unfixed or not fix_eta:
                self.bragg_table.item(row, 10).setText(f"{result_dict['fitted_eta'].get(hkl, np.nan):.3f}")

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
            self.fitting_data_source = "images"
            self.flight_path_source = "App setting"
            self.current_fitting_input = folder_path
            self.tof_axis_centers_us = None
            self.wavelength_depends_on_flight_path = False
            self.nexus_axis_centers = None
            self.nexus_axis_uses_flight_path = False
            self.wavelength_flight_path = None
            # Sort the suffixes to ensure images are in correct order
            sorted_suffixes = sorted(run_dict.keys())
            # Clear existing images if any
            self.images = []
            for suffix in sorted_suffixes:
                self.images.append(run_dict[suffix])
            run_number = len(run_dict)
            self.message_box.append(f"Successfully loaded {run_number} images from the selected folder.")

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
            self.apply_selected_area_from_inputs()
            self._set_fits_image_button_states(is_loading=False)
        else:
            self.message_box.append("No valid images were loaded from the selected folder.")
            self._set_fits_image_button_states(is_loading=False)

    def change_flight_path(self):
        """
        Opens a dialog to allow the user to change the flight path value.
        Then updates the wavelength calculation when the loaded data is flight-path dependent.
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
            if self._apply_instrument_settings_to_loaded_data():
                self.message_box.append("Loaded wavelength data updated with the new flight path.")
            else:
                self.message_box.append("No flight-path-dependent loaded data to update.")
        else:
            self.message_box.append("Flight path change cancelled.")

    def _update_wavelength_bounds_from_current_array(self):
        if self.wavelengths is None or len(self.wavelengths) == 0:
            return
        self.start_wavelength = float(self.wavelengths[0])
        self.end_wavelength = float(self.wavelengths[-1])
        min_input = getattr(self, "min_wavelength_input", None)
        max_input = getattr(self, "max_wavelength_input", None)
        if min_input is not None:
            min_input.setText(f"{self.start_wavelength:.6g}")
        if max_input is not None:
            max_input.setText(f"{self.end_wavelength:.6g}")

    def _recalculate_wavelengths_from_tof_axis(self, source_label=None, show_message=True, axis_label="TOF"):
        centers = getattr(self, "tof_axis_centers_us", None)
        if centers is None:
            centers = getattr(self, "nexus_axis_centers", None)
        depends_on_path = (
            getattr(self, "wavelength_depends_on_flight_path", False)
            or getattr(self, "nexus_axis_uses_flight_path", False)
        )
        if centers is None or not depends_on_path:
            return False
        flight_path = float(getattr(self, "flight_path", 0.0) or 0.0)
        if not np.isfinite(flight_path) or flight_path <= 0:
            self.message_box.append("Flight path must be > 0 to convert TOF to wavelength.")
            return False

        centers = np.asarray(centers, dtype=float)
        adjusted_centers = centers + float(getattr(self, "delay", 0.0) or 0.0) * 1000.0
        self.wavelengths = (adjusted_centers * 3.956) / flight_path / 1000.0
        if len(self.wavelengths) != len(getattr(self, "images", [])):
            self.message_box.append(
                f"Warning: Number of wavelengths ({len(self.wavelengths)}) "
                f"does not match number of images ({len(getattr(self, 'images', []))})."
            )
        self.wavelength_flight_path = flight_path
        if source_label:
            self.flight_path_source = source_label
        self._update_wavelength_bounds_from_current_array()
        if show_message:
            self.message_box.append(
                f"Updated {axis_label} wavelengths: {self.start_wavelength:.6f} / {self.end_wavelength:.6f} "
                f"A using flight path {flight_path:.6f} m."
            )
        return True

    def _recalculate_nexus_wavelengths_from_axis(self, source_label=None, show_message=True):
        return self._recalculate_wavelengths_from_tof_axis(
            source_label=source_label,
            show_message=show_message,
            axis_label="NeXus",
        )

    def _apply_instrument_settings_to_loaded_data(self):
        if getattr(self, "fitting_data_source", "images") == "profile":
            return False
        if getattr(self, "manual_wavelength_mode", False):
            self.update_wavelengths()
            self.update_plots()
            return True
        if self.tof_array is not None and len(self.tof_array) > 0:
            self.flight_path_source = "App setting"
            self.update_wavelengths()
            self.update_plots()
            return True
        if self._recalculate_wavelengths_from_tof_axis(source_label="App setting"):
            self.update_plots()
            return True
        return False

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

            self.wavelength_flight_path = float(self.flight_path)
            self._update_wavelength_bounds_from_current_array()
            self.message_box.append(
                f"Updated start/end wavelengths: {self.start_wavelength:.6f} / {self.end_wavelength:.6f}"
            )
        elif self._recalculate_wavelengths_from_tof_axis():
            return
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
                "Manual mode enabled. Set anchors (wavelength or ToF) in Manual Spectra Setting."
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
            self.message_box.append("Provide at least two anchors in Manual Spectra Setting to compute manual wavelengths.")
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
        """Create the modal progress dialog for image loading if needed."""
        if getattr(self, "_fits_progress_dialog", None) is None:
            self._fits_progress_dialog = QProgressDialog(
                "Loading images...", "Cancel", 0, 100, self
            )
            self._fits_progress_dialog.setWindowTitle("Loading Images")
            self._fits_progress_dialog.setAutoClose(False)
            self._fits_progress_dialog.setAutoReset(False)
            self._fits_progress_dialog.canceled.connect(self._cancel_fits_loading)
        return self._fits_progress_dialog

    def _hide_load_progress_dialog(self):
        """Hide the image-loading progress dialog without treating completion as cancellation."""
        dialog = getattr(self, "_fits_progress_dialog", None)
        if dialog is None:
            return
        previous_blocked = dialog.blockSignals(True)
        dialog.hide()
        dialog.blockSignals(previous_blocked)

    def _cancel_fits_loading(self):
        """Stop the worker if the user cancels loading."""
        worker = getattr(self, "fits_image_load_worker", None)
        if worker is None or not worker.isRunning():
            self._hide_load_progress_dialog()
            return

        worker.requestInterruption()
        worker.stop()
        worker.wait(3000)
        if worker.isRunning():
            self.message_box.append("Stopping image loader...")
            return

        self.fits_image_load_worker = None
        self.images = []
        self.image_slider.setEnabled(False)
        self._set_fits_image_button_states(is_loading=False)
        self.display_image()
        self.message_box.append("Image loading cancelled.")
        self._hide_load_progress_dialog()

    def update_fits_load_progress(self, value):
        """
        Update the image loading progress dialog.
        """
        dialog = self._ensure_load_progress_dialog()
        dialog.setValue(value)
        if value >= 100:
            self._hide_load_progress_dialog()

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
        self._hide_load_progress_dialog()
        self._set_fits_image_button_states(is_loading=False)

    def clear_loaded_fits_images(self):
        worker = getattr(self, "fits_image_load_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.stop()
            worker.wait(3000)
        self.fits_image_load_worker = None

        if getattr(self, "_fits_progress_dialog", None) is not None:
            self._fits_progress_dialog.hide()

        self.images = []
        self.fitting_data_source = "images"
        self.flight_path_source = "App setting"
        self.current_fitting_input = ""
        self.tof_axis_centers_us = None
        self.wavelength_depends_on_flight_path = False
        self.nexus_axis_centers = None
        self.nexus_axis_uses_flight_path = False
        self.wavelength_flight_path = None
        self.intensities = np.array([])
        self.tof_array = None
        self.wavelengths = np.array([])
        self.current_image_index = 0
        self.selected_area = None
        self.current_batch_box = None
        self.batch_box_patch = None
        self.batch_roi_visible = False
        self.auto_vmin = None
        self.auto_vmax = None
        self.current_vmin = None
        self.current_vmax = None
        self.min_slider_value = 0
        self.max_slider_value = 1000
        self.current_r1_min = None
        self.current_r1_max = None
        self.current_r2_min = None
        self.current_r2_max = None
        self.current_r3_min = None
        self.current_r3_max = None
        self.live_fit_enabled = False
        self.live_fit_mode = None
        self.manual_wavelength_mode = False
        self.folder_path = ""
        self.work_directory = ""
        self._bragg_table_ready = False

        if hasattr(self, "_reset_current_bragg_edge_state"):
            self._reset_current_bragg_edge_state()

        self.image_slider.blockSignals(True)
        self.image_slider.setEnabled(False)
        self.image_slider.setRange(0, 0)
        self.image_slider.setValue(0)
        self.image_slider.blockSignals(False)

        self.bragg_table.setRowCount(0)
        self.message_box.clear()
        self._clear_fitting_canvases()
        self._clear_image_canvas()
        self._set_fits_image_button_states(is_loading=False)

    def _clear_image_canvas(self):
        self.canvas.axes.clear()
        self.canvas.axes.set_title("")
        self.canvas.draw_idle()

    def _clear_fitting_canvases(self):
        for title, canvas in (
            ("", getattr(self, "plot_canvas_a", None)),
            ("Region 1", getattr(self, "plot_canvas_b", None)),
            ("Region 2", getattr(self, "plot_canvas_c", None)),
            ("Region 3", getattr(self, "plot_canvas_d", None)),
        ):
            if canvas is None:
                continue
            self._initialize_fit_canvas(canvas, title=title or None)
            canvas.draw_idle()

    @staticmethod
    def _read_intensity_profile_file(file_name):
        """Read wavelength/intensity columns from CSV, TXT, XLSX, or NeXus profile data."""
        ext = os.path.splitext(file_name)[1].lower()
        if ext in (".nxs", ".h5", ".hdf5"):
            return FittingMixin._read_nexus_intensity_profile_file(file_name)
        if ext == ".xlsx":
            frame = pd.read_excel(file_name, header=None)
        elif ext == ".xls":
            raise ValueError("Legacy .xls files are not supported. Save the profile as .xlsx, .csv, or .txt.")
        else:
            read_attempts = (
                {"header": None, "comment": "#", "sep": None, "engine": "python"},
                {"header": None, "comment": "#", "sep": r"[\s,;\t]+", "engine": "python"},
            )
            last_error = None
            frame = None
            for kwargs in read_attempts:
                try:
                    candidate = pd.read_csv(file_name, **kwargs)
                    if candidate.shape[1] >= 2:
                        frame = candidate
                        break
                    frame = candidate
                except Exception as exc:
                    last_error = exc
            if frame is None:
                raise ValueError(f"Could not read profile file: {last_error}")
            if frame.shape[1] < 2:
                raise ValueError("Profile file must contain at least two columns.")

        if frame.shape[1] < 2:
            raise ValueError("Profile file must contain at least two columns.")

        data = frame.iloc[:, :2].copy()
        data.columns = ["wavelength", "intensity"]
        data["wavelength"] = pd.to_numeric(data["wavelength"], errors="coerce")
        data["intensity"] = pd.to_numeric(data["intensity"], errors="coerce")
        data = data.dropna(subset=["wavelength", "intensity"])
        data = data[np.isfinite(data["wavelength"]) & np.isfinite(data["intensity"])]
        if len(data) < 3:
            raise ValueError("Profile file must contain at least three numeric wavelength/intensity rows.")

        data = data.sort_values("wavelength")
        return (
            data["wavelength"].to_numpy(dtype=float),
            data["intensity"].to_numpy(dtype=float),
        )

    @staticmethod
    def _read_nexus_intensity_profile_file(file_name):
        """Read a 1D wavelength/intensity profile from a NeXus/HDF5 file."""
        candidates = []
        with h5py.File(file_name, "r") as handle:
            def collect(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if len(obj.shape) != 1 or obj.shape[0] < 3:
                    return
                if not np.issubdtype(obj.dtype, np.number):
                    return
                label_parts = [name.lower(), os.path.basename(name).lower()]
                for attr_name in ("name", "long_name", "axis", "units"):
                    attr_val = obj.attrs.get(attr_name)
                    if attr_val is None:
                        continue
                    if isinstance(attr_val, bytes):
                        attr_val = attr_val.decode(errors="ignore")
                    elif isinstance(attr_val, np.ndarray):
                        attr_val = " ".join(
                            v.decode(errors="ignore") if isinstance(v, bytes) else str(v)
                            for v in attr_val.ravel()
                        )
                    label_parts.append(str(attr_val).lower())
                candidates.append(
                    {
                        "path": name,
                        "label": " ".join(label_parts),
                        "data": np.asarray(obj[()], dtype=float),
                    }
                )

            handle.visititems(collect)

        if len(candidates) < 2:
            raise ValueError("NeXus file must contain at least two numeric 1D datasets.")

        def score_wavelength(candidate):
            label = candidate["label"]
            score = 0
            for token in ("wavelength", "lambda", "dspacing", "xso", "xs0", "x axis", "axis"):
                if token in label:
                    score += 5
            if os.path.basename(candidate["path"]).lower() in ("x", "axis", "wavelength", "lambda", "column_1"):
                score += 4
            arr = candidate["data"]
            if np.all(np.diff(arr) > 0):
                score += 2
            return score

        def score_intensity(candidate):
            label = candidate["label"]
            score = 0
            for token in ("intensity", "transmission", "signal", "counts", "yso", "ys0", "y axis"):
                if token in label:
                    score += 5
            if os.path.basename(candidate["path"]).lower() in ("y", "data", "signal", "intensity", "column_2"):
                score += 4
            return score

        x_candidates = sorted(candidates, key=score_wavelength, reverse=True)
        for x_candidate in x_candidates:
            x_data = x_candidate["data"]
            y_candidates = [
                candidate for candidate in candidates
                if candidate["path"] != x_candidate["path"] and candidate["data"].shape == x_data.shape
            ]
            if not y_candidates:
                continue
            y_candidate = max(y_candidates, key=score_intensity)
            if score_wavelength(x_candidate) <= 0 and score_intensity(y_candidate) <= 0:
                continue
            return FittingMixin._clean_profile_arrays(x_data, y_candidate["data"])

        raise ValueError("Could not identify matching wavelength and intensity datasets in the NeXus file.")

    @staticmethod
    def _clean_profile_arrays(wavelengths, intensities):
        wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
        intensities = np.asarray(intensities, dtype=float).reshape(-1)
        if wavelengths.size != intensities.size:
            raise ValueError("Wavelength and intensity arrays have different lengths.")
        finite = np.isfinite(wavelengths) & np.isfinite(intensities)
        wavelengths = wavelengths[finite]
        intensities = intensities[finite]
        if wavelengths.size < 3:
            raise ValueError("Profile must contain at least three finite wavelength/intensity rows.")
        order = np.argsort(wavelengths)
        return wavelengths[order], intensities[order]

    def _clear_loaded_images_for_profile_import(self):
        """Release image-stack state before using an imported intensity profile."""
        worker = getattr(self, "fits_image_load_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.stop()
            worker.wait(3000)
        self.fits_image_load_worker = None

        if getattr(self, "_fits_progress_dialog", None) is not None:
            self._fits_progress_dialog.hide()

        self.images = []
        self.tof_axis_centers_us = None
        self.wavelength_depends_on_flight_path = False
        self.nexus_axis_centers = None
        self.nexus_axis_uses_flight_path = False
        self.wavelength_flight_path = None
        self.tof_array = None
        self.current_image_index = 0
        self.selected_area = None
        self.current_batch_box = None
        self.batch_box_patch = None
        self.batch_roi_visible = False
        self.auto_vmin = None
        self.auto_vmax = None
        self.current_vmin = None
        self.current_vmax = None
        self.manual_wavelength_mode = False

        slider = getattr(self, "image_slider", None)
        if slider is not None:
            slider.blockSignals(True)
            slider.setEnabled(False)
            slider.setRange(0, 0)
            slider.setValue(0)
            slider.blockSignals(False)

        self._set_fits_image_button_states(is_loading=False)
        self._clear_image_canvas()
        gc.collect()

    def import_intensity_profile(self):
        """Import a wavelength/intensity line profile and use it as the initial ROI spectrum."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Import Intensity Profile",
            "",
            "Profile Files (*.csv *.txt *.xlsx *.nxs);;CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx);;NeXus Files (*.nxs);;All Files (*)",
            options=options,
        )
        if not file_name:
            return

        try:
            wavelengths, intensities = self._read_intensity_profile_file(file_name)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", f"Could not import intensity profile:\n{exc}")
            return

        self._clear_loaded_images_for_profile_import()
        self.fitting_data_source = "profile"
        self.folder_path = os.path.dirname(file_name)
        self.work_directory = os.path.dirname(file_name)
        self.current_fitting_input = file_name
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.start_wavelength = float(wavelengths[0])
        self.end_wavelength = float(wavelengths[-1])
        self.selected_region_x = wavelengths
        self.selected_region_y = intensities
        self.params_region1 = None
        self.params_region2 = None
        self.params_region3 = {}
        self.live_fit_enabled = False
        self.live_fit_mode = None

        self.min_wavelength_input.setText(f"{self.start_wavelength:.6g}")
        self.max_wavelength_input.setText(f"{self.end_wavelength:.6g}")
        self._bragg_table_ready = True
        if hasattr(self, "_reset_current_bragg_edge_state"):
            self._reset_current_bragg_edge_state()
        self.update_bragg_edge_table()
        self.update_plots()
        self._set_mapping_controls_enabled(False)
        self.message_box.append(
            f"Imported intensity profile from {os.path.basename(file_name)} "
            f"({len(wavelengths)} points). Mapping is disabled for profile data."
        )

    def open_batch_settings_dialog(self):
        """Dialog to edit batch fitting parameters and ROI."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Mapping Setting")
        dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, True)
        dialog_layout = QVBoxLayout(dialog)
        label_width = 95
        field_width = 160

        def add_aligned_row(form, text, widget):
            label = QLabel(text)
            label.setFixedWidth(label_width)
            widget.setFixedWidth(field_width)
            form.addRow(label, widget)

        width_edit = QLineEdit(self.box_width_input.text())
        height_edit = QLineEdit(self.box_height_input.text())
        step_x_edit = QLineEdit(self.step_x_input.text())
        step_y_edit = QLineEdit(self.step_y_input.text())
        help_text = (
            "Macro pixel size sets the mapping box width and height in pixels.\n"
            "Skipping step controls how far the mapping box moves between fits. "
            "Smaller steps increase overlap and density. Interpolation fills skipped positions.\n"
            "Mapping area defines the half-open pixel bounds x[min, max) and y[min, max) "
            "used for the orange mapping ROI."
        )
        dialog.setWhatsThis(help_text)

        macro_group = QGroupBox("Macro pixel size")
        macro_group.setWhatsThis(
            "Set the mapping box size in pixels. This is the ROI size fitted at each mapping position."
        )
        macro_layout = QFormLayout(macro_group)
        macro_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        add_aligned_row(macro_layout, "Width:", width_edit)
        add_aligned_row(macro_layout, "Height:", height_edit)
        dialog_layout.addWidget(macro_group)

        skipping_group = QGroupBox("Skipping step")
        skipping_group.setWhatsThis(
            "Set how far the mapping box moves between fits. Interpolation can fill skipped positions in the exported map."
        )
        skipping_layout = QFormLayout(skipping_group)
        skipping_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        add_aligned_row(skipping_layout, "Step X:", step_x_edit)
        add_aligned_row(skipping_layout, "Step Y:", step_y_edit)
        interpolation_checkbox = QCheckBox("Enable interpolation")
        interpolation_checkbox.setChecked(self.interpolation_checkbox.isChecked())
        interpolation_checkbox.setWhatsThis(
            "Fill output map values between sampled positions when the step is larger than one pixel."
        )
        interpolation_spacer = QLabel("")
        interpolation_spacer.setFixedWidth(label_width)
        skipping_layout.addRow(interpolation_spacer, interpolation_checkbox)
        dialog_layout.addWidget(skipping_group)

        mapping_group = QGroupBox("Mapping area")
        mapping_group.setWhatsThis(
            "Set the orange mapping ROI using half-open bounds. Example: x[10, 12), y[20, 23) includes x pixels 10-11 and y pixels 20-22."
        )
        roi_layout = QFormLayout(mapping_group)
        roi_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
            roi_edit.setWhatsThis(batch_roi_tooltip)
        for edit in (width_edit, height_edit, step_x_edit, step_y_edit):
            edit.setWhatsThis(help_text)
        add_aligned_row(roi_layout, "X Min:", min_x_edit)
        add_aligned_row(roi_layout, "X Max:", max_x_edit)
        add_aligned_row(roi_layout, "Y Min:", min_y_edit)
        add_aligned_row(roi_layout, "Y Max:", max_y_edit)
        dialog_layout.addWidget(mapping_group)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Apply")
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
            self._apply_mapping_settings_from_inputs()

            # Do NOT overwrite the current picked ROI (selected_area); batch settings
            # should not change the spectra used for single-pixel fits.

    def _apply_mapping_settings_from_inputs(self, redraw=True, persist=True):
        """Apply current mapping input fields to the active mapping ROI state."""
        try:
            int(self.box_width_input.text())
            int(self.box_height_input.text())
            int(self.step_x_input.text())
            int(self.step_y_input.text())
            int(self.min_x_input.text())
            int(self.max_x_input.text())
            int(self.min_y_input.text())
            int(self.max_y_input.text())
        except ValueError:
            return False

        self.batch_roi_visible = True
        if redraw:
            self.display_image()
        if persist and hasattr(self, "save_user_settings"):
            self.save_user_settings()
        return True

    def open_general_settings_dialog(self):
        """Dialog for fitting-related flight path and time delay settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Instrument Setting")
        layout = QVBoxLayout(dialog)

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

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            self.flight_path = flight_spin.value()
            self.delay = delay_spin.value()
            self.message_box.append(f"Flight path set to {self.flight_path:.3f} m.")
            self.message_box.append(f"Time delay set to {self.delay:.4f} ms.")
            if self._apply_instrument_settings_to_loaded_data():
                self.message_box.append("Loaded wavelength data updated with the current instrument settings.")
            else:
                self.message_box.append("No flight-path-dependent loaded data to update.")
            self.save_user_settings()
        return

    def open_manual_spectra_settings_dialog(self):
        """Dialog for manual wavelength/ToF anchors and loading saved fitting metadata."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Spectra Setting")
        layout = QVBoxLayout(dialog)

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

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
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
            self.save_user_settings()

    def load_fitting_configuration_from_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select fitting configuration CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_name:
            return

        metadata = {}
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                first = next(reader, None)
                if not first or not first[0].strip().startswith("Metadata Name"):
                    QMessageBox.warning(self, "Invalid file", "Selected CSV does not contain metadata header.")
                    return
                for row in reader:
                    if not row or all(not cell.strip() for cell in row):
                        break
                    if len(row) >= 2:
                        metadata[row[0].strip()] = row[1]
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not read metadata: {exc}")
            return

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

        def _set_fix_state(name, key):
            try:
                checked = str(metadata[key]).strip().lower() in ("1", "true", "yes", "y", "on")
                self.set_fix_parameter_state(name, checked)
            except (KeyError, TypeError, ValueError):
                pass

        def _set_checkbox_bool(checkbox, *keys):
            for key in keys:
                if key not in metadata:
                    continue
                checked = str(metadata[key]).strip().lower() in ("1", "true", "yes", "y", "on")
                checkbox.setChecked(checked)
                return

        _set_line_float(self.min_wavelength_input, "min_wavelength")
        _set_line_float(self.max_wavelength_input, "max_wavelength")
        _set_int(self.box_width_input, "box_width")
        _set_int(self.box_height_input, "box_height")
        _set_int(self.step_x_input, "step_x")
        _set_int(self.step_y_input, "step_y")
        _set_int(self.min_x_input, "roi_x_min")
        _set_int(self.max_x_input, "roi_x_max")
        _set_int(self.min_y_input, "roi_y_min")
        _set_int(self.max_y_input, "roi_y_max")
        _set_checkbox_bool(self.interpolation_checkbox, "interpolation", "interpolation_enabled")

        _set_fix_state("s", "fix_s")
        _set_fix_state("t", "fix_t")
        _set_fix_state("eta", "fix_eta")

        try:
            self.flight_path = float(metadata["flight_path"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            self.delay = float(metadata["delay"])
        except (KeyError, TypeError, ValueError):
            pass

        phase_name = metadata.get("selected_phase")
        if phase_name:
            idx = self.phase_dropdown.findText(phase_name)
            if idx == -1:
                self.phase_dropdown.addItem(phase_name)
                idx = self.phase_dropdown.findText(phase_name)
            if idx >= 0:
                self.phase_dropdown.setCurrentIndex(idx)

        bragg_rows = [(k, v) for k, v in metadata.items() if k.lower().startswith("bragg_table_row_")]
        if bragg_rows:
            bragg_rows.sort(key=lambda kv: int("".join(filter(str.isdigit, kv[0])) or 0))
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

        mapping_keys = {
            "box_width",
            "box_height",
            "step_x",
            "step_y",
            "roi_x_min",
            "roi_x_max",
            "roi_y_min",
            "roi_y_max",
            "interpolation",
            "interpolation_enabled",
        }
        if any(key in metadata for key in mapping_keys):
            self._apply_mapping_settings_from_inputs(persist=False)

        self.save_user_settings()
        QMessageBox.information(self, "Configuration loaded", "Metadata applied where available.")

    def save_fitting_configuration_to_csv(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Fitting Configuration",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_name:
            return
        if not file_name.lower().endswith(".csv"):
            file_name += ".csv"

        metadata = self._build_fitting_configuration_metadata()
        try:
            with open(file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metadata Name", "Metadata Value"])
                writer.writerows(metadata)
                writer.writerow([])
            QMessageBox.information(self, "Configuration saved", f"Fitting configuration saved to:\n{file_name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save fitting configuration:\n{exc}")

    def _build_fitting_configuration_metadata(self):
        context = self._build_batch_fit_context()
        metadata = [
            ("box_width", self.box_width_input.text()),
            ("box_height", self.box_height_input.text()),
            ("step_x", self.step_x_input.text()),
            ("step_y", self.step_y_input.text()),
            ("interpolation", self.interpolation_checkbox.isChecked()),
            ("roi_x_min", self.min_x_input.text()),
            ("roi_x_max", self.max_x_input.text()),
            ("roi_y_min", self.min_y_input.text()),
            ("roi_y_max", self.max_y_input.text()),
            ("number_of_edges", self.bragg_table.rowCount()),
            ("directory", getattr(self, "work_directory", "")),
            ("fix_s", self.fix_s_enabled()),
            ("fix_t", self.fix_t_enabled()),
            ("fix_eta", self.fix_eta_enabled()),
            ("flight_path", getattr(self, "flight_path", "")),
            ("flight_path_source", getattr(self, "flight_path_source", "")),
            ("data_source", getattr(self, "fitting_data_source", "")),
            ("input_file", getattr(self, "current_fitting_input", "")),
            ("delay", getattr(self, "delay", "")),
            ("min_wavelength", context.get("min_wavelength")),
            ("max_wavelength", context.get("max_wavelength")),
            ("selected_phase", context.get("selected_phase")),
        ]
        for row_idx, row_text in enumerate(context.get("bragg_rows_text", []), start=1):
            metadata.append((f"bragg_table_row_{row_idx}", row_text))
        return metadata

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
        Load image files by selecting a folder. Only images with suffixes from _00000 to _02924 are loaded.
        After loading, perform intensity check and scaling if necessary.
        """
        # Open a folder dialog to select a directory containing image files.
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Images", ""
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
            self._set_fits_image_button_states(is_loading=True)
            self._ensure_load_progress_dialog().setWindowTitle("Loading Images")

            # Start image loading in a separate thread using ImageLoadWorker
            self.fits_image_load_worker = ImageLoadWorker(folder_path)
            self.fits_image_load_worker.progress_updated.connect(self.update_fits_load_progress)
            self.fits_image_load_worker.message.connect(self.message_box.append)
            self.fits_image_load_worker.run_loaded.connect(self.handle_fits_run_loaded)
            self.fits_image_load_worker.finished.connect(self.fits_image_loading_finished)
            self.fits_image_load_worker.start()

    def load_nexus_image_stack(self):
        """Load a NeXus file containing a detector-by-wavelength image stack."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select NeXus Image Stack",
            "",
            "NeXus Files (*.nxs *.h5 *.hdf5);;All Files (*)",
        )
        if not file_name:
            return

        existing_worker = getattr(self, "fits_image_load_worker", None)
        if existing_worker is not None and existing_worker.isRunning():
            self.message_box.append("Image loading is already in progress.")
            return

        try:
            info = get_nexus_image_stack_info(file_name)
        except Exception as exc:
            QMessageBox.warning(self, "Load NeXus Stack", f"Could not inspect NeXus file:\n{exc}")
            return

        flight_path, source_label = self._choose_nexus_flight_path(info)
        if flight_path is None:
            self.message_box.append("NeXus image-stack loading cancelled.")
            return

        self.fits_image_load_worker = None
        self.folder_path = os.path.dirname(file_name)
        self.work_directory = os.path.dirname(file_name)
        self._set_fits_image_button_states(is_loading=True)
        self._ensure_load_progress_dialog().setWindowTitle("Loading NeXus Image Stack")

        self.fits_image_load_worker = NexusImageStackLoadWorker(file_name, flight_path)
        self.fits_image_load_worker.progress_updated.connect(self.update_fits_load_progress)
        self.fits_image_load_worker.message.connect(self.message_box.append)
        self.fits_image_load_worker.stack_loaded.connect(
            lambda path, images, wavelengths, used_path, stack_info: self.handle_nexus_stack_loaded(
                path,
                images,
                wavelengths,
                used_path,
                stack_info,
                source_label,
            )
        )
        self.fits_image_load_worker.finished.connect(self.fits_image_loading_finished)
        self.fits_image_load_worker.start()

    @staticmethod
    def _format_byte_size(num_bytes):
        value = float(num_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024.0 or unit == "TB":
                return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0
        return f"{value:.2f} TB"

    def load_raden_tiff_stack(self):
        """Load a RADEN multi-page TIFF stack with sidecar TOF metadata."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select RADEN TIFF Stack Folder",
            "",
        )
        if not folder_path:
            return

        existing_worker = getattr(self, "fits_image_load_worker", None)
        if existing_worker is not None and existing_worker.isRunning():
            self.message_box.append("Image loading is already in progress.")
            return

        try:
            info = get_raden_tiff_stack_info(folder_path)
        except Exception as exc:
            QMessageBox.warning(self, "Load RADEN TIFF Stack", f"Could not inspect RADEN TIFF stack:\n{exc}")
            return

        flight_path = float(getattr(self, "flight_path", 0.0) or 0.0)
        if not np.isfinite(flight_path) or flight_path <= 0:
            QMessageBox.warning(
                self,
                "Load RADEN TIFF Stack",
                "Set a positive flight path in Instrument Setting before loading a RADEN TOF stack.",
            )
            return

        tof_axis = info.get("axes", {}).get("tof", {})
        estimated = self._format_byte_size(info.get("estimated_bytes", 0))
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("Load RADEN TIFF Stack")
        message.setText("Load this RADEN multi-page TIFF stack for Bragg-edge fitting?")
        message.setInformativeText(
            f"Frames: {info['n_frames']}\n"
            f"Image size: {info['image_shape'][1]} x {info['image_shape'][0]}\n"
            f"TOF range: {self._format_number(tof_axis.get('min'), 6)} - "
            f"{self._format_number(tof_axis.get('max'), 6)} {tof_axis.get('units', '')}\n"
            f"Flight path: {flight_path:.6f} m (App setting)\n"
            f"Estimated image memory: {estimated}"
        )
        message.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        message.setDefaultButton(QMessageBox.Cancel)
        if message.exec_() != QMessageBox.Ok:
            self.message_box.append("RADEN TIFF stack loading cancelled.")
            return

        self.fits_image_load_worker = None
        self.folder_path = os.path.dirname(info["file_path"])
        self.work_directory = os.path.dirname(info["file_path"])
        self._set_fits_image_button_states(is_loading=True)
        self._ensure_load_progress_dialog().setWindowTitle("Loading RADEN TIFF Stack")

        self.fits_image_load_worker = RadenTiffStackLoadWorker(info["file_path"], flight_path)
        self.fits_image_load_worker.progress_updated.connect(self.update_fits_load_progress)
        self.fits_image_load_worker.message.connect(self.message_box.append)
        self.fits_image_load_worker.stack_loaded.connect(
            lambda path, images, wavelengths, used_path, stack_info: self.handle_raden_stack_loaded(
                path,
                images,
                wavelengths,
                used_path,
                stack_info,
            )
        )
        self.fits_image_load_worker.finished.connect(self.fits_image_loading_finished)
        self.fits_image_load_worker.start()

    def _choose_nexus_flight_path(self, info):
        app_flight_path = float(getattr(self, "flight_path", 0.0) or 0.0)
        nexus_flight_path = info.get("file_flight_path")
        if nexus_flight_path is None or not np.isfinite(nexus_flight_path) or nexus_flight_path <= 0:
            QMessageBox.information(
                self,
                "NeXus Flight Path",
                (
                    "The NeXus file does not contain enough geometry to calculate a flight path.\n"
                    f"The current app setting ({app_flight_path:.6f} m) will be used."
                ),
            )
            return app_flight_path, "App setting"

        source_sample = info.get("source_sample_distance")
        sample_detector = info.get("sample_detector_distance")
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("NeXus Flight Path")
        message.setText("Choose the flight path used to convert NeXus TOF bins to wavelength.")
        message.setInformativeText(
            f"NeXus geometry: {nexus_flight_path:.6f} m\n"
            f"  source-sample: {self._format_number(source_sample, 6)} m\n"
            f"  sample-detector: {self._format_number(sample_detector, 6)} m\n\n"
            f"Current app setting: {app_flight_path:.6f} m"
        )
        nexus_button = message.addButton("Use NeXus File", QMessageBox.AcceptRole)
        app_button = message.addButton("Use App Setting", QMessageBox.ActionRole)
        message.addButton(QMessageBox.Cancel)
        message.setDefaultButton(nexus_button)
        message.exec_()

        clicked = message.clickedButton()
        if clicked is nexus_button:
            return float(nexus_flight_path), "NeXus file"
        if clicked is app_button:
            return app_flight_path, "App setting"
        return None, ""

    def handle_raden_stack_loaded(self, file_path, images, wavelengths, flight_path, info):
        """Install a loaded RADEN TIFF stack into the standard fitting state."""
        self.fitting_data_source = "images"
        self.flight_path = float(flight_path)
        self.flight_path_source = "App setting"
        self.current_fitting_input = file_path
        self.images = list(images)
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.tof_axis_centers_us = np.asarray(info.get("tof_axis_centers_us"), dtype=float)
        self.wavelength_depends_on_flight_path = True
        self.nexus_axis_centers = None
        self.nexus_axis_uses_flight_path = False
        self.wavelength_flight_path = float(flight_path)
        self.tof_array = None
        self.intensities = np.array([])
        self.selected_region_x = np.array([])
        self.selected_region_y = np.array([])
        self.current_image_index = 0
        self.current_batch_box = None
        self.batch_box_patch = None
        self.batch_roi_visible = False
        self.manual_wavelength_mode = False

        self._recalculate_wavelengths_from_tof_axis(
            source_label=self.flight_path_source,
            show_message=False,
            axis_label="RADEN",
        )

        self.image_slider.setEnabled(True)
        self.image_slider.setRange(0, len(self.images) - 1)
        self.image_slider.setValue(0)
        self._set_fits_image_button_states(is_loading=False)
        self.display_image()
        self.apply_selected_area_from_inputs()
        self.message_box.append(
            "RADEN TIFF stack ready for Bragg-edge fitting. "
            f"Flight path used: {self.flight_path:.6f} m ({self.flight_path_source}). "
            f"Wavelength range: {self.start_wavelength:.6f} - {self.end_wavelength:.6f} A."
        )

    def handle_nexus_stack_loaded(self, file_path, images, wavelengths, flight_path, info, source_label):
        """Install a loaded NeXus image stack into the standard fitting state."""
        self.fitting_data_source = "images"
        self.flight_path = float(flight_path)
        self.flight_path_source = source_label or "NeXus file"
        self.current_fitting_input = file_path
        self.images = list(images)
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.nexus_axis_centers = info.get("axis_centers")
        if self.nexus_axis_centers is not None:
            self.nexus_axis_centers = np.asarray(self.nexus_axis_centers, dtype=float)
        self.nexus_axis_uses_flight_path = bool(info.get("axis_uses_flight_path", False))
        self.tof_axis_centers_us = self.nexus_axis_centers if self.nexus_axis_uses_flight_path else None
        self.wavelength_depends_on_flight_path = self.nexus_axis_uses_flight_path
        self.wavelength_flight_path = float(flight_path) if self.nexus_axis_uses_flight_path else None
        self.tof_array = None
        self.intensities = np.array([])
        self.selected_region_x = np.array([])
        self.selected_region_y = np.array([])
        self.current_image_index = 0
        self.current_batch_box = None
        self.batch_box_patch = None
        self.batch_roi_visible = False
        self.manual_wavelength_mode = False

        if self.nexus_axis_uses_flight_path:
            self._recalculate_wavelengths_from_tof_axis(
                source_label=self.flight_path_source,
                show_message=False,
                axis_label="NeXus",
            )
        elif self.wavelengths.size:
            self._update_wavelength_bounds_from_current_array()

        self.image_slider.setEnabled(True)
        self.image_slider.setRange(0, len(self.images) - 1)
        self.image_slider.setValue(0)
        self._set_fits_image_button_states(is_loading=False)
        self.display_image()
        self.apply_selected_area_from_inputs()
        self.message_box.append(
            "NeXus image stack ready for Bragg-edge fitting. "
            f"Flight path used: {self.flight_path:.6f} m ({self.flight_path_source}). "
            f"Wavelength range: {self.start_wavelength:.6f} - {self.end_wavelength:.6f} A."
        )

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
            self._clear_image_canvas()
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

        # Draw the static batch ROI from min/max inputs (orange) only after the
        # Batch Fit Settings dialog has been applied in this session.
        if getattr(self, "batch_roi_visible", False):
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
    def apply_selected_area_from_inputs(self):
        """Apply ROI coordinate edits immediately when they form a valid image ROI."""
        if not getattr(self, "images", []):
            return
        try:
            xmin = int(self.xmin_input.text())
            xmax = int(self.xmax_input.text())
            ymin = int(self.ymin_input.text())
            ymax = int(self.ymax_input.text())
        except ValueError:
            return
        if xmin >= xmax or ymin >= ymax:
            return
        height, width = self.images[0].shape
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            return
        new_area = (xmin, xmax, ymin, ymax)
        if getattr(self, "selected_area", None) == new_area:
            return
        self.selected_area = new_area
        self._bragg_table_ready = True
        self.update_bragg_edge_table()
        self.display_image()
        self.update_plots()

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
            self.run_live_fit_preview()
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
        if not getattr(self, "batch_roi_visible", False):
            return None
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
        layout_key = self._fit_canvas_layout_key(canvas)

        valid_axes = (
            main_ax is not None
            and residual_ax is not None
            and main_ax in fig.axes
            and residual_ax in fig.axes
            and getattr(canvas, "_fit_canvas_layout_key", None) == layout_key
        )
        if not valid_axes:
            fig.clf()
            grid = fig.add_gridspec(
                2,
                1,
                height_ratios=self._fit_canvas_height_ratios(canvas),
                hspace=0.0,
            )
            main_ax = fig.add_subplot(grid[0, 0])
            residual_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
            canvas.axes = main_ax
            canvas.residual_axes = residual_ax
            canvas._fit_canvas_layout_key = layout_key

        residual_ax.set_xlabel(self._fit_canvas_xlabel(canvas))
        return main_ax, residual_ax

    def _fit_canvas_layout_key(self, canvas):
        """Return a key for figure geometry that should trigger axis recreation."""
        mode = getattr(self, "fitting_plot_layout_mode", "single")
        if canvas is getattr(self, "plot_canvas_a", None):
            return f"{mode}:primary"
        return f"{mode}:secondary"

    def _fit_canvas_height_ratios(self, canvas):
        """Return main/residual height ratios tuned for the active plot layout."""
        if self.is_single_fit_canvas_mode() and canvas is getattr(self, "plot_canvas_a", None):
            return [8, 1]
        return [5, 1]

    def _fit_canvas_xlabel(self, canvas):
        """Return residual x-axis label text for a fit canvas based on its 2x2 position."""
        if self.is_single_fit_canvas_mode() and canvas is getattr(self, "plot_canvas_a", None):
            return "Wavelength (A)"
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

        left_margin, right_margin, top_margin, bottom_margin = self._fit_canvas_margins(canvas, font_size)
        canvas.fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=top_margin,
            bottom=bottom_margin,
            hspace=0.0,
        )

        return main_ax, residual_ax

    def _fit_canvas_margins(self, canvas, font_size):
        """Return figure margins tuned for single-canvas and four-canvas layouts."""
        scale = max(float(font_size), 6.0)
        if self.is_single_fit_canvas_mode() and canvas is getattr(self, "plot_canvas_a", None):
            return (
                min(0.16, max(0.08, 0.04 + 0.0035 * scale)),
                0.985,
                0.965,
                min(0.18, max(0.10, 0.055 + 0.004 * scale)),
            )
        return (
            min(0.24, max(0.14, 0.08 + 0.0045 * scale)),
            0.975,
            0.955,
            min(0.24, max(0.13, 0.075 + 0.0045 * scale)),
        )

    def _append_fit_message(self, text):
        """Append a visually separated fitting message block."""
        box = getattr(self, "message_box", None)
        if box is None:
            return

        lines = str(text).strip().splitlines()
        if not lines:
            return

        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip("\n")
        palette = self._fit_message_palette(title)
        timestamp = time.strftime("%H:%M:%S")
        font_size = max(1, int(getattr(self, "gui_font_size", box.font().pointSize() or 8)))

        body_html = ""
        if body:
            body_html = (
                "<pre style=\""
                "margin:4px 0 8px 10px;"
                "font-family:Consolas, 'Courier New', monospace;"
                "font-size:{font_size}pt;"
                "white-space:pre-wrap;"
                "color:#222;"
                "\">"
                f"{escape(body)}"
                "</pre>"
            ).format(font_size=font_size)

        block_html = (
            "<div style=\""
            "margin-top:8px;"
            "margin-bottom:6px;"
            "border:1px solid {border};"
            "border-left:4px solid {accent};"
            "background:{background};"
            "\">"
            "<div style=\""
            "padding:4px 7px;"
            "font-weight:600;"
            "font-size:{font_size}pt;"
            "color:{header};"
            "background:{header_bg};"
            "\">"
            "<span style=\"color:#666; font-weight:400;\">[{time}]</span> {title}"
            "</div>"
            "{body}"
            "</div>"
        ).format(
            border=palette["border"],
            accent=palette["accent"],
            background=palette["background"],
            header=palette["header"],
            header_bg=palette["header_bg"],
            font_size=font_size,
            time=escape(timestamp),
            title=escape(title),
            body=body_html,
        )
        box.append(block_html)
        scrollbar = box.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _fit_message_palette(title):
        """Return colours for a fitting message block based on its title."""
        lowered = title.lower()
        if "failed" in lowered or "error" in lowered:
            return {
                "accent": "#c62828",
                "border": "#efb7b7",
                "background": "#fff7f7",
                "header": "#7f1d1d",
                "header_bg": "#fde2e2",
            }
        if "completed" in lowered or "saved" in lowered:
            return {
                "accent": "#2e7d32",
                "border": "#b7d9ba",
                "background": "#f6fbf6",
                "header": "#1f5f24",
                "header_bg": "#e3f2e5",
            }
        if "started" in lowered or "requested" in lowered:
            return {
                "accent": "#1565c0",
                "border": "#b8cbe6",
                "background": "#f6f9fd",
                "header": "#0d47a1",
                "header_bg": "#e5eef9",
            }
        return {
            "accent": "#6b7280",
            "border": "#d7dbe0",
            "background": "#fbfbfc",
            "header": "#30343b",
            "header_bg": "#eef0f3",
        }

    def _append_batch_message(self, text):
        """Append a batch-processing message using the fitting block style."""
        raw = str(text or "").strip()
        if not raw:
            return
        if raw.startswith("["):
            self._append_fit_message(raw)
            return

        lowered = raw.lower()
        saved_marker = " saved to "
        if saved_marker in lowered:
            idx = lowered.find(saved_marker)
            title = raw[:idx].strip()
            path = raw[idx + len(saved_marker):].strip()
            self._append_fit_message(f"[Batch fit] {title} saved\n  {path}")
            return

        if "failed" in lowered or "error" in lowered:
            self._append_fit_message(f"[Batch fit] Failed\n  {raw}")
        elif "stopped" in lowered or "cancelled" in lowered:
            self._append_fit_message(f"[Batch fit] Stopped\n  {raw}")
        elif "completed" in lowered:
            self._append_fit_message(f"[Batch fit] Completed\n  {raw}")
        else:
            self._append_fit_message(f"[Batch fit] Update\n  {raw}")

    @staticmethod
    def _format_hkl_label(hkl):
        if isinstance(hkl, tuple):
            return "hkl(" + ", ".join(str(v) for v in hkl) + ")"
        return str(hkl)

    @staticmethod
    def _format_number(value, digits=6):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(number):
            return "n/a"
        return f"{number:.{digits}f}"

    def _format_value_unc(self, value, uncertainty, unit="", digits=6):
        value_text = self._format_number(value, digits)
        unc_text = self._format_number(uncertainty, digits)
        suffix = f" {unit}" if unit else ""
        if unc_text == "n/a":
            return f"{value_text}{suffix}"
        return f"{value_text} +/- {unc_text}{suffix}"

    def _append_pattern_fit_summary(self, result_dict, min_wavelength=None, max_wavelength=None):
        """Append a compact summary for a successful pattern fit."""
        structure_type = result_dict.get("structure_type", "")
        lattice_params = result_dict.get("lattice_params", {})
        lattice_unc = result_dict.get("lattice_uncertainties", {})
        bragg_edges = result_dict.get("bragg_edges", [])
        edge_heights = result_dict.get("edge_heights", {})
        edge_widths = result_dict.get("edge_widths", {})
        fitted_s = result_dict.get("fitted_s", {})
        fitted_t = result_dict.get("fitted_t", {})
        fitted_eta = result_dict.get("fitted_eta", {})
        s_unc = result_dict.get("s_uncertainties", {})
        t_unc = result_dict.get("t_uncertainties", {})
        eta_unc = result_dict.get("eta_uncertainties", {})

        structure_map = {
            "cubic": ["a"],
            "fcc": ["a"],
            "bcc": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"],
        }

        lines = ["[Pattern fit] Results"]
        phase = getattr(self, "phase_dropdown", None)
        phase_text = phase.currentText() if phase is not None else ""
        if phase_text:
            lines.append(f"  Phase: {phase_text}")
        lines.append(f"  Flight path: {self._format_number(getattr(self, 'flight_path', np.nan), 6)} m")
        source = getattr(self, "flight_path_source", "")
        if source:
            lines.append(f"  Flight path source: {source}")
        if np.isfinite(min_wavelength) and np.isfinite(max_wavelength):
            lines.append(f"  Wavelength range: {min_wavelength:.4f} - {max_wavelength:.4f} A")
        lines.append(f"  Edges used: {len(bragg_edges)}")

        param_names = structure_map.get(structure_type, [])
        if param_names:
            lines.append("")
            lines.append("  Lattice:")
            for param in param_names:
                lines.append(
                    f"    {param} = {self._format_value_unc(lattice_params.get(param), lattice_unc.get(param), 'A')}"
                )

        if bragg_edges:
            lines.append("")
            lines.append("  Edge results:")
            for edge in bragg_edges:
                hkl = edge.get("hkl")
                label = self._format_hkl_label(hkl)
                parts = [
                    f"s={self._format_value_unc(fitted_s.get(hkl), s_unc.get(hkl), digits=5)}",
                    f"t={self._format_value_unc(fitted_t.get(hkl), t_unc.get(hkl), digits=5)}",
                    f"eta={self._format_value_unc(fitted_eta.get(hkl), eta_unc.get(hkl), digits=4)}",
                    f"height={self._format_number(edge_heights.get(hkl), 6)}",
                    f"FWHM={self._format_number(edge_widths.get(hkl), 6)} A",
                ]
                lines.append(f"    {label}: " + ", ".join(parts))

        self._append_fit_message("\n".join(lines))
        self._append_fit_message("[Pattern fit] Completed")

    def _append_individual_fit_summary(self, result, row_number):
        """Append one compact summary for a successful individual edge fit."""
        hkl = result.get("hkl")
        d_fit, s_fit, t_fit, eta_fit, d_unc, s_unc, t_unc, eta_unc = result.get(
            "fit_params", (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        )
        baseline = result.get("baseline_params", {})
        lines = [
            f"  Edge {row_number + 1} {self._format_hkl_label(hkl)}:",
            f"    d = {self._format_value_unc(d_fit, d_unc, 'A')}",
            f"    height = {self._format_number(result.get('edge_height'), 6)}",
            f"    FWHM = {self._format_number(result.get('edge_width'), 6)} A",
            f"    RMS = {self._format_number(result.get('rms'), 6)}",
            (
                "    baseline: "
                f"a0={self._format_number(baseline.get('a0'), 6)}, "
                f"b0={self._format_number(baseline.get('b0'), 6)}, "
                f"a_hkl={self._format_number(baseline.get('a_hkl'), 6)}, "
                f"b_hkl={self._format_number(baseline.get('b_hkl'), 6)}"
            ),
        ]
        free_params = []
        if np.isfinite(s_unc):
            free_params.append(f"s={self._format_value_unc(s_fit, s_unc, digits=6)}")
        if np.isfinite(t_unc):
            free_params.append(f"t={self._format_value_unc(t_fit, t_unc, digits=6)}")
        if np.isfinite(eta_unc):
            free_params.append(f"eta={self._format_value_unc(eta_fit, eta_unc, digits=4)}")
        if free_params:
            lines.append("    fitted shape: " + ", ".join(free_params))
        self._append_fit_message("\n".join(lines))

    def _add_live_fit_annotation(self, ax, text, data_x=None, data_y=None, fit_x=None, fit_y=None):
        x_pos, y_pos, ha, va = self._choose_live_annotation_position(
            ax,
            text,
            data_x,
            data_y,
            fit_x=fit_x,
            fit_y=fit_y,
        )
        ax.text(
            x_pos,
            y_pos,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=getattr(self, "plot_font_size", 12),
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 2,
            },
        )

    def _choose_live_annotation_position(self, ax, text, data_x=None, data_y=None, fit_x=None, fit_y=None):
        text_lines = str(text).splitlines() or [""]
        line_count = len(text_lines)
        max_line_length = max(len(line) for line in text_lines)
        box_width = min(0.62, max(0.24, 0.015 * max_line_length + 0.08))
        box_height = min(0.44, max(0.12, 0.075 * line_count + 0.06))
        margin = 0.02
        bottom_margin = 0.08
        candidates = [
            self._live_annotation_candidate(margin, 1.0 - margin, "left", "top", box_width, box_height),
            self._live_annotation_candidate(1.0 - margin, 1.0 - margin, "right", "top", box_width, box_height),
            self._live_annotation_candidate(margin, bottom_margin, "left", "bottom", box_width, box_height),
            self._live_annotation_candidate(1.0 - margin, bottom_margin, "right", "bottom", box_width, box_height),
        ]
        points = []
        for x_arr, y_arr, weight in (
            (data_x, data_y, 1.0),
            (fit_x, fit_y, 2.0),
        ):
            axes_points = self._data_points_to_axes_fraction(ax, x_arr, y_arr)
            if axes_points.size:
                points.append((axes_points, weight))
        for axes_points, weight in self._axes_artist_occupancy_points(ax):
            if axes_points.size:
                points.append((axes_points, weight))

        best = candidates[1]
        best_score = np.inf
        for preference, candidate in enumerate(candidates):
            x0, x1, y0, y1 = candidate["bounds"]
            score = 0.0
            for axes_points, weight in points:
                xs = axes_points[:, 0]
                ys = axes_points[:, 1]
                in_box = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                score += weight * np.count_nonzero(in_box)
                if np.any(in_box):
                    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
                    distances = np.linalg.norm(axes_points[in_box] - center, axis=1)
                    score += weight * np.sum(np.maximum(0.0, 1.0 - distances))
            score += preference * 0.01
            if score < best_score:
                best_score = score
                best = candidate
        return best["x"], best["y"], best["ha"], best["va"]

    @staticmethod
    def _live_annotation_candidate(x, y, ha, va, width, height):
        if ha == "right":
            x0 = max(0.0, x - width)
            x1 = min(1.0, x)
        else:
            x0 = max(0.0, x)
            x1 = min(1.0, x + width)
        if va == "top":
            y0 = max(0.0, y - height)
            y1 = min(1.0, y)
        else:
            y0 = max(0.0, y)
            y1 = min(1.0, y + height)
        return {"x": x, "y": y, "ha": ha, "va": va, "bounds": (x0, x1, y0, y1)}

    def _axes_artist_occupancy_points(self, ax):
        points = []
        for line in ax.lines:
            try:
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
            except (TypeError, ValueError):
                continue
            if x.size == 0 or y.size == 0:
                continue
            n = min(x.size, y.size)
            x = x[:n]
            y = y[:n]
            if n == 2 and np.isfinite(x).all() and np.isclose(x[0], x[1]):
                x_axes = self._data_points_to_axes_fraction(ax, [x[0]], [np.mean(ax.get_ylim())])
                if x_axes.size:
                    vertical_points = np.column_stack(
                        (np.full(24, x_axes[0, 0]), np.linspace(0.0, 1.0, 24))
                    )
                    points.append((vertical_points, 1.5))
                continue
            if n > 400:
                sample = np.linspace(0, n - 1, 400, dtype=int)
                x = x[sample]
                y = y[sample]
            axes_points = self._data_points_to_axes_fraction(ax, x, y)
            if axes_points.size:
                weight = 2.0 if line.get_linestyle() not in ("None", "none", "") else 1.0
                points.append((axes_points, weight))
        return points

    @staticmethod
    def _data_points_to_axes_fraction(ax, x_arr, y_arr):
        if x_arr is None or y_arr is None:
            return np.empty((0, 2))
        try:
            x = np.asarray(x_arr, dtype=float)
            y = np.asarray(y_arr, dtype=float)
        except (TypeError, ValueError):
            return np.empty((0, 2))
        if x.size == 0 or y.size == 0:
            return np.empty((0, 2))
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.empty((0, 2))
        xy_display = ax.transData.transform(np.column_stack((x[finite], y[finite])))
        return ax.transAxes.inverted().transform(xy_display)

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
        if getattr(self, "fitting_data_source", "images") == "profile":
            if not hasattr(self, "wavelengths") or self.wavelengths is None:
                return
            if not hasattr(self, "intensities") or self.intensities is None:
                return
            self.wavelengths = np.asarray(self.wavelengths, dtype=float)
            self.intensities = np.asarray(self.intensities, dtype=float)
            if self.wavelengths.size == 0 or self.intensities.size == 0:
                return
            if self.wavelengths.size != self.intensities.size:
                self.message_box.append("Cannot plot because wavelength and intensity profile lengths do not match.")
                return
        else:
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
        edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength) if self.show_edge_lines_enabled() else []
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

        if self.is_single_fit_canvas_mode():
            return

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

    def fit_all_regions(self, emit_messages=True, update_table=True, only_update_unfixed=False):
        """
        Performs the Region 1, 2, 3 fit for each row in the bragg_table, but only
        for rows that have valid (non-empty) data in all required columns.
        """
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

        # Start each run from clean result canvases so previous fits do not accumulate.
        max_rows = min(5, self.bragg_table.rowCount())
        region_ranges = {
            "Region 1": _range_from_rows(2, 3, max_rows),  # plotted on canvas_b
            "Region 2": _range_from_rows(4, 5, max_rows),  # plotted on canvas_c
            "Region 3": _range_from_rows(6, 7, max_rows),  # plotted on canvas_d
        }
        if self.is_single_fit_canvas_mode():
            ax_main, _ = self._initialize_fit_canvas(self.plot_canvas_a)
            raw_x = np.asarray(getattr(self, "selected_region_x", []), dtype=float)
            raw_y = np.asarray(getattr(self, "selected_region_y", []), dtype=float)
            if raw_x.size and raw_y.size:
                n_points = min(raw_x.size, raw_y.size)
                ax_main.plot(
                    raw_x[:n_points],
                    raw_y[:n_points],
                    "o",
                    markersize=getattr(self, "symbol_size", 4),
                    markerfacecolor="blue",
                    markeredgecolor="none",
                )
            if self.show_edge_lines_enabled():
                try:
                    min_wavelength = float(self.min_wavelength_input.text())
                    max_wavelength = float(self.max_wavelength_input.text())
                except (AttributeError, TypeError, ValueError):
                    finite_x = raw_x[np.isfinite(raw_x)]
                    if finite_x.size:
                        min_wavelength = float(np.nanmin(finite_x))
                        max_wavelength = float(np.nanmax(finite_x))
                    else:
                        min_wavelength = max_wavelength = np.nan
                if np.isfinite(min_wavelength) and np.isfinite(max_wavelength) and min_wavelength < max_wavelength:
                    x_span = max_wavelength - min_wavelength
                    for hkl, x_hkl in self.get_edges_in_range(min_wavelength, max_wavelength):
                        ax_main.axvline(x=x_hkl, color="red", linestyle="--")
                        y_max = ax_main.get_ylim()[1]
                        ax_main.text(
                            x_hkl + 0.01 * x_span,
                            y_max * 0.95,
                            f"hkl{hkl}",
                            rotation=90,
                            verticalalignment="top",
                            color="red",
                            fontsize=getattr(self, "plot_font_size", 12),
                        )
            x_range = region_ranges.get("Region 3")
            if x_range is not None:
                xlim = self._xlim_with_margin(x_range[0], x_range[1])
                if xlim is not None:
                    ax_main.set_xlim(*xlim)
            self.plot_canvas_a.draw()
        else:
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

        fitted_any = False
        fit_results = []
        attempted = 0
        failed = 0
        if emit_messages:
            complete_rows = sum(1 for row in range(max_rows) if self.is_row_complete(row))
            self._append_fit_message(
                "[Individual edge fit] Started\n"
                f"  Edges requested: {complete_rows}\n"
                f"  Flight path: {self._format_number(getattr(self, 'flight_path', np.nan), 6)} m"
            )
        for row in range(max_rows):
            # Check whether this row has all required inputs
            if not self.is_row_complete(row):
                # self.message_box.append(f"Skipping Row {row + 1}: incomplete or missing data.")
                continue

            # If row is valid, do the fit
            attempted += 1
            result = self.fit_region(
                row,
                emit_messages=emit_messages,
                update_table=update_table,
                only_update_unfixed=only_update_unfixed,
            )
            fitted_any = fitted_any or bool(result)
            if result:
                fit_results.append(result)
                if emit_messages:
                    self._append_individual_fit_summary(result, row)
            else:
                failed += 1
                if emit_messages:
                    self._append_fit_message(f"  Edge {row + 1}: failed")

        if emit_messages:
            self._append_fit_message(
                "[Individual edge fit] Completed\n"
                f"  Succeeded: {len(fit_results)}\n"
                f"  Failed: {failed}\n"
                f"  Attempted: {attempted}"
            )
        self.set_fit_action_buttons_enabled(True)
        if not emit_messages:
            return fit_results
        return fitted_any

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
        emit_messages=True,
        update_table=True,
        only_update_unfixed=False,
    ):

        draw_plots = not skip_ui_updates
        emit_messages = bool(emit_messages and draw_plots)
        update_table = bool(update_table and draw_plots)
        if fit_flags is None:
            fix_s = self.fix_s_enabled()
            fix_t = self.fix_t_enabled()
            fix_eta = self.fix_eta_enabled()
        else:
            fix_s, fix_t, fix_eta = fit_flags
        selected_phase = selected_phase if selected_phase is not None else self.phase_dropdown.currentText()
        is_known_phase = selected_phase != "Unknown_Phase"
        wavelengths_data = self.wavelengths if wavelengths is None else np.asarray(wavelengths)
        intensities_data = self.intensities if intensities is None else np.asarray(intensities)
        lattice_params_data = dict(lattice_params if lattice_params is not None else getattr(self, "lattice_params", {}))
        structure_type_data = structure_type if structure_type is not None else getattr(self, "structure_type", "fcc" if is_known_phase else "unknown")
        params_store = {}
        if draw_plots:
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
            if emit_messages:
                self._append_fit_message(f"  Edge {row_number + 1}: setup failed - {e}")
            return
        # -------------------------------------------------
        #  Region 1 Fitting
        # -------------------------------------------------
        try:
            mask_r1 = (wavelengths_data >= r1_min) & (wavelengths_data <= r1_max)
            x_r1 = wavelengths_data[mask_r1]
            y_r1 = intensities_data[mask_r1]

            if x_r1.size == 0:
                if emit_messages:
                    self._append_fit_message(f"  Edge {row_number + 1}: Region 1 has no data in range")
                return

            if draw_plots and not self.is_single_fit_canvas_mode():
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

            if draw_plots and not self.is_single_fit_canvas_mode():
                ax_r1_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_c)
                ax_r1_main.plot(x_r1, fit_y1, 'r-', 
                                             # label=f"Edge {row_number + 1} Fit R1"
                                             )
                self._plot_residual_line(self.plot_canvas_c, x_r1, y_r1, fit_y1)
                # self.plot_canvas_c.axes.legend()
                self.plot_canvas_c.draw()
        except Exception as e:
            if emit_messages:
                self._append_fit_message(f"  Edge {row_number + 1}: Region 1 fit failed - {e}")
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
                if emit_messages:
                    self._append_fit_message(f"  Edge {row_number + 1}: Region 2 has no data in range")
                return

            if draw_plots and not self.is_single_fit_canvas_mode():
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

            if draw_plots and not self.is_single_fit_canvas_mode():
                ax_r2_main, _ = self._ensure_fit_canvas_axes(self.plot_canvas_b)
                ax_r2_main.plot(
                    x_r2, fit_y2, 'r-', 
                    # label=f"Edge {row_number + 1} Fit R2"
                    )
                self._plot_residual_line(self.plot_canvas_b, x_r2, y_r2, fit_y2)
                # self.plot_canvas_b.axes.legend()
                self.plot_canvas_b.draw()
        except Exception as e:
            if emit_messages:
                self._append_fit_message(f"  Edge {row_number + 1}: Region 2 fit failed - {e}")
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

            if draw_plots:
                final_canvas = self._final_fit_canvas()
                ax_r3_main, _ = self._ensure_fit_canvas_axes(final_canvas)
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
                    "rms": rms_3,
                    "baseline_params": {
                        "a0": a0_fit,
                        "b0": b0_fit,
                        "a_hkl": a_hkl_fit,
                        "b_hkl": b_hkl_fit,
                    },
                }

            # ---------- live plotting / messages (GUI) -----------
            if draw_plots:
                final_canvas = self._final_fit_canvas()
                ax_r3_main, _ = self._ensure_fit_canvas_axes(final_canvas)
                ax_r3_main.plot(
                    x_r3, func_r3(x_r3, *popt_3), "r-",
                    # label=f"Edge {hkl} Fit R3"
                )
                self._plot_residual_line(final_canvas, x_r3, y_r3, y3_fit)
                # self.plot_canvas_d.axes.legend()
                final_canvas.draw()
            if False and emit_messages:
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

            if update_table:
                if not only_update_unfixed or not fix_s:
                    self._update_cell(row_number, 8,  f"{s_fit:.4f}")
                if not only_update_unfixed or not fix_t:
                    self._update_cell(row_number, 9,  f"{t_fit:.4f}")
                if not only_update_unfixed or not fix_eta:
                    self._update_cell(row_number, 10, f"{eta_fit:.3f}")
            return {
                "hkl": hkl,
                "x": x_r3,
                "y": y_r3,
                "fit": y3_fit,
                "fit_params": (d_fit, s_fit, t_fit, eta_fit, d_unc, s_unc, t_unc, eta_unc),
                "edge_height": edge_height,
                "edge_width": edge_width,
                "rms": rms_3,
                "baseline_params": {
                    "a0": a0_fit,
                    "b0": b0_fit,
                    "a_hkl": a_hkl_fit,
                    "b_hkl": b_hkl_fit,
                },
            }

        except Exception as e:
            if emit_messages:
                self._append_fit_message(f"  Edge {row_number + 1}: Region 3 fit failed - {e}")
            if False and emit_messages:
                self.message_box.append(
                    f"Edge {row_number+1} – Region‑3 Fit Error: {e}"
                )

    def batch_fit(self):
        """
        Initiates the batch fitting process over the ROI by calling the fit_full_pattern
        function for each box in the defined grid.
        """
        if getattr(self, "fitting_data_source", "images") == "profile":
            self.message_box.append("Mapping is not available for imported intensity profiles.")
            return
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
            self.message_box.append("Working directory is not set. Please load images first.")
            return

        # Get the state of the fix flags from the Bragg table header.
        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()
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
        self._append_batch_message(text)

    def batch_fit_finished(self, filename):
        if filename:
            self._append_fit_message(f"[Batch fit] Completed\n  Results saved to {filename}")
        else:
            self._append_fit_message("[Batch fit] Completed with errors")
        self.update_timer.stop()
        self.batch_progress_bar.setValue(100)
        self.batch_remaining_time_label.setText("Remaining: ")
        self.batch_progress_dialog.hide()

    def visualize_region3_fits(self):
        return
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

        fix_s = self.fix_s_enabled()
        fix_t = self.fix_t_enabled()
        fix_eta = self.fix_eta_enabled()

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

    def show_edge_lines_enabled(self):
        """Return whether theoretical edge guide lines should be shown."""
        action = getattr(self, "show_edge_line_action", None)
        if action is not None:
            return action.isChecked()
        return bool(getattr(self, "show_edge_line", True))

    def increase_plot_font_size(self):
        """Shortcut handler to increase plot font."""
        if self.plot_font_size < 48:
            self.plot_font_size += 1
            self.update_plot_font_size(self.plot_font_size)
            if hasattr(self, "save_user_settings"):
                self.save_user_settings()

    def decrease_plot_font_size(self):
        """Shortcut handler to decrease plot font."""
        if self.plot_font_size > 6:
            self.plot_font_size -= 1
            self.update_plot_font_size(self.plot_font_size)
            if hasattr(self, "save_user_settings"):
                self.save_user_settings()

