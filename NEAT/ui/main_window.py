"""Main application window for NEAT."""

import gc
import glob
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from PyQt5 import QtWidgets
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
    QAction,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QFont
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit, least_squares

from .. import __version__
from ..core import (
    PHASE_DATA,
    calculate_theoretical_bragg_edges,
    fitting_function_1,
    fitting_function_2,
    fitting_function_3,
)
from .mixins.fitting import FittingMixin
from .mixins.postprocessing import PostProcessingMixin
from .mixins.preprocessing import PreprocessingMixin
from .utils import update_all_widget_fonts


UPDATE_API_URL = "https://api.github.com/repos/RayZhang2024/NEAT/releases/latest"
UPDATE_RELEASES_URL = "https://github.com/RayZhang2024/NEAT/releases/latest"


def _version_tuple(raw):
    """Convert a version string to a numeric tuple for comparison."""
    cleaned = str(raw or "").strip().lstrip("vV")
    match = re.search(r"\d+(?:\.\d+)*", cleaned)
    if not match:
        return tuple()
    return tuple(int(part) for part in match.group(0).split("."))


class UpdateCheckWorker(QThread):
    """Check GitHub releases in the background to avoid blocking the UI."""

    check_finished = pyqtSignal(dict)

    def __init__(self, current_version, api_url=UPDATE_API_URL, parent=None):
        super().__init__(parent)
        self.current_version = current_version
        self.api_url = api_url

    def run(self):
        try:
            request = Request(
                self.api_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "NEAT-UpdateChecker",
                },
            )
            with urlopen(request, timeout=6) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            self.check_finished.emit({"ok": False, "error": str(exc)})
            return

        latest_raw = payload.get("tag_name") or payload.get("name") or ""
        latest_version = str(latest_raw).strip()
        result = {
            "ok": True,
            "latest_version": latest_version,
            "latest_tuple": _version_tuple(latest_version),
            "current_tuple": _version_tuple(self.current_version),
            "html_url": payload.get("html_url") or UPDATE_RELEASES_URL,
        }
        result["update_available"] = bool(
            result["latest_tuple"] and result["latest_tuple"] > result["current_tuple"]
        )
        self.check_finished.emit(result)


class FitsViewer(QMainWindow, PreprocessingMixin, FittingMixin, PostProcessingMixin):
    def __init__(self):
        super().__init__()
        
        self.flight_path = 56.4
        self.flight_path_source = "App setting"
        self.current_fitting_input = ""
        self.wavelength_flight_path = None
        self.tof_axis_centers_us = None
        self.wavelength_depends_on_flight_path = False
        self.nexus_axis_centers = None
        self.nexus_axis_uses_flight_path = False
        self.delay = 0.0
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.app_version = str(__version__).strip()
        
        self.tof_array = None
        self.setMinimumSize(800, 600)
        self.setWindowTitle(f"NEAT Neutron Bragg Edge Analysis Toolkit v{self.app_version}")
        # self.setGeometry(100, 100, window_width, window_height)  # Increased size to accommodate new layout

        # Phase storage (built-ins + user-defined)
        self.custom_phases_path = os.path.join(os.path.expanduser("~"), ".neat_custom_phases.json")
        self.custom_phases = {}
        self.removed_phases = set()
        self.phase_data = dict(PHASE_DATA)
        self.load_custom_phases()

        # Create a central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create a tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self.on_main_tab_changed)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.tabs)

        # Create the Preprocessing tab
        self.PreProcessingTab = QWidget()
        self.tabs.addTab(self.PreProcessingTab, "Data Preprocessing")

        # Create the first tab (FITS Viewer)
        self.FittingTab = QWidget()
        self.tabs.addTab(self.FittingTab, "Bragg Edge Fitting")

        # Create the second tab (Data Post-Processing)
        self.PostProcessingTab = QWidget()
        self.tabs.addTab(self.PostProcessingTab, "Data Post-Processing")
        
        # Set up the layout for Preprocessing tab
        self.setup_preprocessing_tab()

        # Set up the layout for FittingTab (FITS Viewer)
        self.setup_FittingTab()
        
        # Set up the layout for PostProcessingTab (Data Post-Processing)
        self.setup_PostProcessingTab()
        self.setup_menu_bar()

        # Variables for storing image data and auto-adjust parameters
        self.images = []  # To store multiple image frames
        self.current_image_index = 0
        self.selected_area = None  # Stores (xmin, xmax, ymin, ymax)
        self.auto_vmin, self.auto_vmax = None, None
        self.current_vmin, self.current_vmax = None, None

        # Variables to store adjustment values
        self.contrast_slider_value = 100
        self.brightness_slider_value = 0
        self.min_slider_value = 0
        self.max_slider_value = 1000

        # Variables for Bragg edge data
        self.wavelengths = np.array([])
        self.intensities = []  # Summed intensities over the selected area

        # Variables for storing fitted parameters
        self.params_region1 = None
        self.params_region2 = None
        self.params_region3 = None

        # Worker threads
        self.image_load_worker = None
        self.summation_worker = None
        self.scaling_worker = None
        self.normalisation_worker = None  # For normalisation
     
        # Variables for updating the moving box
        self.current_batch_box = None  # Stores the current box during batch fitting
        self.batch_box_patch = None  # Matplotlib patch for the moving box

        # Timer for updating the display during batch fitting
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)  # Update every 100 ms
        self.update_timer.timeout.connect(self.update_display)

        # CSV data for post-processing
        self.csv_data = None
        self.metadata = {}

        # Initialize colorbar reference
        self.colorbar = None

        # Variables for Preprocessing
        self.summation_image_runs = []  # List to store summation image runs. Each run is a dict.
        self.scaling_image_runs = []     # List to store scaling image runs. Each run is a dict.
        self.normalisation_image_runs = []  # List to store normalisation data image runs. Each run is a dict.
        self.normalisation_open_beam_runs = []  # List to store normalisation open beam runs. Each run is a dict.
        self.open_beam_plot_dialogs = []  # List to keep references to open beam plot dialogs
        self.stack_image_runs = []  # List to store stack image runs. Each run is a dict.
        self.overlap_correction_image_runs = []
        self.filter_image_runs = []
        self.outlier_image_runs = []
        
        # Initialize phase-related variables
        self.hkl_list = []
        self.lattice_param = None
        self.theoretical_bragg_edges = []  # Initialize as an empty list
 
        # Initialize region wavelength variables
        self.current_r1_min = None
        self.current_r1_max = None
        self.current_r2_min = None
        self.current_r2_max = None
        self.current_r3_min = None
        self.current_r3_max = None
        
        self.manual_wavelength_mode = False
        self.manual_anchor_mode = "wavelength"  # "wavelength" or "tof"
        self.manual_wavelength_anchors = []

        self._summation_cancelled = False
        self._two_level_subfolders  = []
        self._expected_suffix_set = None      # <-- NEW
        self._expected_image_cnt  = None      # <-- NEW
        
        self.gui_font_size = 8
        self.check_updates_on_startup = True
        self.last_update_check_utc = ""
        self.ignored_update_version = ""
        self._update_check_worker = None
        self._update_check_manual = False

        self.config_path = os.path.join(os.path.expanduser("~"), ".neat_gui_settings.json")
        self.load_user_settings()
        
        # Set initial global font and Matplotlib settings
        self.setGlobalFont()
        
        QTimer.singleShot(1500, self.maybe_check_for_updates_on_startup)

    def setGlobalFont(self):
        # Update the global font for the QApplication
        global_font = QFont("Arial", self.gui_font_size)
        QApplication.instance().setFont(global_font)
        
        # Update Matplotlib parameters to match the new font size
        mpl.rcParams['font.size'] = 2*self.gui_font_size
        mpl.rcParams['axes.labelsize'] = 2*self.gui_font_size
        mpl.rcParams['xtick.labelsize'] = 2*self.gui_font_size
        mpl.rcParams['ytick.labelsize'] = 2*self.gui_font_size
        mpl.rcParams['legend.fontsize'] = 2*self.gui_font_size
        # mpl.rcParams['figure.dpi'] = 100 * self.scale_factor
        # Update the font for all existing widgets
        update_all_widget_fonts(global_font)
        message_box = getattr(self, "message_box", None)
        if message_box is not None:
            message_box.setFont(global_font)
            message_box.setStyleSheet(
                "QTextEdit { font-family: Arial; font-size: %dpt; }"
                % int(self.gui_font_size)
            )

    def increaseFontSize(self):
        self.gui_font_size += 1
        self.setGlobalFont()
        self.save_user_settings()
        print("Font size increased to", self.gui_font_size)

    def decreaseFontSize(self):
        # Prevent font size from going below a reasonable minimum (e.g., 1)
        if self.gui_font_size > 1:
            self.gui_font_size -= 1
            self.setGlobalFont()
            self.save_user_settings()
            print("Font size decreased to", self.gui_font_size)

    def setup_menu_bar(self):
        """Create a context-sensitive menu bar for the active workflow tab."""
        self._menu_bar_ready = True
        self.refresh_menu_bar_for_current_tab()

    def refresh_menu_bar_for_current_tab(self):
        """Rebuild the native menu bar so it matches the active top-level tab."""
        if not getattr(self, "_menu_bar_ready", False):
            return

        menu_bar = self.menuBar()
        menu_bar.clear()
        self.show_edge_line_action = None
        self.live_fit_preview_action = None
        self.fitting_plot_layout_action = None
        self.update_check_startup_action = None

        current_widget = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        if current_widget is self.FittingTab:
            self._build_fitting_menu_bar(menu_bar)
        elif current_widget is self.PostProcessingTab:
            self._build_postprocessing_menu_bar(menu_bar)
        else:
            self._build_preprocessing_menu_bar(menu_bar)

        self._add_about_menu(menu_bar)

    def _build_preprocessing_menu_bar(self, menu_bar):
        file_menu = menu_bar.addMenu("File")
        clear_loaded_images_action = QAction("Clear Loaded Images", self)
        clear_loaded_images_action.triggered.connect(self.clear_loaded_preprocessing_images)
        file_menu.addAction(clear_loaded_images_action)

        clear_messages_action = QAction("Clear Messages", self)
        clear_messages_action.triggered.connect(self.clear_messages)
        file_menu.addAction(clear_messages_action)
        self._add_exit_action(file_menu)

        view_menu = menu_bar.addMenu("View")
        self._add_ui_font_actions(view_menu)

    def _build_fitting_menu_bar(self, menu_bar):
        file_menu = menu_bar.addMenu("File")
        load_images_action = QAction("Load MCP images", self)
        load_images_action.setShortcut("Ctrl+O")
        load_images_action.triggered.connect(self.load_fits_images)
        file_menu.addAction(load_images_action)

        load_nexus_action = QAction("Load Nexus images", self)
        load_nexus_action.triggered.connect(self.open_nexus_image_stack_from_menu)
        file_menu.addAction(load_nexus_action)

        load_raden_action = QAction("Load Raden images", self)
        load_raden_action.triggered.connect(self.open_raden_tiff_stack_from_menu)
        file_menu.addAction(load_raden_action)

        import_profile_action = QAction("Load line profile", self)
        import_profile_action.triggered.connect(self.open_import_intensity_profile_from_menu)
        file_menu.addAction(import_profile_action)

        export_profile_action = QAction("Export line profile", self)
        export_profile_action.triggered.connect(self.export_data)
        file_menu.addAction(export_profile_action)

        load_fitting_config_action = QAction("Load fitting configuration", self)
        load_fitting_config_action.triggered.connect(self.open_load_fitting_configuration_from_menu)
        file_menu.addAction(load_fitting_config_action)

        save_fitting_config_action = QAction("Export fitting configuration", self)
        save_fitting_config_action.triggered.connect(self.open_save_fitting_configuration_from_menu)
        file_menu.addAction(save_fitting_config_action)

        clear_images_action = QAction("Clear images", self)
        clear_images_action.triggered.connect(self.clear_loaded_fits_images)
        file_menu.addAction(clear_images_action)

        self._add_exit_action(file_menu)

        setting_menu = menu_bar.addMenu("Setting")
        fitting_config_action = QAction("Instrument Setting...", self)
        fitting_config_action.triggered.connect(self.open_fitting_config_from_menu)
        setting_menu.addAction(fitting_config_action)

        manual_spectra_action = QAction("Manual Spectra Setting...", self)
        manual_spectra_action.triggered.connect(self.open_manual_spectra_setting_from_menu)
        setting_menu.addAction(manual_spectra_action)

        phase_management_action = QAction("Phase Management...", self)
        phase_management_action.triggered.connect(self.open_phase_management_from_menu)
        setting_menu.addAction(phase_management_action)

        view_menu = menu_bar.addMenu("View")
        self.show_edge_line_action = QAction("Show Edge Line", self)
        self.show_edge_line_action.setCheckable(True)
        self.show_edge_line_action.setChecked(bool(getattr(self, "show_edge_line", True)))
        self.show_edge_line_action.toggled.connect(self.on_show_edge_line_toggled)
        view_menu.addAction(self.show_edge_line_action)

        self.live_fit_preview_action = QAction("Live Fit Preview", self)
        self.live_fit_preview_action.setCheckable(True)
        self.live_fit_preview_action.setChecked(
            bool(getattr(self, "live_fit_preview_enabled", True))
        )
        self.live_fit_preview_action.toggled.connect(self.on_live_fit_preview_toggled)
        view_menu.addAction(self.live_fit_preview_action)

        self.fitting_plot_layout_action = QAction("Four Fit Canvases", self)
        self.fitting_plot_layout_action.setCheckable(True)
        self.fitting_plot_layout_action.setChecked(
            getattr(self, "fitting_plot_layout_mode", "single") == "quad"
        )
        self.fitting_plot_layout_action.toggled.connect(self.on_fitting_plot_layout_toggled)
        view_menu.addAction(self.fitting_plot_layout_action)
        view_menu.addSeparator()

        auto_adjust_action = QAction("Auto Adjust", self)
        auto_adjust_action.triggered.connect(self.open_auto_adjust_from_menu)
        view_menu.addAction(auto_adjust_action)

        manual_adjust_action = QAction("Manual Adjust...", self)
        manual_adjust_action.triggered.connect(self.open_manual_adjust_from_menu)
        view_menu.addAction(manual_adjust_action)
        view_menu.addSeparator()

        symbol_size_action = QAction("Symbol Size...", self)
        symbol_size_action.triggered.connect(self.open_symbol_size_dialog)
        view_menu.addAction(symbol_size_action)

        self._add_ui_font_actions(view_menu)
        view_menu.addSeparator()
        self._add_canvas_font_actions(view_menu)

    def _build_postprocessing_menu_bar(self, menu_bar):
        file_menu = menu_bar.addMenu("File")
        load_csv_action = QAction("Load Post-processing CSV...", self)
        load_csv_action.setShortcut("Ctrl+O")
        load_csv_action.triggered.connect(self.open_postprocessing_csv_from_menu)
        file_menu.addAction(load_csv_action)

        load_image_action = QAction("Load Image Result...", self)
        load_image_action.triggered.connect(self.open_image_result_from_menu)
        file_menu.addAction(load_image_action)

        clear_results_action = QAction("Clear Results", self)
        clear_results_action.triggered.connect(self.clear_postprocessing_results_from_menu)
        file_menu.addAction(clear_results_action)
        self._add_exit_action(file_menu)

        view_menu = menu_bar.addMenu("View")
        self._add_ui_font_actions(view_menu)

    def _add_ui_font_actions(self, menu):
        increase_font_action = QAction("Increase UI Font", self)
        increase_font_action.setShortcut("Shift+Up")
        increase_font_action.triggered.connect(self.increaseFontSize)
        menu.addAction(increase_font_action)

        decrease_font_action = QAction("Decrease UI Font", self)
        decrease_font_action.setShortcut("Shift+Down")
        decrease_font_action.triggered.connect(self.decreaseFontSize)
        menu.addAction(decrease_font_action)

    def _add_canvas_font_actions(self, menu):
        increase_plot_font_action = QAction("Increase Canvas Font", self)
        increase_plot_font_action.setShortcut("Ctrl+Up")
        increase_plot_font_action.triggered.connect(self.increase_plot_font_size)
        menu.addAction(increase_plot_font_action)

        decrease_plot_font_action = QAction("Decrease Canvas Font", self)
        decrease_plot_font_action.setShortcut("Ctrl+Down")
        decrease_plot_font_action.triggered.connect(self.decrease_plot_font_size)
        menu.addAction(decrease_plot_font_action)

    def _add_exit_action(self, menu):
        menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _add_about_menu(self, menu_bar):
        about_menu = menu_bar.addMenu("About")
        about_action = QAction("About NEAT", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

        self.update_check_startup_action = QAction("Check for Updates on Startup", self)
        self.update_check_startup_action.setCheckable(True)
        self.update_check_startup_action.setChecked(
            bool(getattr(self, "check_updates_on_startup", True))
        )
        self.update_check_startup_action.toggled.connect(self.on_update_check_startup_toggled)
        about_menu.addAction(self.update_check_startup_action)

        check_updates_action = QAction("Check for Updates Now", self)
        check_updates_action.triggered.connect(self.check_for_updates_now)
        about_menu.addAction(check_updates_action)
        about_menu.addSeparator()

        user_manual_action = QAction("User Manual", self)
        user_manual_action.triggered.connect(self.open_user_manual)
        about_menu.addAction(user_manual_action)

        github_action = QAction("GitHub Repository", self)
        github_action.triggered.connect(self.open_github_repository)
        about_menu.addAction(github_action)

        video_action = QAction("Video Tutorials", self)
        video_action.triggered.connect(self.open_video_tutorials)
        about_menu.addAction(video_action)

    def open_postprocessing_csv_from_menu(self):
        self.tabs.setCurrentWidget(self.PostProcessingTab)
        self.load_csv_file()

    def open_image_result_from_menu(self):
        self.tabs.setCurrentWidget(self.PostProcessingTab)
        self.load_image_results()

    def clear_postprocessing_results_from_menu(self):
        self.tabs.setCurrentWidget(self.PostProcessingTab)
        self.clear_postprocessing_results()

    def open_fitting_config_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.open_general_settings_dialog()

    def open_manual_spectra_setting_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.open_manual_spectra_settings_dialog()

    def open_load_fitting_configuration_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.load_fitting_configuration_from_csv()

    def open_save_fitting_configuration_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.save_fitting_configuration_to_csv()

    def open_nexus_image_stack_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.load_nexus_image_stack()

    def open_raden_tiff_stack_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.load_raden_tiff_stack()

    def open_import_intensity_profile_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.import_intensity_profile()

    def open_auto_adjust_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.auto_adjust()

    def open_manual_adjust_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.open_adjustments_dialog()

    def open_phase_management_from_menu(self):
        self.tabs.setCurrentWidget(self.FittingTab)
        self.open_add_phase_dialog()

    def open_symbol_size_dialog(self):
        value, ok = QInputDialog.getInt(
            self,
            "Symbol Size",
            "Symbol Size:",
            int(getattr(self, "symbol_size", 4)),
            1,
            20,
            1,
        )
        if not ok:
            return
        self.symbol_size = value
        self.apply_symbol_size()
        if getattr(self, "intensities", None) is not None and len(self.intensities) > 0:
            self.update_plots()
        self.save_user_settings()

    def open_user_manual(self):
        QDesktopServices.openUrl(
            QUrl("https://github.com/RayZhang2024/NEAT/blob/main/User%20manual.md")
        )

    def open_github_repository(self):
        QDesktopServices.openUrl(QUrl("https://github.com/RayZhang2024/NEAT"))

    def open_video_tutorials(self):
        QDesktopServices.openUrl(QUrl("https://www.youtube.com/@RayOnNeutrons"))

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About NEAT",
            (
                f"<h2>NEAT Neutron Bragg Edge Analysis Toolkit v{self.app_version}</h2>"
                "<p><b>Developed by:</b><br>"
                "Engineering and imaging group<br>"
                "ISIS Neutron and Muon Source<br>"
                "Rutherford Appleton Laboratory</p>"
                "<p><b>Main authors:</b><br>"
                "Ruiyao Zhang (ruiyao.zhang@stfc.ac.uk)<br>"
                "Ranggi Ramadhan (ranggi.ramadhan@stfc.ac.uk)</p>"
                "<p><b>Project resources:</b><br>"
                "Use the About menu to open the user manual, GitHub repository, "
                "video tutorials, or check for updates.</p>"
            ),
        )

    def sync_update_check_controls(self):
        checked = bool(getattr(self, "check_updates_on_startup", True))
        for control in (
            getattr(self, "update_check_startup_checkbox", None),
            getattr(self, "update_check_startup_action", None),
        ):
            if control is None:
                continue
            control.blockSignals(True)
            control.setChecked(checked)
            control.blockSignals(False)

    def sync_show_edge_line_controls(self):
        checked = bool(getattr(self, "show_edge_line", True))
        action = getattr(self, "show_edge_line_action", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def sync_live_fit_preview_controls(self):
        checked = bool(getattr(self, "live_fit_preview_enabled", True))
        action = getattr(self, "live_fit_preview_action", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def sync_fitting_plot_layout_controls(self):
        checked = getattr(self, "fitting_plot_layout_mode", "single") == "quad"
        action = getattr(self, "fitting_plot_layout_action", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def get_edges_in_range(self, min_wavelength, max_wavelength):
        """
        Returns a list of theoretical Bragg edges within the specified wavelength range.
        Each entry is a tuple: ((h, k, l), x_hkl)
        
        Parameters:
            min_wavelength (float): The minimum wavelength of the range.
            max_wavelength (float): The maximum wavelength of the range.
        
        Returns:
            list of tuples: Each tuple contains ((h, k, l), x_hkl) for edges within the range.
        """
        if not self.theoretical_bragg_edges:
            self.message_box.append("No theoretical Bragg edges available.")
            return []
        
        edges_in_range = [
            (hkl, x_hkl) for (hkl, x_hkl) in self.theoretical_bragg_edges
            if min_wavelength <= x_hkl <= max_wavelength and not np.isnan(x_hkl)
        ]
        
         
        return edges_in_range

        
    def phase_selection_changed(self, index):
        selected_phase = self.phase_dropdown.currentText()
        self.current_phase = selected_phase
        placeholder = getattr(self, "phase_placeholder", "Select Phase")
        if selected_phase not in (placeholder, "Phase"):
            self.message_box.append(f"Selected phase: {selected_phase}")
    
            phase_info = self.phase_data.get(selected_phase)
            if phase_info:
                if selected_phase == "Unknown_Phase":
                    self.structure_type = "unknown"
                    self.lattice_params = {}
                    self.hkl_list = []
                    self.theoretical_bragg_edges = []
                    self.setup_unknown_phase_table()
                    self.update_plots()
                else:
                    self.structure_type = phase_info.get("structure", "fcc")
                    self.lattice_params = phase_info.get("lattice_params", {})
                    self.hkl_list = phase_info.get("hkl_list", [])
                    self.update_plots()
            else:
                self.message_box.append(f"No data found for phase: {selected_phase}")
                self.structure_type = "unknown"
                self.lattice_params = {}
                self.hkl_list = []
                self.theoretical_bragg_edges = []
                self.update_plots()
        else:
            self.message_box.append("No phase selected.")
            self.structure_type = "unknown"
            self.lattice_params = {}
            self.hkl_list = []
            self.theoretical_bragg_edges = []
            self.update_plots()


    def setup_unknown_phase_table(self):
        """
        Setup the Bragg edges table for 'Unknown Phase' selection.
        Allows user to input Bragg edge values manually, up to 5 rows.
        """
        self.bragg_table.setRowCount(0)  # Clear existing rows
        max_rows = 5  # Limit to 5 rows
    
        for _ in range(max_rows):
            row_position = self.bragg_table.rowCount()
            self.bragg_table.insertRow(row_position)
    
            # hkl column - read-only or display as 'N/A'
            hkl_item = QTableWidgetItem("N/A")
            hkl_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 0, hkl_item)
    
            # d column - Editable for Unknown Phase
            d_item = QTableWidgetItem("")
            d_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 1, d_item)
    
            # Region 1 Min Wavelength - Editable
            r1_min_item = QTableWidgetItem("")
            r1_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 4, r1_min_item)
    
            # Region 1 Max Wavelength - Editable
            r1_max_item = QTableWidgetItem("")
            r1_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 5, r1_max_item)
    
            # Region 2 Min Wavelength - Editable
            r2_min_item = QTableWidgetItem("")
            r2_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 2, r2_min_item)
    
            # Region 2 Max Wavelength - Editable
            r2_max_item = QTableWidgetItem("")
            r2_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 3, r2_max_item)
    
            # Region 3 Min Wavelength - Editable
            r3_min_item = QTableWidgetItem("")
            r3_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 6, r3_min_item)
    
            # Region 3 Max Wavelength - Editable
            r3_max_item = QTableWidgetItem("")
            r3_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 7, r3_max_item)
    
            # Parameter 's' - Editable
            s_item = QTableWidgetItem("")
            s_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 8, s_item)
    
            # Parameter 't' - Editable
            t_item = QTableWidgetItem("")
            t_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 9, t_item)
            
            # Parameter 'η' - Editable
            eta_item = QTableWidgetItem("")
            eta_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 10, eta_item)
    
        self.message_box.append("Configured table for 'Unknown Phase'. Please input Bragg edge values.")
   
    def compute_theoretical_bragg_edges(self):
        """
        Compute theoretical Bragg edge positions (x_hkl) based on the selected phase's
        structure, lattice parameters, and hkl_list. Each x_hkl is paired with its
        corresponding (h, k, l) index.
        """
        if not self.hkl_list or not self.lattice_params:
            self.message_box.append("Incomplete phase data. Cannot compute Bragg edges.")
            self.theoretical_bragg_edges = []
            return

        self.theoretical_bragg_edges = calculate_theoretical_bragg_edges(
            self.structure_type,
            self.lattice_params,
            self.hkl_list,
        )
        for hkl, x_hkl in self.theoretical_bragg_edges:
            if np.isnan(x_hkl) or x_hkl <= 0:
                self.message_box.append(f"Skipping hkl{hkl}: invalid or undefined d-spacing.")
            else:
                self.message_box.append(f"Theoretical Bragg edge for hkl{hkl}: {x_hkl:.4f} A")


    def _reset_current_bragg_edge_state(self):
        """Clear cached values for the currently active Bragg edge row."""
        self.current_d = None
        self.current_r1_min = None
        self.current_r1_max = None
        self.current_r2_min = None
        self.current_r2_max = None
        self.current_r3_min = None
        self.current_r3_max = None
        self.current_s = None
        self.current_t = None
        self.current_eta = None

    def _sync_current_bragg_edge_from_row(self, row_index):
        """
        Copy Bragg-edge values from a table row into cached current_* attributes.
        Returns True when the row can be parsed, otherwise False.
        """
        if row_index < 0 or row_index >= self.bragg_table.rowCount():
            return False

        try:
            d_item = self.bragg_table.item(row_index, 1)
            d_text = d_item.text().strip() if d_item else ""
            d = float(d_text) if d_text and d_text not in ("N/A", "") else None

            r1_min = float(self.bragg_table.item(row_index, 4).text())
            r1_max = float(self.bragg_table.item(row_index, 5).text())
            r2_min = float(self.bragg_table.item(row_index, 2).text())
            r2_max = float(self.bragg_table.item(row_index, 3).text())
            r3_min = float(self.bragg_table.item(row_index, 6).text())
            r3_max = float(self.bragg_table.item(row_index, 7).text())
            s = float(self.bragg_table.item(row_index, 8).text())
            t = float(self.bragg_table.item(row_index, 9).text())
            eta = float(self.bragg_table.item(row_index, 10).text())
        except (ValueError, AttributeError, TypeError):
            return False

        self.current_d = d
        self.current_r1_min = r1_min
        self.current_r1_max = r1_max
        self.current_r2_min = r2_min
        self.current_r2_max = r2_max
        self.current_r3_min = r3_min
        self.current_r3_max = r3_max
        self.current_s = s
        self.current_t = t
        self.current_eta = eta
        return True

    def update_bragg_edge_table(self):
        """
        Updates the Bragg edges table based on the selected phase and wavelength range.
        For 'Unknown_Phase', allows manual input up to 5 rows.
        Otherwise, computes theoretical edges (including non-cubic) and populates the table.
        """
        prev_hkl_text = None
        selected_row = self.bragg_table.currentRow()
        if selected_row >= 0:
            selected_hkl_item = self.bragg_table.item(selected_row, 0)
            if selected_hkl_item is not None:
                prev_hkl_text = selected_hkl_item.text().strip()

        # Only populate after user explicitly picks a region.
        if not getattr(self, "_bragg_table_ready", False):
            self._reset_current_bragg_edge_state()
            self.bragg_table.setRowCount(0)
            return

        selected_phase = self.phase_dropdown.currentText()
        if selected_phase == "Unknown_Phase":
            self._reset_current_bragg_edge_state()
            self.setup_unknown_phase_table()
            return  # Exit early as manual input is handled separately
    
        if selected_phase not in self.phase_data:
            self.message_box.append("Please select a valid phase.")
            self._reset_current_bragg_edge_state()
            self.bragg_table.setRowCount(0)  # Clear table if invalid phase
            return
    
        # Validate the min/max wavelength inputs
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
            if min_wavelength >= max_wavelength:
                QMessageBox.warning(self, "Invalid Wavelength Range",
                                    "Minimum wavelength must be less than Maximum wavelength.")
                self._reset_current_bragg_edge_state()
                self.bragg_table.setRowCount(0)
                return
        except ValueError:
            self.message_box.append("Please enter valid min and max wavelengths.")
            self._reset_current_bragg_edge_state()
            self.bragg_table.setRowCount(0)
            return
    
        # ------------------------------
        # Retrieve Phase Info & Compute
        # ------------------------------
        phase_info = self.phase_data[selected_phase]
        # For non-cubic phases, we might have multiple lattice params
        self.structure_type = phase_info.get("structure", "cubic")
        self.lattice_params = phase_info.get("lattice_params", {})
        self.hkl_list = phase_info.get("hkl_list", [])
    
        # Make sure we compute the edges if not already done
        # (Your compute_theoretical_bragg_edges should handle any structure.)
        self.compute_theoretical_bragg_edges()
    
        # Now filter edges based on the chosen wavelength range
        edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength)
    
        # ------------------------------
        # Populate Table
        # ------------------------------
        self.bragg_table.setRowCount(0)  # Clear existing rows
    
        if not edges_in_range:
            self._reset_current_bragg_edge_state()
            self.message_box.append("No Bragg edges found within the specified wavelength range.")
            return
    
        # Insert each edge as a new row in the table
        for (hkl, x_hkl) in edges_in_range:
            row_position = self.bragg_table.rowCount()
            self.bragg_table.insertRow(row_position)
    
            # hkl column - Read-only
            hkl_item = QTableWidgetItem(str(hkl))
            hkl_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 0, hkl_item)
    
            # ---------------------------------------------
            # d column - For known phases, we can compute d
            #           d = x_hkl / 2, so let's store that.
            # ---------------------------------------------
            if x_hkl is not None and not np.isnan(x_hkl):
                d_val = x_hkl / 2.0
                d_item = QTableWidgetItem(f"{d_val:.4f}")
            else:
                d_item = QTableWidgetItem("N/A")
    
            d_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 1, d_item)
    
            # ---------------------------------------------
            # Region 1 (min/max), Region 2 (min/max), ...
            # Arbitrary logic for how you set defaults:
            # region 1 might be a small offset above x_hkl, etc.
            # ---------------------------------------------
            r1_min = x_hkl * 1.04
            r1_min_item = QTableWidgetItem(f"{r1_min:.2f}")
            r1_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 4, r1_min_item)
    
            r1_max = min(x_hkl * 1.12, x_hkl + 0.4)
            r1_max_item = QTableWidgetItem(f"{r1_max:.2f}")
            r1_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 5, r1_max_item)
    
            # Region 2
            r2_min = max(x_hkl * 0.90, x_hkl - 0.3)
            r2_min_item = QTableWidgetItem(f"{r2_min:.2f}")
            r2_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 2, r2_min_item)
    
            r2_max = x_hkl * 0.98
            r2_max_item = QTableWidgetItem(f"{r2_max:.2f}")
            r2_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 3, r2_max_item)
    
            # Region 3
            r3_min = r2_min
            r3_min_item = QTableWidgetItem(f"{r3_min:.2f}")
            r3_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 6, r3_min_item)
    
            r3_max = r1_max
            r3_max_item = QTableWidgetItem(f"{r3_max:.2f}")
            r3_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 7, r3_max_item)
    
            # s parameter
            s = 0.001
            s_item = QTableWidgetItem(f"{s:.4f}")
            s_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 8, s_item)
    
            # t parameter
            t = 0.01
            t_item = QTableWidgetItem(f"{t:.4f}")
            t_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 9, t_item)
            
            # eta parameter
            eta = 0.5
            eta_item = QTableWidgetItem(f"{eta:.3f}")
            eta_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 10, eta_item)
    
        # If we got here, we successfully populated the table.
        target_row = 0
        if prev_hkl_text:
            for row in range(self.bragg_table.rowCount()):
                item = self.bragg_table.item(row, 0)
                if item and item.text().strip() == prev_hkl_text:
                    target_row = row
                    break

        self.bragg_table.blockSignals(True)
        self.bragg_table.clearSelection()
        self.bragg_table.setCurrentCell(target_row, 0)
        self.bragg_table.selectRow(target_row)
        self.bragg_table.blockSignals(False)
        if not self._sync_current_bragg_edge_from_row(target_row):
            self._reset_current_bragg_edge_state()

        self.message_box.append(
            f"Updated Bragg edges table with {self.bragg_table.rowCount()} edge(s)."
        )


    def on_bragg_edge_selected(self):
        """
        Handler called when a Bragg edge row is selected in the table.
        Updates the region min/max wavelength inputs, 'd', s, t, etc.
        """
        selected_row = self.bragg_table.currentRow()
        if selected_row < 0:
            selected_items = self.bragg_table.selectedItems()
            if not selected_items:
                return  # No selection
            selected_row = selected_items[0].row()

        if not self._sync_current_bragg_edge_from_row(selected_row):
            self.message_box.append("Error reading values from selected row.")
            return

        # Refresh plots immediately when a row is selected (if data is available)
        if getattr(self, "images", None) is not None and getattr(self, "selected_area", None):
            self.update_plots()

    def cleanup_resources(self):
        """Release images, workers, and temporary handles before exit."""
        worker_attrs = [
            "fits_image_load_worker",
            "image_load_worker",
            "summation_worker",
            "scaling_worker",
            "normalisation_worker",
            "open_beam_load_worker",
            "outlier_worker",
            "overlap_worker",
            "filtering_worker",
            "batch_fit_worker",
            "batch_fit_edges_worker",
            "full_process_worker",
        ]
        for name in worker_attrs:
            worker = getattr(self, name, None)
            if worker is None:
                continue
            try:
                if hasattr(worker, "stop"):
                    worker.stop()
            except Exception:
                pass
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(1000)
            setattr(self, name, None)

        self.images = []
        self.intensities = np.array([])
        self.tof_array = None
        self.wavelengths = np.array([])
        if hasattr(self, "image_slider"):
            self.image_slider.setEnabled(False)
        self.display_image()
        self.manual_wavelength_mode = False

    def rebuild_phase_data(self):
        """Rebuild the combined phase dictionary from built-ins, removals, and customs."""
        base = {k: v for k, v in PHASE_DATA.items() if k not in self.removed_phases}
        base.update(self.custom_phases)
        self.phase_data = base

    def load_custom_phases(self):
        """Load user-defined phases (and deletions) from disk into memory."""
        if not os.path.exists(self.custom_phases_path):
            self.rebuild_phase_data()
            return
        try:
            with open(self.custom_phases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.rebuild_phase_data()
            return

        phases_payload = {}
        removed_payload = set()
        if isinstance(data, dict) and ("phases" in data or "removed" in data):
            phases_payload = data.get("phases", {})
            removed_payload = set(data.get("removed", []))
        elif isinstance(data, dict):
            # Backward compatibility: old format stored phases directly.
            phases_payload = data

        valid = {
            name: info
            for name, info in phases_payload.items()
            if isinstance(info, dict)
            and "structure" in info
            and "lattice_params" in info
            and "hkl_list" in info
        }
        self.custom_phases = valid
        self.removed_phases = removed_payload
        self.rebuild_phase_data()

    def save_custom_phases(self):
        """Persist user-defined phases and removed built-ins to disk."""
        try:
            payload = {
                "phases": self.custom_phases,
                "removed": sorted(self.removed_phases),
            }
            with open(self.custom_phases_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            if hasattr(self, "message_box"):
                self.message_box.append("Could not save custom phases to disk.")

    def register_custom_phase(self, name, phase_definition):
        """Add or update a user-defined phase and refresh the UI list."""
        self.custom_phases[name] = phase_definition
        if name in self.removed_phases:
            self.removed_phases.discard(name)
        self.rebuild_phase_data()
        self.save_custom_phases()
        if hasattr(self, "refresh_phase_dropdown"):
            self.refresh_phase_dropdown(select=name)
        if hasattr(self, "message_box"):
            self.message_box.append(f"Added custom phase '{name}'.")

    def delete_phase(self, name):
        """Delete a phase (custom or built-in override) and refresh the UI list."""
        placeholder = getattr(self, "phase_placeholder", "Select Phase")
        if name in (placeholder, "Phase", "Unknown_Phase"):
            if hasattr(self, "message_box"):
                self.message_box.append(f"'{name}' cannot be deleted.")
            return False

        removed_any = False
        if name in self.custom_phases:
            del self.custom_phases[name]
            removed_any = True
        if name in PHASE_DATA and name not in self.removed_phases:
            self.removed_phases.add(name)
            removed_any = True
        if name in self.phase_data:
            self.phase_data.pop(name, None)
            removed_any = True

        if removed_any:
            self.rebuild_phase_data()
            self.save_custom_phases()
            if hasattr(self, "refresh_phase_dropdown"):
                self.refresh_phase_dropdown(select=placeholder)
            if hasattr(self, "message_box"):
                self.message_box.append(f"Deleted phase '{name}'.")
        else:
            if hasattr(self, "message_box"):
                self.message_box.append(f"No phase named '{name}' to delete.")
        return removed_any

    def _parse_utc_timestamp(self, value):
        """Parse ISO-8601 timestamp and return UTC datetime, or None."""
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _checked_for_updates_recently(self, hours=24):
        """Return True if update check was performed within the last window."""
        last_check = self._parse_utc_timestamp(self.last_update_check_utc)
        if last_check is None:
            return False
        return (datetime.now(timezone.utc) - last_check) < timedelta(hours=hours)

    def maybe_check_for_updates_on_startup(self):
        """Startup hook for passive update checks."""
        if not self.check_updates_on_startup:
            return
        if self._checked_for_updates_recently(hours=24):
            return
        self.start_update_check(manual=False)

    def on_update_check_startup_toggled(self, checked):
        """Apply preference from About tab checkbox."""
        self.check_updates_on_startup = bool(checked)
        self.sync_update_check_controls()
        self.save_user_settings()

    def on_show_edge_line_toggled(self, checked):
        """Apply theoretical edge line visibility from the Setting menu."""
        self.show_edge_line = bool(checked)
        self.sync_show_edge_line_controls()
        if hasattr(self, "update_plots"):
            self.update_plots()
        self.save_user_settings()

    def on_live_fit_preview_toggled(self, checked):
        """Apply live fit preview visibility from the View menu."""
        self.live_fit_preview_enabled = bool(checked)
        self.sync_live_fit_preview_controls()
        self.save_user_settings()

    def on_fitting_plot_layout_toggled(self, checked):
        """Apply fitting plot layout from the View menu."""
        mode = "quad" if checked else "single"
        if hasattr(self, "set_fitting_plot_layout_mode"):
            self.set_fitting_plot_layout_mode(mode, refresh_plots=True)
        else:
            self.fitting_plot_layout_mode = mode
            self.sync_fitting_plot_layout_controls()
        self.save_user_settings()

    def on_main_tab_changed(self, index):
        """Track the active top-level tab for next launch."""
        self.last_open_tab_index = int(index)
        self.refresh_menu_bar_for_current_tab()

    def check_for_updates_now(self):
        """Manual update-check action from UI."""
        self.start_update_check(manual=True)

    def start_update_check(self, manual=False):
        """Run a background check against GitHub latest release."""
        if self._update_check_worker is not None and self._update_check_worker.isRunning():
            if manual and hasattr(self, "message_box"):
                self.message_box.append("Update check is already running.")
            return

        self._update_check_manual = bool(manual)
        if manual and hasattr(self, "message_box"):
            self.message_box.append("Checking for updates...")

        worker = UpdateCheckWorker(current_version=self.app_version, parent=self)
        worker.check_finished.connect(self._on_update_check_finished)
        worker.finished.connect(worker.deleteLater)
        self._update_check_worker = worker
        worker.start()

    def _show_update_available_dialog(self, latest_version, release_url):
        """Show update prompt with open/skip/later actions."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("Update Available")
        dialog.setText(f"A newer NEAT version is available: {latest_version}")
        dialog.setInformativeText(
            f"Current version: {self.app_version}\nOpen the release download page now?"
        )

        open_btn = dialog.addButton("Open Download Page", QMessageBox.AcceptRole)
        skip_btn = dialog.addButton("Skip This Version", QMessageBox.RejectRole)
        dialog.addButton("Later", QMessageBox.ActionRole)
        dialog.setDefaultButton(open_btn)
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked is open_btn:
            QDesktopServices.openUrl(QUrl(release_url))
            self.ignored_update_version = ""
        elif clicked is skip_btn:
            self.ignored_update_version = str(latest_version).strip()

    def _on_update_check_finished(self, result):
        """Handle completion of background update check."""
        manual = self._update_check_manual
        self._update_check_manual = False
        self._update_check_worker = None
        self.last_update_check_utc = datetime.now(timezone.utc).isoformat()

        if not result.get("ok"):
            error_text = result.get("error", "Unknown update-check error")
            if manual:
                QMessageBox.warning(self, "Update Check Failed", error_text)
            if hasattr(self, "message_box"):
                self.message_box.append(f"Update check failed: {error_text}")
            self.save_user_settings()
            return

        latest_version = str(result.get("latest_version", "")).strip()
        release_url = result.get("html_url") or UPDATE_RELEASES_URL
        update_available = bool(result.get("update_available"))

        if not update_available:
            if manual:
                QMessageBox.information(
                    self,
                    "No Updates",
                    f"NEAT is up to date (v{self.app_version}).",
                )
                if hasattr(self, "message_box"):
                    self.message_box.append("No updates available.")
            self.save_user_settings()
            return

        if (not manual) and latest_version and latest_version == self.ignored_update_version:
            self.save_user_settings()
            return

        self._show_update_available_dialog(latest_version, release_url)
        self.save_user_settings()

    def load_user_settings(self):
        """Load persisted GUI settings if available."""
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        self.gui_font_size = data.get("gui_font_size", self.gui_font_size)
        self.setGlobalFont()

        plot_font = data.get("plot_font_size")
        if plot_font and hasattr(self, "plot_font_size"):
            self.plot_font_size = plot_font
            self.update_plot_font_size(self.plot_font_size)

        self.flight_path = data.get("flight_path", self.flight_path)
        self.delay = data.get("delay", self.delay)

        phase = data.get("phase")
        if phase:
            idx = self.phase_dropdown.findText(phase)
            if idx >= 0:
                self.phase_dropdown.setCurrentIndex(idx)

        roi = data.get("initial_roi", {})
        self.xmin_input.setText(str(roi.get("xmin", self.xmin_input.text())))
        self.xmax_input.setText(str(roi.get("xmax", self.xmax_input.text())))
        self.ymin_input.setText(str(roi.get("ymin", self.ymin_input.text())))
        self.ymax_input.setText(str(roi.get("ymax", self.ymax_input.text())))

        wavelength = data.get("wavelength_window", {})
        self.min_wavelength_input.setText(str(wavelength.get("min", self.min_wavelength_input.text())))
        self.max_wavelength_input.setText(str(wavelength.get("max", self.max_wavelength_input.text())))

        self.set_fix_parameter_state("s", data.get("fix_s", self.fix_s_enabled()))
        self.set_fix_parameter_state("t", data.get("fix_t", self.fix_t_enabled()))
        self.set_fix_parameter_state("eta", data.get("fix_eta", self.fix_eta_enabled()))
        self.symbol_size = data.get("symbol_size", getattr(self, "symbol_size", 4))
        if hasattr(self, "apply_symbol_size"):
            self.apply_symbol_size()
        self.show_edge_line = bool(data.get("show_edge_line", getattr(self, "show_edge_line", True)))
        self.sync_show_edge_line_controls()
        self.live_fit_preview_enabled = bool(
            data.get("live_fit_preview_enabled", getattr(self, "live_fit_preview_enabled", True))
        )
        self.sync_live_fit_preview_controls()
        fitting_plot_layout_mode = data.get(
            "fitting_plot_layout_mode",
            getattr(self, "fitting_plot_layout_mode", "single"),
        )
        if hasattr(self, "set_fitting_plot_layout_mode"):
            self.set_fitting_plot_layout_mode(fitting_plot_layout_mode, refresh_plots=False)
        else:
            self.fitting_plot_layout_mode = fitting_plot_layout_mode
            self.sync_fitting_plot_layout_controls()
        if hasattr(self, "tabs"):
            tab_index = int(data.get("last_open_tab_index", getattr(self, "last_open_tab_index", 0)))
            tab_index = max(0, min(tab_index, self.tabs.count() - 1))
            self.last_open_tab_index = tab_index
            self.tabs.setCurrentIndex(tab_index)

        manual_range = data.get("manual_wavelength", {})
        anchors = manual_range.get("anchors")
        if isinstance(anchors, list):
            self.manual_wavelength_anchors = anchors
        self.manual_anchor_mode = manual_range.get("mode", getattr(self, "manual_anchor_mode", "wavelength"))

        batch = data.get("batch_settings", {})
        self.box_width_input.setText(str(batch.get("box_width", self.box_width_input.text())))
        self.box_height_input.setText(str(batch.get("box_height", self.box_height_input.text())))
        self.step_x_input.setText(str(batch.get("step_x", self.step_x_input.text())))
        self.step_y_input.setText(str(batch.get("step_y", self.step_y_input.text())))
        self.interpolation_checkbox.setChecked(batch.get("interpolation", self.interpolation_checkbox.isChecked()))

        batch_roi = batch.get("roi", {})
        self.min_x_input.setText(str(batch_roi.get("xmin", self.min_x_input.text())))
        self.max_x_input.setText(str(batch_roi.get("xmax", self.max_x_input.text())))
        self.min_y_input.setText(str(batch_roi.get("ymin", self.min_y_input.text())))
        self.max_y_input.setText(str(batch_roi.get("ymax", self.max_y_input.text())))

        updates = data.get("updates", {})
        self.check_updates_on_startup = bool(
            updates.get("check_on_startup", self.check_updates_on_startup)
        )
        self.last_update_check_utc = str(
            updates.get("last_check_utc", self.last_update_check_utc)
        )
        self.ignored_update_version = str(
            updates.get("ignored_version", self.ignored_update_version)
        )
        self.sync_update_check_controls()

    def save_user_settings(self):
        """Persist key GUI parameters to disk."""
        data = {
            "gui_font_size": self.gui_font_size,
            "plot_font_size": getattr(self, "plot_font_size", 12),
            "flight_path": self.flight_path,
            "delay": getattr(self, "delay", 0.0),
            "phase": self.phase_dropdown.currentText(),
            "initial_roi": {
                "xmin": self.xmin_input.text(),
                "xmax": self.xmax_input.text(),
                "ymin": self.ymin_input.text(),
                "ymax": self.ymax_input.text(),
            },
            "wavelength_window": {
                "min": self.min_wavelength_input.text(),
                "max": self.max_wavelength_input.text(),
            },
            "fix_s": self.fix_s_enabled(),
            "fix_t": self.fix_t_enabled(),
            "fix_eta": self.fix_eta_enabled(),
            "symbol_size": getattr(self, "symbol_size", 4),
            "show_edge_line": bool(getattr(self, "show_edge_line", True)),
            "live_fit_preview_enabled": bool(getattr(self, "live_fit_preview_enabled", True)),
            "fitting_plot_layout_mode": getattr(self, "fitting_plot_layout_mode", "single"),
            "last_open_tab_index": self.tabs.currentIndex() if hasattr(self, "tabs") else 0,
            "manual_wavelength": {
                "mode": getattr(self, "manual_anchor_mode", "wavelength"),
                "anchors": getattr(self, "manual_wavelength_anchors", []),
            },
            "batch_settings": {
                "box_width": self.box_width_input.text(),
                "box_height": self.box_height_input.text(),
                "step_x": self.step_x_input.text(),
                "step_y": self.step_y_input.text(),
                "interpolation": self.interpolation_checkbox.isChecked(),
                "roi": {
                    "xmin": self.min_x_input.text(),
                    "xmax": self.max_x_input.text(),
                    "ymin": self.min_y_input.text(),
                    "ymax": self.max_y_input.text(),
                },
            },
            "updates": {
                "check_on_startup": bool(self.check_updates_on_startup),
                "last_check_utc": self.last_update_check_utc,
                "ignored_version": self.ignored_update_version,
            },
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        self.save_user_settings()
        self.cleanup_resources()
        super().closeEvent(event)

    
    # --------------------------------------------
    # 3-Level Summation (folder -> sample -> run)
    # --------------------------------------------
    
    
    
    # def _on_run_loaded_3level(self, folder, run_dict):
    #     if run_dict:
    #         for suffix, image_data in run_dict.items():
    #             if suffix not in self._combined_run_3['images']:
    #                 self._combined_run_3['images'][suffix] = image_data.copy()
    #             else:
    #                 self._combined_run_3['images'][suffix] += image_data
    #         gc.collect()
  
    # --------------------------------------------
    # TWO-LEVEL Summation (folder -> subfolders)
    # --------------------------------------------
 
            # run_number = len(self.normalisation_image_runs)
            # self.preproc_message_box.append(f"Successfully loaded normalisation Data Run {run_number} with {len(run_dict)} images.")
            # self.normalisation_start_button.setEnabled(True)
            
            # self.preproc_message_box.append("normalisation data image loading thread has finished.")



    # Add methods to update normalisation progress and handle completion
        

        # dialog = OpenBeamPlotDialog(wavelengths, sorted_intensities, parent=self)
        # self.open_beam_plot_dialogs.append(dialog)
        # dialog.show()                                                             

    # **Update normalisation Load Progress**

    # **Handler when Open Beam Loading is Finished**

  
        # Optionally remove filtered data from memory or handle it
        # self.filtering_image_runs = []
        # self.filtering_mask_image = None

  
                # self.message_box.append(f"Failed to load CSV file: {e}")

    

    # def open_adjustments_dialog(self):
    #     dialog = AdjustmentsDialog(self)
    #     dialog.show()
    












            

    
    
    


                                                                             














        
    # **Handler for Open Beam Data Loaded**

__all__ = ["FitsViewer"]

