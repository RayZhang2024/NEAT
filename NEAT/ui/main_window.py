"""Main application window for NEAT."""

import gc
import glob
import json
import os
import time

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
    QShortcut,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit, least_squares

from ..core import (
    PHASE_DATA,
    calculate_x_hkl_general,
    fitting_function_1,
    fitting_function_2,
    fitting_function_3,
)
from .mixins.fitting import FittingMixin
from .mixins.postprocessing import PostProcessingMixin
from .mixins.preprocessing import PreprocessingMixin
from .utils import update_all_widget_fonts

class FitsViewer(QMainWindow, PreprocessingMixin, FittingMixin, PostProcessingMixin):
    def __init__(self):
        super().__init__()
        
        self.flight_path = 56.4
        self.delay = 0.0
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        self.tof_array = None
        self.setMinimumSize(800, 600)
        self.setWindowTitle("NEAT Neutron Bragg Edge Analysis Toolkit v4.6.0")
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
        
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "About")
        
        # Set up the layout for Preprocessing tab
        self.setup_preprocessing_tab()

        # Set up the layout for FittingTab (FITS Viewer)
        self.setup_FittingTab()
        
        # Set up the layout for PostProcessingTab (Data Post-Processing)
        self.setup_PostProcessingTab()
        
        self.setup_about_tab()

        # Variables for storing image data and auto-adjust parameters
        self.images = []  # To store multiple FITS images
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
        self.manual_wavelength_min = 1.0
        self.manual_wavelength_max = 3.0

        self._summation_cancelled = False
        self._two_level_subfolders  = []
        self._expected_suffix_set = None      # <-- NEW
        self._expected_image_cnt  = None      # <-- NEW
        
        self.gui_font_size = 8

        self.config_path = os.path.join(os.path.expanduser("~"), ".neat_gui_settings.json")
        self.load_user_settings()
        
        # Set initial global font and Matplotlib settings
        self.setGlobalFont()
        
        # Create shortcuts to adjust font size
        shortcutIncrease = QShortcut(QKeySequence("Shift+Up"), self)
        shortcutIncrease.activated.connect(self.increaseFontSize)
        
        shortcutDecrease = QShortcut(QKeySequence("Shift+Down"), self)
        shortcutDecrease.activated.connect(self.decreaseFontSize)

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

    def increaseFontSize(self):
        self.gui_font_size += 1
        self.setGlobalFont()
        print("Font size increased to", self.gui_font_size)

    def decreaseFontSize(self):
        # Prevent font size from going below a reasonable minimum (e.g., 1)
        if self.gui_font_size > 1:
            self.gui_font_size -= 1
            self.setGlobalFont()
            print("Font size decreased to", self.gui_font_size)

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
            self.bragg_table.setItem(row_position, 10, t_item)
    
        self.message_box.append("Configured table for 'Unknown Phase'. Please input Bragg edge values.")
   
    def compute_theoretical_bragg_edges(self):
        """
        Compute theoretical Bragg edge positions (x_hkl) based on the selected phase's 
        structure, lattice parameters, and hkl_list.  Each x_hkl is paired with its 
        corresponding (h, k, l) index.
        """
        if not self.hkl_list or not self.lattice_params:
            self.message_box.append("Incomplete phase data. Cannot compute Bragg edges.")
            self.theoretical_bragg_edges = []
            return
    
        # Example: Suppose you store the currently selected phase in self.current_phase
        # and retrieve its data from PHASE_DATA. Then:
        structure_type = self.structure_type
        lattice_params = self.lattice_params
        # hkl_list = self.hkl_list
    
        # For convenience, define a small helper function to get d(hkl):
        def d_spacing(h, k, l, structure, lat):
            """
            Return the d-spacing (in Å) for the given hkl and structure.
            lat is a dict with the needed lattice parameters, e.g. lat["a"], lat["b"], etc.
            """
            if structure in ("cubic", "fcc", "bcc"):
                # Requires lat["a"]
                a = lat["a"]
                return a / np.sqrt(h**2 + k**2 + l**2)
    
            elif structure == "tetragonal":
                # Requires lat["a"], lat["c"]
                a = lat["a"]
                c = lat["c"]
                # d_{hkl} = 1 / sqrt( (h^2+k^2)/a^2 + l^2/c^2 )
                return 1.0 / np.sqrt((h**2 + k**2)/(a**2) + (l**2)/(c**2))
    
            elif structure == "hexagonal":
                # Requires lat["a"], lat["c"]
                a = lat["a"]
                c = lat["c"]
                # d_{hkl} = 1 / sqrt( (4/3)*((h^2 + h*k + k^2)/a^2) + l^2/c^2 )
                # For standard HCP formula:
                return 1.0 / np.sqrt((4.0/3.0) * ((h**2 + h*k + k**2)/(a**2)) + (l**2)/(c**2))
    
            elif structure == "orthorhombic":
                # Requires lat["a"], lat["b"], lat["c"]
                a = lat["a"]
                b = lat["b"]
                c = lat["c"]
                return 1.0 / np.sqrt((h**2)/(a**2) + (k**2)/(b**2) + (l**2)/(c**2))
    
            # You could add "monoclinic", "triclinic", etc. as needed.
            # For anything else, default to a 'cubic' or return NaN:
            return np.nan
    
        # Now calculate Bragg edges
        self.theoretical_bragg_edges = []
        for (h, k, l) in self.hkl_list:
            d_hkl = d_spacing(h, k, l, structure_type, lattice_params)
    
            if np.isnan(d_hkl) or d_hkl <= 0:
                self.theoretical_bragg_edges.append(((h, k, l), np.nan))
                self.message_box.append(
                    f"Skipping hkl{(h, k, l)}: invalid or undefined d-spacing.")
                continue
    
            # Bragg edge (assuming sinθ=1 => λ = 2*d)
            x_hkl = 2.0 * d_hkl
            self.theoretical_bragg_edges.append(((h, k, l), x_hkl))
            self.message_box.append(
                f"Theoretical Bragg edge for hkl{(h, k, l)}: {x_hkl:.4f} Å"
            )


    def update_bragg_edge_table(self):
        """
        Updates the Bragg edges table based on the selected phase and wavelength range.
        For 'Unknown_Phase', allows manual input up to 5 rows.
        Otherwise, computes theoretical edges (including non-cubic) and populates the table.
        """
        # Only populate after user explicitly picks a region.
        if not getattr(self, "_bragg_table_ready", False):
            self.bragg_table.setRowCount(0)
            return

        selected_phase = self.phase_dropdown.currentText()
        if selected_phase == "Unknown_Phase":
            self.setup_unknown_phase_table()
            return  # Exit early as manual input is handled separately
    
        if selected_phase not in self.phase_data:
            self.message_box.append("Please select a valid phase.")
            self.bragg_table.setRowCount(0)  # Clear table if invalid phase
            return
    
        # Validate the min/max wavelength inputs
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
            if min_wavelength >= max_wavelength:
                QMessageBox.warning(self, "Invalid Wavelength Range",
                                    "Minimum wavelength must be less than Maximum wavelength.")
                self.bragg_table.setRowCount(0)
                return
        except ValueError:
            self.message_box.append("Please enter valid min and max wavelengths.")
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
                d_val = x_hkl
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
            r1_min_item = QTableWidgetItem(f"{r1_min:.4f}")
            r1_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 4, r1_min_item)
    
            r1_max = min(x_hkl * 1.12, x_hkl + 0.4)
            r1_max_item = QTableWidgetItem(f"{r1_max:.4f}")
            r1_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 5, r1_max_item)
    
            # Region 2
            r2_min = max(x_hkl * 0.90, x_hkl - 0.3)
            r2_min_item = QTableWidgetItem(f"{r2_min:.4f}")
            r2_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 2, r2_min_item)
    
            r2_max = x_hkl * 0.98
            r2_max_item = QTableWidgetItem(f"{r2_max:.4f}")
            r2_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 3, r2_max_item)
    
            # Region 3
            r3_min = r2_min
            r3_min_item = QTableWidgetItem(f"{r3_min:.4f}")
            r3_min_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 6, r3_min_item)
    
            r3_max = r1_max
            r3_max_item = QTableWidgetItem(f"{r3_max:.4f}")
            r3_max_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 7, r3_max_item)
    
            # s parameter
            s = 0.001
            s_item = QTableWidgetItem(f"{s:.5f}")
            s_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 8, s_item)
    
            # t parameter
            t = 0.01
            t_item = QTableWidgetItem(f"{t:.5f}")
            t_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 9, t_item)
            
            # eta parameter
            eta = 0.5
            eta_item = QTableWidgetItem(f"{t:.3f}")
            eta_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.bragg_table.setItem(row_position, 10, eta_item)
    
        # If we got here, we successfully populated the table.
        self.message_box.append(
            f"Updated Bragg edges table with {self.bragg_table.rowCount()} edge(s)."
        )


    def on_bragg_edge_selected(self):
        """
        Handler called when a Bragg edge row is selected in the table.
        Updates the region min/max wavelength inputs, 'd', s, t, etc.
        """
        selected_items = self.bragg_table.selectedItems()
        if not selected_items:
            return  # No selection
    
        selected_row = selected_items[0].row()
    
        try:
            # Column 1 is 'd'
            d_text = self.bragg_table.item(selected_row, 1).text()
            if d_text and d_text not in ("N/A", ""):
                d = float(d_text)
            else:
                d = None
    
            r1_min = float(self.bragg_table.item(selected_row, 4).text())
            r1_max = float(self.bragg_table.item(selected_row, 5).text())
            r2_min = float(self.bragg_table.item(selected_row, 2).text())
            r2_max = float(self.bragg_table.item(selected_row, 3).text())
            r3_min = float(self.bragg_table.item(selected_row, 6).text())
            r3_max = float(self.bragg_table.item(selected_row, 7).text())
            s = float(self.bragg_table.item(selected_row, 8).text())
            t = float(self.bragg_table.item(selected_row, 9).text())
            eta = float(self.bragg_table.item(selected_row, 10).text())
    
        except (ValueError, AttributeError) as e:
            self.message_box.append(f"Error reading values from selected row: {e}")
            return
    
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

        self.fix_s_checkbox.setChecked(data.get("fix_s", self.fix_s_checkbox.isChecked()))
        self.fix_t_checkbox.setChecked(data.get("fix_t", self.fix_t_checkbox.isChecked()))
        self.fix_eta_checkbox.setChecked(data.get("fix_eta", self.fix_eta_checkbox.isChecked()))
        self.symbol_size = data.get("symbol_size", getattr(self, "symbol_size", 4))
        if hasattr(self, "apply_symbol_size"):
            self.apply_symbol_size()

        manual_range = data.get("manual_wavelength", {})
        self.manual_wavelength_min = manual_range.get("min", self.manual_wavelength_min)
        self.manual_wavelength_max = manual_range.get("max", self.manual_wavelength_max)

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
            "fix_s": self.fix_s_checkbox.isChecked(),
            "fix_t": self.fix_t_checkbox.isChecked(),
            "fix_eta": self.fix_eta_checkbox.isChecked(),
            "symbol_size": getattr(self, "symbol_size", 4),
            "manual_wavelength": {
                "min": self.manual_wavelength_min,
                "max": self.manual_wavelength_max,
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
