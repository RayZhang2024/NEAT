
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 18:55:38 2025

@author: Ruiyao Zhang
"""

# -*- coding: utf-8 -*-
"""
Integrated script with Bragg edge fitting and automated region detection,
with optimized Preprocessing tab layout divided into Summation, Scaling, and Normalisation sections.
Includes normalisation functionality to normalise data images using open beam intensities.
@author: Ruiyao Zhang
ruiyao.zhang@stfc.ac.uk
"""

import sys
import numpy as np
import shutil
import datetime  # For timestamp
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
    QWidget, QSlider, QLabel, QLineEdit, QGridLayout, QProgressBar,
    QMessageBox, QDialog, QSizePolicy, QTabWidget, QGroupBox, QDoubleSpinBox, QSplitter, QDesktopWidget
    , QInputDialog, QCheckBox, QScrollArea, QComboBox, QHeaderView, QTableWidget, QTableWidgetItem, QAbstractItemView
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEventLoop, QCoreApplication
from PyQt5.QtGui import QFont, QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
# import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import special
from scipy.interpolate import RegularGridInterpolator
import pandas as pd  # For CSV file handling
import os
import gc  # Garbage collector
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import time
import glob
from scipy.optimize import least_squares

# from scipy.signal import convolve2d

from PyQt5.QtWidgets import QShortcut
import matplotlib as mpl


# app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
app = QApplication([])

# Constants for wavelength range

# TOTAL_IMAGES = 2925  # Total images in the original sequence

# Fitting functions
def fitting_function_1(x, a0, b0):
    return np.exp(-(a0 + b0 * x))

def fitting_function_2(x, a_hkl, b_hkl, a0, b0):
    return np.exp(-((a0) + (b0 * x))) * np.exp(-((a_hkl) + (b_hkl * x)))

def fitting_function_3(
    x, a0, b0, a_hkl, b_hkl,
    s, t, eta,
    hkl_list, 
    min_wavelength, max_wavelength,
    structure_type,        # <--- new
    lattice_params         # <--- new
):
    """
    Fitting function for Region 3, but now capable of using a more general
    formula for x_hkl based on the crystal structure.
    """
    if hkl_list:
        # Use a general function that handles any crystal system
        x_hkl_list = calculate_x_hkl_general(structure_type, lattice_params, hkl_list)
    else:
        # If no hkl_list is provided, fallback to using 'a' directly or skip
        x_hkl_list = [2 * lattice_params.get("a", 0)]  # Fallback for unknown phases

    # Initialize total modeled intensity to zero
    total_intensity = np.zeros_like(x, dtype=float)

    # Iterate over each x_hkl
    for x_hkl in x_hkl_list:
        # Only apply region3 logic if x_hkl is within [min_wavelength, max_wavelength]
        if np.isnan(x_hkl):
            continue
        if (x_hkl < min_wavelength) or (x_hkl > max_wavelength):
            continue


        try:         
            # -------------------------------------------------
            # helper: Lorentzian integral + exp-tail (τ = t > 0)
            # -------------------------------------------------
            def _l_int_exp_tail(x, x_hkl, s, t):
                """
                Integrated Lorentzian convolved with a 1-sided exponential tail.
                Good to <0.3 % for t ≥ s/3 (Thomsen, JAC 52 (2019) 456).
                """
                u      = (x - x_hkl) / (s * np.sqrt(2))
                L      = 0.5 + np.arctan(u) / np.pi                # plain CDF
                shift  = np.exp(-(x - x_hkl) / t)                  # tail weight
                u_s    = (x - x_hkl - s**2 / t) / (s * np.sqrt(2)) # τ–shifted arg
                L_s    = 0.5 + np.arctan(u_s) / np.pi              # shifted CDF
                return L - shift * (L - L_s)
            
            # -------------------------------------------------
            # inside your existing loop / try–block
            # -------------------------------------------------
            g_term = 0.5 * (
                    special.erfc(-(x - x_hkl) / (np.sqrt(2) * s)) -
                    np.exp(-(x - x_hkl) / t + s**2 / (2 * t**2))
                    * special.erfc(-(x - x_hkl)/(np.sqrt(2)*s) + s / t)
            )
            
            l_term = _l_int_exp_tail(x, x_hkl, s, t)        # ← tailed Lorentzian
            
            step   = (1.0 - eta) * g_term + eta * l_term    # pseudo-Voigt mix
            
            pre    = np.exp(-(a_hkl + b_hkl * x))
            edge   = np.exp(-(a0 + b0 * x)) * (pre + (1.0 - pre) * step)
            total_intensity += edge
            

        except Exception as e:
            print(f"Error calculating edge for x_hkl={x_hkl}: {e}")
            continue

    return total_intensity


def calculate_x_hkl_general(structure_type, lattice_params, hkl_list):
    """
    Returns a list of theoretical Bragg edges (lambda_hkl) for the given structure.
    lambda_hkl = 2 * d_hkl. 
    d_hkl depends on the structure and lattice parameters.
    
    Parameters:
        structure_type (str): e.g. "cubic", "tetragonal", "hexagonal", etc.
        lattice_params (dict): e.g. {"a": 3.266, "c": 4.8} for tetragonal, etc.
        hkl_list (list of tuples): (h,k,l) indices.
    """

    def d_spacing(h, k, l, structure, lat):
        if structure == "cubic":
            a = lat["a"]
            return a / np.sqrt(h**2 + k**2 + l**2)

        elif structure == "tetragonal":
            a, c = lat["a"], lat["c"]
            # d_{hkl} = 1 / sqrt( (h^2+k^2)/a^2 + l^2/c^2 )
            return 1.0 / np.sqrt((h**2 + k**2)/(a**2) + (l**2)/(c**2))

        elif structure == "hexagonal":
            a, c = lat["a"], lat["c"]
            # d_{hkl} = 1 / sqrt( (4/3)*((h^2 + h*k + k^2)/a^2) + l^2/c^2 )
            return 1.0 / np.sqrt((4.0/3.0)*((h**2 + h*k + k**2)/(a**2)) + (l**2)/(c**2))

        elif structure == "orthorhombic":
            a, b, c = lat["a"], lat["b"], lat["c"]
            return 1.0 / np.sqrt((h**2)/(a**2) + (k**2)/(b**2) + (l**2)/(c**2))

        # Add more structures if needed...

        # Fallback:
        return np.nan

    x_hkl_list = []
    for (h, k, l) in hkl_list:
        d_hkl = d_spacing(h, k, l, structure_type, lattice_params)
        if d_hkl > 0:
            x_hkl = 2.0 * d_hkl  # Bragg's law (assuming sin(θ)=1 at reflection edge)
            x_hkl_list.append(x_hkl)
        else:
            x_hkl_list.append(np.nan)

    return x_hkl_list


screen_geometry = QDesktopWidget().screenGeometry()  # Get the screen geometry
screen_width = screen_geometry.width()  # Get screen width
screen_height = screen_geometry.height()  # Get screen height

# Set the window size as a percentage of the screen size (e.g., 80%)
window_width = int(screen_width * 0.9)  # 80% of screen width
window_height = int(screen_height * 0.8)  # 80% of screen height

# Define phase information with hkl lists and lattice parameters
PHASE_DATA = {   
    "Unknown_Phase": {
        "structure": "unknown",  # or "custom"
        "lattice_params": {},    # No known lattice constants
        "hkl_list": []
    },
    "Cu_fcc": {
        "structure": "cubic",
        "lattice_params": {"a": 5.431},  # in Å
        "hkl_list": [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (3,2,1), (4,0,0)]
    },
    "Fe_bcc": {
        "structure": "cubic",
        "lattice_params": {"a": 2.86},   # in Å
        "hkl_list": [(1,1,0), (2,0,0), (2,1,1), (2,2,0),
                     (3,1,0), (2,2,2), (3,2,1), (3,3,0)]
    },
    "Fe_fcc": {
        "structure": "cubic",
        "lattice_params": {"a": 3.61},
        "hkl_list": [(1,1,1), (2,0,0), (2,2,0), (3,1,1),
                     (3,2,1), (4,0,0)]
    },
    "Al": {
        "structure": "cubic",
        "lattice_params": {"a": 4.05},
        "hkl_list": [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0),(3,3,1),(4,2,0)],
    },
    "Ni_gamma": {
        "structure": "cubic",
        "lattice_params": {"a": 3.60},
        "hkl_list": [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (3,2,1), (4,0,0)],
    },
    "CeO2": {
        "structure": "cubic",
        "lattice_params": {"a": 5.41},
        "hkl_list": [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (3,2,1), (4,0,0)],
    },
    
    "Ti_Beta": {
        "structure": "tetragonal",
        # For demonstration. Beta-Ti can actually be bcc, but assume tetragonal for example:
        "lattice_params": {"a": 3.266, "c": 4.80},  
        "hkl_list": [(1,1,0), (2,0,0), (2,1,1),
                     (2,2,0), (3,1,0), (2,2,2)]
    },
    "Ti_alpha_hex": {
        "structure": "hexagonal",
        "lattice_params": {"a": 2.95, "c": 4.68},
        "hkl_list": [(1,0,0), (0,0,2), (1,0,1), (1,1,0),
                     (2,0,0), (1,1,2), (2,1,1)]
    },
    
}
    
def update_all_widget_fonts(new_font):
    for widget in QApplication.allWidgets():
        widget.setFont(new_font)
        widget.update()


# Main application window
class FitsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.flight_path = 56.4
        self.delay = 0.0
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        self.tof_array = None
        self.setMinimumSize(800, 600)
        self.setWindowTitle("NEAT Neutron Bragg Edge Analysis Toolkit v4_beta")
        # self.setGeometry(100, 100, window_width, window_height)  # Increased size to accommodate new layout

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
        
        self._summation_cancelled = False
        self._two_level_subfolders  = []
        self._expected_suffix_set = None      # <-- NEW
        self._expected_image_cnt  = None      # <-- NEW
        
        self.gui_font_size = 8
        
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
        if selected_phase != "Select Phase":
            self.message_box.append(f"Selected phase: {selected_phase}")
    
            phase_info = PHASE_DATA.get(selected_phase)
            if phase_info:
                if selected_phase == "Unknown_Phase":
                    # Prepare the table for user input
                    self.structure_type = "unknown"
                    self.lattice_params = {}
                    self.hkl_list = []
                    self.theoretical_bragg_edges = []
                    self.setup_unknown_phase_table()
                    self.update_plots()
                else:
                    # Retrieve multiple things:
                    self.structure_type = phase_info.get("structure", "cubic")
                    self.lattice_params = phase_info.get("lattice_params", {})
                    self.hkl_list = phase_info.get("hkl_list", [])
                    
                    # Now compute theoretical edges using structure + lattice_params
                    # self.compute_theoretical_bragg_edges()
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
            if structure == "cubic":
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
        selected_phase = self.phase_dropdown.currentText()
        if selected_phase == "Unknown_Phase":
            self.setup_unknown_phase_table()
            return  # Exit early as manual input is handled separately
    
        if selected_phase not in PHASE_DATA:
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
        phase_info = PHASE_DATA[selected_phase]
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
    
        # Refresh the plots
        self.update_plots()
    
        if d is not None:
            self.message_box.append(f"Selected 'd' value: {d:.4f}")
        self.message_box.append(
            "--------------------   Bragg Edge: "
            f"{self.bragg_table.item(selected_row, 0).text()}   --------------------"
        )


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
        # ---------------------------

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
        # self.preproc_message_box.append("<b> Full Process finished. </b>")        

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
            # self.remove_overlap_correction_images()

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
            # Clear existing overlap correction images
            # self.remove_overlap_correction_images()

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
            # self.remove_summation_images()

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
    
    
    # --------------------------------------------
    # 3-Level Summation (folder -> sample -> run)
    # --------------------------------------------
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
    
    # def _on_run_loaded_3level(self, folder, run_dict):
    #     if run_dict:
    #         for suffix, image_data in run_dict.items():
    #             if suffix not in self._combined_run_3['images']:
    #                 self._combined_run_3['images'][suffix] = image_data.copy()
    #             else:
    #                 self._combined_run_3['images'][suffix] += image_data
    #         gc.collect()
    
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
    
    
    # --------------------------------------------
    # TWO-LEVEL Summation (folder -> subfolders)
    # --------------------------------------------
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
            # run_number = len(self.normalisation_image_runs)
            # self.preproc_message_box.append(f"Successfully loaded normalisation Data Run {run_number} with {len(run_dict)} images.")
            # self.normalisation_start_button.setEnabled(True)
            
    def normalisation_data_image_loading_finished(self):
        if self.normalisation_data_load_worker:
            self.normalisation_data_load_worker = None  # Cleanup
            # self.preproc_message_box.append("normalisation data image loading thread has finished.")

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


    # Add methods to update normalisation progress and handle completion
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
    
        # dialog = OpenBeamPlotDialog(wavelengths, sorted_intensities, parent=self)
        # self.open_beam_plot_dialogs.append(dialog)
        # dialog.show()                                                             

    # **Update normalisation Load Progress**
    def update_normalisation_load_progress(self, value):
        self.normalisation_load_progress.setValue(value)

    # **Handler when Open Beam Loading is Finished**
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
    
        # Optionally remove filtered data from memory or handle it
        # self.filtering_image_runs = []
        # self.filtering_mask_image = None

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


    def setup_FittingTab(self):
        """
        Bragg edge fitting
        """
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
        
        # Create and add Navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        upper_left_layout.addWidget(self.toolbar) 
        
        upper_left_layout.addWidget(self.canvas)
        upper_left_widget.setMinimumSize(200, 200)
    
    
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
        
        self.auto_adjust_button = QPushButton("Auto Adj.")
        self.auto_adjust_button.clicked.connect(self.auto_adjust)
        self.auto_adjust_button.setToolTip('Auto adjust brightness and contrast')
        button_layout.addWidget(self.auto_adjust_button)
    
        self.adjust_image_button = QPushButton("Manual Adj.")
        self.adjust_image_button.clicked.connect(self.open_adjustments_dialog)
        self.adjust_image_button.setToolTip('Manually adjust brightness and contrast')
        button_layout.addWidget(self.adjust_image_button)
    
        self.flight_path_button = QPushButton("Flight Path")
        self.flight_path_button.setToolTip('Change flight path if needed, default is 56.4 m for IMAT')
        self.flight_path_button.clicked.connect(self.change_flight_path)
        button_layout.addWidget(self.flight_path_button)
    
        self.delay_button = QPushButton("Delay")
        self.delay_button.setToolTip('Set time delay correction')
        self.delay_button.clicked.connect(self.set_delay)
        button_layout.addWidget(self.delay_button)
    
        self.smooth_checkbox = QCheckBox("Smooth")
        self.smooth_checkbox.setChecked(False)
        button_layout.addWidget(self.smooth_checkbox)
    
        self.phase_dropdown = QComboBox()
        self.phase_dropdown.addItem("Phase")  # Placeholder/default item
        self.phase_dropdown.setToolTip('Select a phase from the drop down list, or select "unknown_phase" if the material is unknown')
        self.phase_dropdown.addItems(list(PHASE_DATA.keys()))  # Dynamically add phase names
        self.phase_dropdown.currentIndexChanged.connect(self.phase_selection_changed)  # Connect to handler
        button_layout.addWidget(self.phase_dropdown)
    
        self.show_theoretical_checkbox = QCheckBox("Show")
        self.show_theoretical_checkbox.setToolTip('Check to show the theorectical edges of a selected phase')
        self.show_theoretical_checkbox.setChecked(True)  # Default to shown
        self.show_theoretical_checkbox.stateChanged.connect(self.update_plots)  # Trigger plot update on toggle
        button_layout.addWidget(self.show_theoretical_checkbox)
    
        lower_left_layout.addLayout(button_layout)
    
        # Slider
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setEnabled(False)  # Disabled until images are loaded
        self.image_slider.setMinimumWidth(300)
        self.image_slider.valueChanged.connect(self.update_image)
        lower_left_layout.addWidget(QLabel("Image Selection"))
        lower_left_layout.addWidget(self.image_slider)
    
        # Rectangle selection inputs
        selection_layout = QGridLayout()
        self.xmin_input = QLineEdit("300")
        self.xmin_input.setToolTip('Set x_min coordinate of the ROI for initial edge fitting')
        self.xmax_input = QLineEdit("320")
        self.xmax_input.setToolTip('Set x_max coordinate of the ROI for initial edge fitting')
        self.ymin_input = QLineEdit("300")
        self.ymin_input.setToolTip('Set y_min coordinate of the ROI for initial edge fitting')
        self.ymax_input = QLineEdit("320")
        self.ymax_input.setToolTip('Set y_max coordinate of the ROI for initial edge fitting')
    
        selection_layout.addWidget(QLabel("X Min:"), 0, 0)
        selection_layout.addWidget(self.xmin_input, 0, 1)
        selection_layout.addWidget(QLabel("X Max:"), 0, 2)
        selection_layout.addWidget(self.xmax_input, 0, 3)
        selection_layout.addWidget(QLabel("Y Min:"), 0, 4)
        selection_layout.addWidget(self.ymin_input, 0, 5)
        selection_layout.addWidget(QLabel("Y Max:"), 0, 6)
        selection_layout.addWidget(self.ymax_input, 0, 7)
    
        self.select_area_button = QPushButton(" Pick ")
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
        self.max_wavelength_input = QLineEdit("3.4")
        self.max_wavelength_input.setToolTip('Set upper bound of wavelength range')
    
        wavelength_layout.addWidget(QLabel("Min Wavelength (Å):"), 0, 0)
        wavelength_layout.addWidget(self.min_wavelength_input, 0, 1)
        wavelength_layout.addWidget(QLabel("Max Wavelength (Å):"), 0, 2)
        wavelength_layout.addWidget(self.max_wavelength_input, 0, 3)
    
        self.fix_s_checkbox = QCheckBox("Fix s")
        self.fix_s_checkbox.setToolTip('Check to fix "s" for batch fitting')
        self.fix_s_checkbox.setChecked(True)
        wavelength_layout.addWidget(self.fix_s_checkbox, 0, 4)
    
        self.fix_t_checkbox = QCheckBox("Fix t")
        self.fix_t_checkbox.setToolTip('Check to fix "t" for batch fitting')
        self.fix_t_checkbox.setChecked(True)
        wavelength_layout.addWidget(self.fix_t_checkbox, 0, 5)
        
        self.fix_eta_checkbox = QCheckBox("Fix eta")
        self.fix_eta_checkbox.setToolTip('Check to fix "eta" for batch fitting')
        self.fix_eta_checkbox.setChecked(True)
        wavelength_layout.addWidget(self.fix_eta_checkbox, 0, 6)
    
        lower_left_layout.addLayout(wavelength_layout)
    
        # Bragg Table
        self.bragg_table = QTableWidget()
        self.bragg_table.setColumnCount(11)
        
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
            "Miller indices (h k l)",
            "d-spacing (Å)",
            "lower bound of region 1 (the linear part to the left of the edge)",
            "upper bound of region 1 (the linear part to the left of the edge)",
            "lower bound of region 2 (the linear part to the right of the edge)",
            "upper bound of region 2 (the linear part to the right of the edge)",
            "lower bound of region 3 (the whole region of the edge to be fitted)",
            "upper bound of region 3 (the whole region of the edge to be fitted)",
            "fitting parameter s (broadening)",
            "moderator decay constant t",
            "pseudo-Voigt shape parameter, 0 => pure Gaussian, 1=> pure Lorentzian"
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
    
        self.stop_batch_fit_button = QPushButton("Stop")
        self.stop_batch_fit_button.setToolTip('Stop batch fitting')
        self.stop_batch_fit_button.clicked.connect(self.stop_batch_fit)
        fit_buttons_layout.addWidget(self.stop_batch_fit_button)
    
        self.visualize_button = QPushButton("Fit Check")
        self.visualize_button.setToolTip('Illustrate fitting at specific locations')
        self.visualize_button.clicked.connect(self.visualize_region3_fits)
        fit_buttons_layout.addWidget(self.visualize_button)
    
        lower_left_layout.addLayout(fit_buttons_layout)
    
        # Box/Step Layout
        box_step_layout = QHBoxLayout()
        self.box_width_input = QLineEdit("20")
        self.box_width_input.setToolTip('Macro pixel width')
        self.box_height_input = QLineEdit("20")
        self.box_height_input.setToolTip('Macro pixel height')
        self.step_x_input = QLineEdit("5")
        self.step_x_input.setToolTip('Step in X')
        self.step_y_input = QLineEdit("5")
        self.step_y_input.setToolTip('Step in Y')
    
        self.interpolation_checkbox = QCheckBox("Inter")
        self.interpolation_checkbox.setToolTip('Allow interpolation of results')
        self.interpolation_checkbox.setChecked(True)
    
        box_step_layout.addWidget(QLabel("Box Width:"))
        box_step_layout.addWidget(self.box_width_input)
        box_step_layout.addWidget(QLabel("Box Height:"))
        box_step_layout.addWidget(self.box_height_input)
        box_step_layout.addWidget(QLabel("Step X:"))
        box_step_layout.addWidget(self.step_x_input)
        box_step_layout.addWidget(QLabel("Step Y:"))
        box_step_layout.addWidget(self.step_y_input)
        box_step_layout.addWidget(self.interpolation_checkbox)
        lower_left_layout.addLayout(box_step_layout)
    
        # Region layout
        region_layout = QGridLayout()
        self.min_x_input = QLineEdit("200")
        self.min_x_input.setToolTip('Set x_min coordinate of the ROI for batch fitting')
        self.max_x_input = QLineEdit("500")
        self.max_x_input.setToolTip('Set x_max coordinate of the ROI for batch fitting')
        self.min_y_input = QLineEdit("100")
        self.min_x_input.setToolTip('Set y_min coordinate of the ROI for batch fitting')
        self.max_y_input = QLineEdit("400")
        self.max_y_input.setToolTip('Set y_max coordinate of the ROI for batch fitting')
    
        region_layout.addWidget(QLabel("X Min:"), 0, 0)
        region_layout.addWidget(self.min_x_input, 0, 1)
        region_layout.addWidget(QLabel("X Max:"), 0, 2)
        region_layout.addWidget(self.max_x_input, 0, 3)
        region_layout.addWidget(QLabel("Y Min:"), 0, 4)
        region_layout.addWidget(self.min_y_input, 0, 5)
        region_layout.addWidget(QLabel("Y Max:"), 0, 6)
        region_layout.addWidget(self.max_y_input, 0, 7)
    
        self.select_area_button = QPushButton("ROI")
        self.select_area_button.setToolTip('Define the ROI')
        self.select_area_button.clicked.connect(self.apply_selected_area)
        region_layout.addWidget(self.select_area_button, 0, 8)
        lower_left_layout.addLayout(region_layout)
    
        # Progress bars
        fits_load_progress_layout = QHBoxLayout()
        self.fits_load_progress_label = QLabel("Image Loading Progress:")
        self.fits_load_progress_bar = QProgressBar()
        self.fits_load_progress_bar.setValue(0)
        fits_load_progress_layout.addWidget(self.fits_load_progress_label)
        fits_load_progress_layout.addWidget(self.fits_load_progress_bar)
        lower_left_layout.addLayout(fits_load_progress_layout)
    
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Fitting Progress:")
        self.progress_bar = QProgressBar()
        self.remaining_time_label = QLabel("Remaining: ")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.remaining_time_label)
        lower_left_layout.addLayout(progress_layout)
    
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
    
        # Compute total boxes for progress bar
        total_boxes = ((fit_area_height - box_height) // step_y + 1) * ((fit_area_width - box_width) // step_x + 1)
    
        # Reset and set up the progress bar
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
    
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
        self.batch_fit_edges_worker = BatchFitEdgesWorker(
            parent=self,
            images=self.images,
            wavelengths=self.wavelengths,
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
        self.progress_bar.setValue(100)
        self.remaining_time_label.setText("Remaining: ")


    def fit_full_pattern_core(self, fix_s=False, fix_t=False, fix_eta=False):
        """Core logic for full-pattern fitting supporting multiple crystal structures 
           and returning parameter uncertainties.
        """
        STRUCTURE_CONFIG = {
            "cubic": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"]
        }
    
        # -------------------------------
        # 1) Validate structure & params
        # -------------------------------
        structure_type = getattr(self, "structure_type", "cubic")
        if structure_type not in STRUCTURE_CONFIG:
            return None, f"Unsupported structure type: {structure_type}"
    
        required_params = STRUCTURE_CONFIG[structure_type]
        if not hasattr(self, "lattice_params"):
            return None, "Lattice parameters not initialized"
    
        missing_params = [p for p in required_params if p not in self.lattice_params]
        if missing_params:
            return None, f"Missing parameters for {structure_type}: {missing_params}"
    
        # -------------------------------
        # 2) Collect valid Bragg edges w/ Region 3 data
        # -------------------------------
        total_edges = self.bragg_table.rowCount()
        if total_edges == 0:
            return None, "No Bragg edges to fit"
    
        bragg_edges = []
        for row in range(total_edges):
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
            mask_r3 = (self.wavelengths >= r3_min) & (self.wavelengths <= r3_max)
            x_r3 = self.wavelengths[mask_r3]
            y_r3 = self.intensities[mask_r3]
            if len(x_r3) == 0:
                # Skip edges that have no Region 3 coverage
                continue
    
            # Fit Region 1 and 2
            try:
                # Region 1
                mask_r1 = (self.wavelengths >= regions[1]['min_wavelength']) & (self.wavelengths <= regions[1]['max_wavelength'])
                x_r1 = self.wavelengths[mask_r1]
                y_r1 = self.intensities[mask_r1]
                p0 = [0,0]
                lower = [-10, -10]
                upper = [10, 10]

                popt_r1, _ = curve_fit(fitting_function_1, x_r1, y_r1, p0=p0, bounds=(lower, upper))
                a0, b0 = popt_r1
    
                # Region 2
                mask_r2 = (self.wavelengths >= regions[0]['min_wavelength']) & (self.wavelengths <= regions[0]['max_wavelength'])
                x_r2 = self.wavelengths[mask_r2]
                y_r2 = self.intensities[mask_r2]
                popt_r2, _ = curve_fit(
                    lambda xx, a, b: fitting_function_2(xx, a, b, a0, b0),
                    x_r2, y_r2, p0=[0, 0]
                )
                a_hkl, b_hkl = popt_r2
    
            except Exception as e:
                return None, f"Fitting error for hkl{hkl}: {str(e)}"
    
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
        lattice_initial = [self.lattice_params[p] for p in required_params]
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
                max_nfev=2000,
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
        self.lattice_params.update(lattice_fit)
    
        # -------------------------------
        # 7) Compute parameter uncertainties from Jacobian
        # -------------------------------
        # residuals at the final solution:
        final_res = residuals(final_params)
        N = len(concatenated_y)
        M = len(final_params)
        RSS = np.sum(final_res**2)
        # Basic variance estimate:
        if N > M:
            variance = RSS / (N - M)
        else:
            variance = RSS / N
    
        try:
            J = result.jac
            JTJ = J.T @ J
            cov = np.linalg.inv(JTJ) * variance
            param_stderr = np.sqrt(np.diag(cov))  # std dev for each param
        except np.linalg.LinAlgError:
            # If singular, set uncertainties to inf
            param_stderr = np.full(len(final_params), np.inf)
    
        # Lattice param uncertainties
        lattice_uncertainties = {}
        for i, p in enumerate(required_params):
            lattice_uncertainties[p] = param_stderr[i]
    
        # s uncertainties
        s_uncertainties_list = []
        if not fix_s:
            s_uncertainties_list = param_stderr[len(required_params) : len(required_params) + len(bragg_edges)]
            idx_ = len(required_params) + len(bragg_edges)
        else:
            s_uncertainties_list = [0.0]*len(bragg_edges)
            idx_ = len(required_params)
    
        # t uncertainties
        t_uncertainties_list = []
        if not fix_t:
            t_uncertainties_list = param_stderr[idx_ : idx_ + len(bragg_edges)]
        else:
            t_uncertainties_list = [0.0]*len(bragg_edges)
            
        # eta uncertainties
        eta_uncertainties_list = []
        if not fix_eta:
            eta_uncertainties_list = param_stderr[idx_ : idx_ + len(bragg_edges)]
        else:
            eta_uncertainties_list = [0.0]*len(bragg_edges)
    
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
            'residuals': final_res,
            'success': result.success,
            'message': result.message
        }
        
        edge_heights = {}
        edge_widths = {}
        
        # For each edge in bragg_edges, we now have final s/t + final lattice params.
        # We can compute the edge wavelength x_edge from "lattice_fit" + h,k,l,
        # then region1 & region2 intensities => height = f1(x_edge) - f2(x_edge).
        # Example for "cubic" or general structure:
        def d_spacing(h, k, l, structure, lat):
            # minimal version or call your "calculate_x_hkl_general"
            if structure == "cubic":
                a_val = lat["a"]
                return a_val / np.sqrt(h**2 + k**2 + l**2)
            elif structure == "tetragonal":
                a_val = lat["a"]
                c_val = lat["c"]
                return 1.0 / np.sqrt((h**2 + k**2)/(a_val**2) + (l**2)/(c_val**2))
            elif structure == "hexagonal":
                a_val = lat["a"]
                c_val = lat["c"]
                return 1.0 / np.sqrt((4/3)*((h**2 + h*k + k**2)/a_val**2) + (l**2)/(c_val**2))
            elif structure == "orthorhombic":
                a_val = lat["a"]
                b_val = lat["b"]
                c_val = lat["c"]
                return 1.0 / np.sqrt((h**2)/(a_val**2) + (k**2)/(b_val**2) + (l**2)/(c_val**2))
            else:
                return np.nan
    
        for i, edge in enumerate(bragg_edges):
            (h, k, l) = edge['hkl']
            # final s/t
            s_val = fitted_s_vals[i]
            t_val = fitted_t_vals[i]
            eta_val = fitted_eta_vals[i]
            a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_fit_block[i]
            # a0  = edge['a0']
            # b0  = edge['b0']
            # ahk = edge['a_hkl']
            # bhk = edge['b_hkl']
    
            # compute d_{hkl} from final lattice param
            d_hkl = d_spacing(h, k, l, structure_type, lattice_fit)
            if np.isnan(d_hkl) or d_hkl <= 0:
                # skip if invalid
                edge_heights[edge['hkl']] = np.nan
                edge_widths[edge['hkl']] = np.nan
                continue

            
            """
            Sample Region-3 on a fine grid, take the derivative,
            and return its FWHM in Å.
            """
            xx = np.linspace(r3_min, r3_max, 14000)
            yy = fitting_function_3(
                    xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_val, t_val, eta_val, [edge['hkl']],
                    r3_min, r3_max,
                    structure_type, lattice_fit  # 'a' value irrelevant for width
                  )
            try:
                dy = np.gradient(yy, xx)
                half = dy.max() / 2
                left = xx[dy >= half][0]
                right = xx[dy >= half][-1]
                edge_width = right - left
         
                edge_height = yy.max() - yy.min()
                
            except:
                edge_width = np.nan
                edge_height = np.nan
            
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
            self.plot_canvas_a.axes.clear()
            # Original data for the "selected region"
            # (Assuming you have self.selected_region_x, etc.)
            self.plot_canvas_a.axes.plot(
                self.selected_region_x,
                self.selected_region_y,
                'o',
                markersize=3,
                color='blue',
                # label='Experimental Data'
            )
            # Fitted model
            x_fit = result_dict['x_data']
            y_fit = result_dict['y_data']
            self.plot_canvas_a.axes.plot(
                x_fit,
                y_fit,
                'r-',
                # label='Fitted Model'
            )
    
            # Plot theoretical edges in [min_wavelength, max_wavelength]
            if self.show_theoretical_checkbox.isChecked():
                edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength)
                for (hkl, x_hkl) in edges_in_range:
                    self.plot_canvas_a.axes.axvline(x=x_hkl, color='red', linestyle='--')
                    y_max = self.plot_canvas_a.axes.get_ylim()[1]
                    self.plot_canvas_a.axes.text(
                        x_hkl * 1.02,
                        y_max * 0.95,
                        f'hkl{hkl}',
                        rotation=90,
                        verticalalignment='top',
                        color='red',
                        fontsize=18
                    )
    
            self.plot_canvas_a.axes.set_xlabel("Wavelength (Å)")
            self.plot_canvas_a.axes.set_ylabel("Summed Intensity")
            self.plot_canvas_a.axes.set_title(f"Fit Results - {structure_type.capitalize()} Structure")
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
        self.message_box.append(f"Selected area: x({self.min_x}, {self.max_x}), y({self.min_y}, {self.max_y})")
        
        # Now update the image to show the selected area with a rectangle
        self.display_image()
    
    
    def setup_PostProcessingTab(self):
        layout = QGridLayout(self.PostProcessingTab)  # Change to grid layout
    
        # Add widgets to specific grid positions
        self.load_csv_button = QPushButton("Load CSV File")
        self.load_csv_button.clicked.connect(self.load_csv_file)
        layout.addWidget(self.load_csv_button, 0, 0)  # Row 0, Column 0
        # self.load_csv_button.setStyleSheet("""
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
    
        self.parameter_scroll_area = QScrollArea()
        self.parameter_scroll_area.setWidgetResizable(True)
        self.parameter_widget = QWidget()
        self.parameter_layout = QGridLayout(self.parameter_widget)
        self.parameter_scroll_area.setWidget(self.parameter_widget)
        layout.addWidget(self.parameter_scroll_area, 1, 1)  # Row 0, Column 1
        # Store references to dynamically created buttons to manage them later
        self.parameter_buttons = {}
    
        self.metadata_display = QTextEdit()
        self.metadata_display.setReadOnly(True)
        layout.addWidget(self.metadata_display, 1, 0)  # Row 1 spans 2 columns
        
        # Set column stretch factors
        layout.setColumnStretch(0, 1)  # Column 0 gets 1 part of the space
        layout.setColumnStretch(1, 1)  # Column 1 gets 2 parts of the space
        layout.setColumnStretch(2, 4)  # Column 1 gets 2 parts of the space
    

        self.PostProcessingTab.setLayout(layout)


        # Initialize variables to store data
        self.csv_data = None
        self.current_csv_metadata = {}
        self.current_csv_filename = ""
        
    def setup_about_tab(self):
        layout = QVBoxLayout(self.tab3)
        
        about_text = """
        <h2>NEAT Neutron Bragg Edge Analysis Toolkit v4.4_beta</h2>
        
        <p><b>Developed by:</b><br>
        Engineering and imaging group<br>
        ISIS Neutron and Muon Source<br>
        Rutherford Appleton Laboratory</p>
        
        <p><b>Main authors:</b><br>
        • Ruiyao Zhang (ruiyao.zhang@stfc.ac.uk)<br>
        • Ranggi Ramadhan (ranggi.ramadhan@stfc.ac.uk)</p>
        
        <p><b>Release date:</b><br>
        May 2025</p>
        
        <p><b>Font size adjustment:</b><br>
        'Shift + Up' to increase font size<br>
        'Shift + Down' to decrease font size</p>
        
        """

        about_label = QLabel(about_text)
        about_label.setAlignment(Qt.AlignTop)
        about_label.setWordWrap(True)
        about_label.setTextFormat(Qt.RichText)
        
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(about_label)
                
        layout.addWidget(scroll)
                    

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
                # self.message_box.append(f"Using spectra file: {spectra_file}")
    
                try:
                    # Load the first column (ToF) values
                    self.tof_array = np.loadtxt(spectra_file, usecols=0)
                    # Compute wavelength from ToF: (ToF * 3.956) / self.flight_path * 1000
                    self.update_wavelengths()
    
                except Exception as e:
                    self.message_box.append(f"Error reading spectra file: {e}")
                    self.wavelengths = np.array([])
                    self.tof_array = None
            else:
                # No spectra file to load, set empty wavelengths
                self.wavelengths = np.array([])
                self.tof_array = None
    
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

    def update_fits_load_progress(self, value):
        """
        Update the FITS Viewer image loading progress bar.
        """
        self.fits_load_progress_bar.setValue(value)
    
    def fits_image_loading_finished(self):
        """
        Handle the completion of the image loading process.
        Reset the progress bar and re-enable the load button.
        """
        if hasattr(self, 'fits_image_load_worker'):
            self.fits_image_load_worker = None  # Cleanup
            self.fits_load_progress_bar.setValue(0)  # Reset progress bar
            self.load_button.setEnabled(True)        # Re-enable the load button

  
    def load_csv_file(self):
        """
        Loads a CSV file, parses metadata and data, and displays metadata.
        Dynamically creates buttons for each parameter available in the CSV.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Load CSV File", 
            "", 
            "CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    # Read metadata
                    metadata = {}
                    line = f.readline()
                    if line.strip() != "Metadata Name,Metadata Value":
                        raise ValueError("CSV file does not contain expected metadata header.")
                    
                    # Read metadata lines until a blank line is encountered
                    for line in f:
                        if line.strip() == "":
                            break
                        key, value = line.strip().split(",", 1)
                        metadata[key] = value
                    
                    # Read the rest as data
                    data = pd.read_csv(f)
                    
                    # Prepare metadata text with the file name
                # file_name_display = f"File: {file_name}\n"  # Add the file name
                file_name_display = os.path.basename(file_name)
                metadata_text = file_name_display + "\n" + "\n" + "\n".join([f"{key}: {value}" for key, value in metadata.items()])

                
                # Display metadata in the metadata_display widget
                # metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
                self.metadata_display.setPlainText(metadata_text)
                
                # Store data for plotting
                self.csv_data = data
                self.current_csv_metadata = metadata  # Store metadata for use in plot_parameter
                self.current_csv_filename = file_name
                file_name_short = self.get_short_path(file_name, levels=3)
                
                
                # Inform the user
                QMessageBox.information(self, "Success", f"CSV file loaded successfully: {file_name_short}")
                # self.message_box.append(f"CSV file loaded successfully: {file_name}")
                
                # Dynamically create parameter buttons
                self.create_parameter_buttons()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load CSV file:\n{e}")
                # self.message_box.append(f"Failed to load CSV file: {e}")

    def create_parameter_buttons(self):
        """
        Dynamically creates buttons for each parameter available in the CSV data.
        Excludes 'x' and 'y' columns.
        """
        if self.csv_data is None:
            return

        # Clear existing buttons
        for button in self.parameter_buttons.values():
            button.setParent(None)
        self.parameter_buttons.clear()

        # Identify parameter columns (exclude 'x' and 'y')
        parameter_columns = [col for col in self.csv_data.columns if col not in ['x', 'y']]

        if not parameter_columns:
            QMessageBox.warning(self, "No Parameters", "No parameter columns found in the CSV file.")
            return

        # Define grid placement
        columns = 1  # Number of buttons per row
        row = 0
        col = 0

        for param in parameter_columns:
            button = QPushButton(f"{param}")
            button.clicked.connect(lambda checked, p=param: self.plot_parameter(p))
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.parameter_layout.addWidget(button, row, col)
            self.parameter_buttons[param] = button
            # Add a spacer below each button to set vertical distance
            # spacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
            # self.parameter_layout.addItem(spacer, row + 1, col)

            col += 1
            if col >= columns:
                col = 0
                row += 1

        # Adjust the layout
        # self.parameter_layout.setVerticalSpacing(5)
        self.parameter_widget.setLayout(self.parameter_layout)
        self.parameter_scroll_area.setWidget(self.parameter_widget)
    
    def plot_parameter(self, parameter_name):
        # Check if CSV data is loaded
        if self.csv_data is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV file first.")
            return
    
        # Check if the parameter exists in the CSV data
        if parameter_name not in self.csv_data.columns:
            QMessageBox.warning(
                self, "Parameter Not Found",
                f"Parameter '{parameter_name}' not found in CSV data."
            )
            return
    
        # Get x, y coordinates and the parameter values
        x = self.csv_data['x'].values
        y = self.csv_data['y'].values
        z = self.csv_data[parameter_name].values
    
        # Create unique x and y values
        X_unique = np.sort(np.unique(x))
        Y_unique = np.sort(np.unique(y))
    
        # Create a grid of Z values
        try:
            # Initialize Z grid with NaNs
            Z = np.full((len(Y_unique), len(X_unique)), np.nan)
            # Create a mapping from x and y values to indices
            x_to_idx = {val: idx for idx, val in enumerate(X_unique)}
            y_to_idx = {val: idx for idx, val in enumerate(Y_unique)}
            # Fill Z grid
            for xi, yi, zi in zip(x, y, z):
                ix = x_to_idx[xi]
                iy = y_to_idx[yi]
                Z[iy, ix] = zi
        except Exception as e:
            QMessageBox.warning(
                self, "Data Error",
                f"Error processing data for plotting: {e}"
            )
            return
    
        # Open ParameterPlotDialog to display the parameter
        dialog = ParameterPlotDialog(
            X_unique, Y_unique, Z, parameter_name,
            metadata=self.current_csv_metadata,  # Pass the metadata
            csv_filename=self.current_csv_filename,
            work_directory=getattr(self, 'work_directory', None),
            parent=self
        )
        dialog.show()


    # def open_adjustments_dialog(self):
    #     dialog = AdjustmentsDialog(self)
    #     dialog.show()
    
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

    
        # Display the adjusted image on the Matplotlib canvas
        self.canvas.axes.clear()
        self.canvas.axes.imshow(current_image, cmap='gray', vmin=self.current_vmin, vmax=self.current_vmax)
    
        # Draw the selected area, if available
        if self.selected_area:
            xmin, xmax, ymin, ymax = self.selected_area
            width = xmax - xmin
            height = ymax - ymin
            rect = Rectangle((xmin, ymin), width, height, edgecolor='yellow', facecolor='none', lw=1)
            self.canvas.axes.add_patch(rect)
    
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
            xmin, xmax, ymin, ymax = self.current_batch_box
        
            # If a rectangle already exists, update it
            if self.batch_box_patch is not None and self.batch_box_patch.axes is not None:
                try:
                    # Update position and size instead of removing/re-adding
                    self.batch_box_patch.set_xy((ymin, xmin))
                    self.batch_box_patch.set_width(ymax - ymin)
                    self.batch_box_patch.set_height(xmax - xmin)
                except Exception:
                    # Fallback: recreate it cleanly
                    self.batch_box_patch.remove()
                    self.batch_box_patch = Rectangle(
                        (ymin, xmin),
                        ymax - ymin,
                        xmax - xmin,
                        edgecolor="red",
                        facecolor="none",
                        lw=1,
                    )
                    self.canvas.axes.add_patch(self.batch_box_patch)
            else:
                # No existing patch — create a new one
                self.batch_box_patch = Rectangle(
                    (ymin, xmin),
                    ymax - ymin,
                    xmax - xmin,
                    edgecolor="red",
                    facecolor="none",
                    lw=1,
                )
                self.canvas.axes.add_patch(self.batch_box_patch)
        
        self.canvas.axes.set_title(f"Image {self.current_image_index + 1}/{len(self.images)}")
        self.canvas.draw_idle()


    def update_display(self):
        # Update the displayed image to show the moving batch box
        self.display_image()

    def update_current_box(self, xmin, xmax, ymin, ymax):
        # Update the current batch box coordinates
        self.current_batch_box = (xmin, xmax, ymin, ymax)

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
            self.display_image()  # Refresh display to show the selected area
            self.update_plots()  # Update all plots
        except ValueError:
            self.message_box.append("Please enter valid integers for the coordinates.")

    def update_plots(self):
        if not self.images or not self.selected_area:
            return
    
        # Sum intensity in the selected area for each image
        intensities = []
        for img in self.images:
            xmin, xmax, ymin, ymax = self.selected_area
            selected_area = img[ymin:ymax, xmin:xmax]
            intensities.append(np.sum(selected_area))
    
        self.intensities = np.array(intensities)/[(xmax-xmin)*(ymax-ymin)]
        
        # Apply smoothing if checkbox is checked
        if self.smooth_checkbox.isChecked() and len(self.intensities) > 2:
            self.intensities = np.convolve(self.intensities, np.ones(3)/3, mode='same')
    
        # Clear previous fitted parameters
        self.params_region1 = None
        self.params_region2 = None
        self.params_region3 = {}
    
        # Get the wavelength range from inputs, if provided
        try:
            min_wavelength = float(self.min_wavelength_input.text())
            max_wavelength = float(self.max_wavelength_input.text())
            r1_min = self.current_r1_min
            r1_max = self.current_r1_max
            r2_min = self.current_r2_min
            r2_max = self.current_r2_max
            r3_min = self.current_r3_min
            r3_max = self.current_r3_max
        except ValueError:
            # If inputs are empty or invalid, use full range
            min_wavelength = self.wavelengths[0]
            max_wavelength = self.wavelengths[-1]
        
        try:
            # Filter the wavelengths and intensities based on the range
            mask = (self.wavelengths >= min_wavelength) & (self.wavelengths <= max_wavelength)
            self.selected_region_x = self.wavelengths[mask]  # Store as instance variable
            self.selected_region_y = self.intensities[mask]  # Store as instance variable
        except TypeError:
            # QMessageBox.warning(self, "Save Data", "Failed to save data:\n")
            self.message_box.append("Select a Bragg Edge to display the regions")
            
        # Plot (a): Intensity vs Wavelength
        self.plot_canvas_a.axes.clear()
        self.plot_canvas_a.axes.plot(
            self.selected_region_x,
            self.selected_region_y,
            'o',
            markersize=3,
            color='blue',
            # label='Experimental Data'
        )
        
        # **Calculate and Plot Theoretical Bragg Edges within the Range**
        edges_in_range = self.get_edges_in_range(min_wavelength, max_wavelength) if self.show_theoretical_checkbox.isChecked() else []
        for (hkl, x_hkl) in edges_in_range:
            # Plot vertical dashed line for the theoretical Bragg edge
            self.plot_canvas_a.axes.axvline(
                x=x_hkl,
                color='red',
                linestyle='--',
                # label='Theoretical Bragg Edge'
            )
            # Annotate the (h, k, l) index near the top of the plot
            y_max = self.plot_canvas_a.axes.get_ylim()[1]
            self.plot_canvas_a.axes.text(
                x_hkl * 1.02,
                y_max * 0.95,  # Position text slightly below the top
                f'hkl{hkl}',
                rotation=90,
                verticalalignment='top',
                color='red',
                fontsize=18  # Adjust font size as needed
            )
    
        self.plot_canvas_a.axes.set_xlabel("Wavelength (Å)")
        self.plot_canvas_a.axes.set_ylabel("Summed Intensity")
        self.plot_canvas_a.axes.set_title("Intensity vs Wavelength")
        
        # **Handle Legend to Avoid Duplicate Labels**
        handles, labels = self.plot_canvas_a.axes.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        # self.plot_canvas_a.axes.legend(unique_handles, unique_labels)
        
        self.plot_canvas_a.draw()

        try:    
            regions = [
                ('Region 1',  r2_min, r2_max, self.plot_canvas_b),
                ('Region 2',  r1_min, r1_max, self.plot_canvas_c),
                ('Region 3',  r3_min, r3_max, self.plot_canvas_d),
            ]
        # try:    
        #     regions = [
        #         (r2_min, r2_max, self.plot_canvas_b),
        #         (r1_min, r1_max, self.plot_canvas_c),
        #         (r3_min, r3_max, self.plot_canvas_d),
        #     ]
        except ValueError:
            self.message_box.append("Please enter valid min and max wavelengths.")
            return
        

    
        for name, region_min_wavelength, region_max_wavelength, canvas in regions:
            
            try:
                # Filter data for this region
                region_mask = (self.wavelengths >= region_min_wavelength) & (self.wavelengths <= region_max_wavelength)
                region_x = self.wavelengths[region_mask]
                region_y = self.intensities[region_mask]
           
        
                canvas.axes.clear()
                canvas.axes.plot(
                    region_x,
                    region_y,
                    'o',
                    markersize=3,
                    # label=name
                )
                canvas.axes.set_xlabel("Wavelength (Å)")
                canvas.axes.set_ylabel("Summed Intensity")
                canvas.axes.set_title(f"Intensity Profile of {name}")
                # canvas.axes.legend()
                canvas.draw()
            except Exception:
                # self.message_box.append("Select a Bragg Edge to display the regions")
                return

            
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
    
        # We'll limit to at most 5 rows, as in your original code:
        max_rows = min(5, self.bragg_table.rowCount())
    
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

                                                                             

    def fit_region(self, row_number, skip_ui_updates=False):
        
        fix_s = self.fix_s_checkbox.isChecked()
        fix_t = self.fix_t_checkbox.isChecked()
        fix_eta = self.fix_eta_checkbox.isChecked()
        selected_phase = self.phase_dropdown.currentText()
        is_known_phase = selected_phase != "Unknown_Phase"
    
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
                if "a" not in self.lattice_params:
                    raise ValueError("Missing cubic lattice parameter 'a'")
                a_guess = self.lattice_params["a"]
            else:
                # Unknown phase: use 'd' value from table as 'a' estimate
                d_item = self.bragg_table.item(row_number, 1)
                if not d_item or not d_item.text().strip():
                    raise ValueError("Missing 'd' value for unknown phase")
                a_guess = float(d_item.text())/2
    
            # Get region boundaries and parameters
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
        # Initialize parameter storage for this row if not already done
        if not hasattr(self, 'params_unknown'):
            self.params_unknown = {}
    
        # -------------------------------------------------
        #  Region 1 Fitting
        # -------------------------------------------------
        try:
            mask_r1 = (self.wavelengths >= r1_min) & (self.wavelengths <= r1_max)
            x_r1 = self.wavelengths[mask_r1]
            y_r1 = self.intensities[mask_r1]
    
            if x_r1.size == 0:
                if not skip_ui_updates:
                    self.message_box.append(
                        f"Edge {row_number + 1} - Region 1: No data in the specified range.")
                return
    
            if not skip_ui_updates:
                self.plot_canvas_c.axes.plot(x_r1, y_r1, 'b.', 
                                             # label=f"Edge {row_number + 1} R1 Data"
                                             )
            p0 = [0,0]
            lower = [-10, -10]
            upper = [10, 10]
            popt_r1, _ = curve_fit(fitting_function_1, x_r1, y_r1, p0=p0, bounds=(lower, upper))
            a0, b0 = popt_r1
            self.params_unknown[row_number, 1] = popt_r1  # (a0, b0)
            fit_y1 = fitting_function_1(x_r1, *popt_r1)
    
            if not skip_ui_updates:
                self.plot_canvas_c.axes.plot(x_r1, fit_y1, 'r-', 
                                             # label=f"Edge {row_number + 1} Fit R1"
                                             )
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
            if (row_number, 1) not in self.params_unknown:
                # Region 1 must be fitted first
                return
            a0, b0 = self.params_unknown[row_number, 1]
    
            mask_r2 = (self.wavelengths >= r2_min) & (self.wavelengths <= r2_max)
            x_r2 = self.wavelengths[mask_r2]
            y_r2 = self.intensities[mask_r2]
    
            if x_r2.size == 0:
                if not skip_ui_updates:
                    self.message_box.append(
                        f"Edge {row_number + 1} - Region 2: No data in range.")
                return
    
            if not skip_ui_updates:
                self.plot_canvas_b.axes.plot(x_r2, y_r2, 'b.', 
                                             # label=f"Edge {row_number + 1} R2 Data"
                                             )
    
            popt_r2, _ = curve_fit(
                lambda xx, a_hkl, b_hkl: fitting_function_2(xx, a_hkl, b_hkl, a0, b0),
                x_r2, y_r2, p0=[0, 0]
            )
            a_hkl, b_hkl = popt_r2
            self.params_unknown[row_number, 2] = popt_r2  # (a_hkl, b_hkl)
            fit_y2 = fitting_function_2(x_r2, *popt_r2, a0, b0)
    
            if not skip_ui_updates:
                self.plot_canvas_b.axes.plot(
                    x_r2, fit_y2, 'r-', 
                    # label=f"Edge {row_number + 1} Fit R2"
                    )
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
            a0_hat,  b0_hat  = self.params_unknown[row_number, 1]
            a_hkl_hat, b_hkl_hat = self.params_unknown[row_number, 2]
            
            # 2. first four bounds ( ±50 % each ) ---------------------------------
            lb4, ub4 = zip(*(span50(p) for p in (a0_hat, b0_hat, a_hkl_hat, b_hkl_hat)))

            # ---------- hkl tuple ---------------------------------
            if is_known_phase:
                hkl_item = self.bragg_table.item(row_number, 0)
                h, k, l = map(int, hkl_item.text().strip("()").split(","))
                hkl = (h, k, l)
            else:
                hkl = f"edge{row_number+1}"        # label for unknown

            # ---------- masks & data ------------------------------
            mask_r3 = (self.wavelengths >= r3_min) & (self.wavelengths <= r3_max)
            x_r3    = self.wavelengths[mask_r3]
            y_r3    = self.intensities[mask_r3]

            if not skip_ui_updates:
                self.plot_canvas_d.axes.plot(
                    x_r3, y_r3, "b.", 
                    # label=f"Edge {row_number+1} R3 Data"
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
                bounds=(lb, ub), maxfev=2000
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
            
  
            """
            Sample Region-3 on a fine grid, take the derivative,
            and return its FWHM in Å.
            """
            xx = np.linspace(r3_min, r3_max, 14000)
            yy = fitting_function_3(
                    xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_fit, t_fit, eta_fit, [hkl] if is_known_phase else [],
                    r3_min, r3_max,
                    "cubic", {"a": a_fit/2}  # 'a' value irrelevant for width
                  )
            try:
                dy = np.gradient(yy, xx)
                half = dy.max() / 2
                left = xx[dy >= half][0]
                right = xx[dy >= half][-1]
                edge_width = right - left
                yy_max = yy.max()
                yy_min = yy.min()
                edge_height = yy_max - yy_min
            except:
                edge_width = np.nan
                edge_height = np.nan
            self.params_unknown[row_number, 4] = edge_height
            self.params_unknown[row_number, 5] = edge_width
            
            # <<< END NEW ----------------------------------------
            
            # ----- store everything for later use ----------------
            self.params_unknown[row_number, 3] = (
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
                }

            # ---------- live plotting / messages (GUI) -----------
            if not skip_ui_updates:
                self.plot_canvas_d.axes.plot(
                    x_r3, func_r3(x_r3, *popt_3), "r-",
                    # label=f"Edge {hkl} Fit R3"
                )
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
    
        # Compute the total number of boxes for progress bar
        fit_area_width = max_x - min_x
        fit_area_height = max_y - min_y
        total_boxes = ((fit_area_height - box_height) // step_y + 1) * ((fit_area_width - box_width) // step_x + 1)
    
        # Set up progress bar
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
    
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
    
        # Start the batch fitting worker
        self.batch_fit_worker = BatchFitWorker(
            parent=self,
            images=self.images,
            wavelengths=self.wavelengths,
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
        self.remaining_time_label.setText("Remaining: ")

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_remaining_time(self):
        # Get the current progress value
        value = self.progress_bar.value()
    
        if value > 0:
            current_time = time.time()
            elapsed_time = current_time - self.fit_start_time
            estimated_total_time = elapsed_time / (value / 100.0)
            remaining_time = estimated_total_time - elapsed_time
    
            # Format the remaining time
            remaining_time_str = self.format_time(remaining_time)
            self.remaining_time_label.setText(f"Remaining: {remaining_time_str}")
        else:
            self.remaining_time_label.setText("Remaining: Calculating...")

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
        self.progress_bar.setValue(100)
        # Reset remaining time label
        self.remaining_time_label.setText("Remaining: ")

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
        
    # **Handler for Open Beam Data Loaded**
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

  
class BatchFitEdgesWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)
    message = pyqtSignal(str)
    current_box_changed = pyqtSignal(int, int, int, int)

    def __init__(self, parent, images, wavelengths,
                 min_x, max_x, min_y, max_y,
                 box_width, box_height, step_x, step_y,
                 total_boxes, interpolation_enabled, work_directory,
                 fix_s=False, fix_t=False, fix_eta=False):
        super().__init__()
        self.parent = parent
        self.images = images
        self.wavelengths = wavelengths
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.box_width = box_width
        self.box_height = box_height
        self.step_x = step_x
        self.step_y = step_y
        self.total_boxes = total_boxes
        self.interpolation_enabled = interpolation_enabled
        self.work_directory = work_directory
        self.fix_s = fix_s
        self.fix_t = fix_t
        self.fix_eta = fix_eta
        self.work_directory_short = self.get_short_path(self.work_directory, levels=2)
   

        self.stop_requested = False

        # Dimensions of the first image
        self.image_height, self.image_width = self.images[0].shape

        # We'll only look at up to 5 rows (or however many your table allows)
        max_rows = min(5, self.parent.bragg_table.rowCount())

        # Collect only the row indices that are "valid" (fully filled)
        self.valid_rows = []
        for row_idx in range(max_rows):
            if self.is_row_filled(row_idx):
                self.valid_rows.append(row_idx)

        self.num_edges = len(self.valid_rows)
        if self.num_edges == 0:
            self.message.emit("No edges in the bragg_table with valid data. Aborting.")
            return

        # Build a local hkl_list from the valid rows
        self.hkl_list = []
        for row_idx in self.valid_rows:
            hkl_item = self.parent.bragg_table.item(row_idx, 0)
            hkl_text = hkl_item.text().strip("()") if hkl_item else "0,0,0"
            try:
                h, k, l = map(int, hkl_text.split(","))
                self.hkl_list.append((h, k, l))
            except ValueError:
                # If user typed something invalid, skip or set default
                self.hkl_list.append((0, 0, 0))
        self.parent.hkl_list = self.hkl_list.copy()

        # Initialize arrays for storing final (Region 3) fit parameters
        # shape = (height, width, num_edges)
        self.a_array = None
        self.s_array = None
        self.t_array = None
        self.eta_array = None

        self.a_unc_array = None
        self.s_unc_array = None
        self.t_unc_array = None
        self.eta_unc_array = None

        # --- NEW: add a 3D array for edge heights ---
        self.height_array = None
        self.width_array = None

        self.initialize_arrays()

    def is_row_filled(self, row_idx):
        """
        Returns True if this row has non-empty data in the essential columns.
        """
        required_cols = [0, 2, 3, 4, 5, 6, 7, 8, 9,10]
        table = self.parent.bragg_table
        for col in required_cols:
            item = table.item(row_idx, col)
            if (not item) or (not item.text().strip()):
                return False
        return True

    def initialize_arrays(self):
        shape_3d = (self.image_height, self.image_width, self.num_edges)
        self.a_array = np.full(shape_3d, np.nan)
        self.s_array = np.full(shape_3d, np.nan)
        self.t_array = np.full(shape_3d, np.nan)
        self.eta_array = np.full(shape_3d, np.nan)
        self.width_array = np.full(shape_3d, np.nan)

        self.a_unc_array = np.full(shape_3d, np.nan)
        self.s_unc_array = np.full(shape_3d, np.nan)
        self.t_unc_array = np.full(shape_3d, np.nan)
        self.eta_unc_array = np.full(shape_3d, np.nan)

        # Also initialize the height array
        self.height_array = np.full(shape_3d, np.nan)

    def run(self):
        if self.num_edges == 0:
            self.finished.emit("No valid edges.")
            return

        box_counter = 0

        for y in range(self.min_y, self.max_y - self.box_height + 1, self.step_y):
            for x in range(self.min_x, self.max_x - self.box_width + 1, self.step_x):
                if self.stop_requested:
                    self.message.emit("Batch fit edges stopped by user.")
                    self.finished.emit("")
                    return

                box_counter += 1
                progress_percentage = int((box_counter / self.total_boxes) * 100)
                self.progress_updated.emit(progress_percentage)

                row_min = y
                row_max = y + self.box_height
                col_min = x
                col_max = x + self.box_width

                self.current_box_changed.emit(row_min, row_max, col_min, col_max)

                # Sum intensities in sub-area
                intensities = []
                for img in self.images:
                    sub_img = img[row_min:row_max, col_min:col_max]
                    intensities.append(sub_img.sum())
                self.parent.intensities = np.array(intensities)

                # We'll store the final fit at the center pixel
                center_row = row_min + self.box_height // 2
                center_col = col_min + self.box_width // 2

                # For each valid row => do fit_region
                self.parent.params_unknown = {}  # reset for each sub-area

                for i, row_idx in enumerate(self.valid_rows):
                    
                    self.parent.fit_region(row_idx, skip_ui_updates=True)
                    popt_3 = self.parent.params_unknown.get((row_idx, 3))
                    edge_height = self.parent.params_unknown.get((row_idx, 4), np.nan)
                    width = self.parent.params_unknown.get((row_idx, 5), np.nan)

                    if popt_3 is None:
                        # Region 3 fit failed => store NaNs
                        self.assign_nan_to_pixel(center_row, center_col, i)
                    else:
                        # popt_3 = (a_fit, s_fit, t_fit)
                        (a_fit, s_fit, t_fit, eta_fit,
                         a_unc, s_unc, t_unc, eta_unc) = popt_3
                        self.a_array[center_row, center_col, i] = a_fit
                        self.s_array[center_row, center_col, i] = s_fit
                        self.t_array[center_row, center_col, i] = t_fit
                        self.eta_array[center_row, center_col, i] = eta_fit
                        
                        self.width_array[center_row, center_col, i] = width

                        self.a_unc_array[center_row, center_col, i]   = a_unc
                        self.s_unc_array[center_row, center_col, i]   = s_unc
                        self.t_unc_array[center_row, center_col, i]   = t_unc
                        self.eta_unc_array[center_row, center_col, i] = eta_unc

                        # Also store the edge height
                        self.height_array[center_row, center_col, i] = edge_height

        # Check if we got any success
        if np.all(np.isnan(self.a_array)):
            self.message.emit("No successful fits performed. No results to save.")
            self.finished.emit("No fits.")
            return

        # Save ungridded
        self.save_results_to_csv_ungrid()

        # Interpolate if desired
        if self.interpolation_enabled:
            self.interpolate_results()

        # Save gridded
        self.save_results_to_csv()

        self.finished.emit("Batch fit edges completed.")

    def stop(self):
        self.stop_requested = True
        
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

    def assign_nan_to_pixel(self, r, c, edge_idx):
        self.a_array[r, c, edge_idx] = np.nan
        self.s_array[r, c, edge_idx] = np.nan
        self.t_array[r, c, edge_idx] = np.nan
        self.eta_array[r, c, edge_idx] = np.nan
        self.width_array[r, c, edge_idx] = np.nan
        # uncertainties
        self.a_unc_array[r, c, edge_idx] = np.nan
        self.s_unc_array[r, c, edge_idx] = np.nan
        self.t_unc_array[r, c, edge_idx] = np.nan
        self.eta_unc_array[r, c, edge_idx] = np.nan
        # edge height
        self.height_array[r, c, edge_idx] = np.nan

    def save_results_to_csv_ungrid(self):
        """
        Write out the ungridded (x,y => a,s,t,...) data with metadata,
        plus edge_height. 
        """
        import datetime, os
        import pandas as pd

        h, w, e = self.a_array.shape
        N = h * w

        # Flatten
        flat_a = self.a_array.reshape(N, e)
        flat_s = self.s_array.reshape(N, e)
        flat_t = self.t_array.reshape(N, e)
        flat_eta = self.eta_array.reshape(N, e)
        flat_width = self.width_array.reshape(N, e)
        flat_a_unc = self.a_unc_array.reshape(N, e)
        flat_s_unc = self.s_unc_array.reshape(N, e)
        flat_t_unc = self.t_unc_array.reshape(N, e)
        flat_eta_unc = self.eta_unc_array.reshape(N, e)

        flat_height = self.height_array.reshape(N, e)  # <--- new

        yy, xx = np.mgrid[0:h, 0:w]
        X_flat = xx.flatten()
        Y_flat = yy.flatten()

        if not hasattr(self.parent, 'hkl_list') or len(self.parent.hkl_list) != e:
            self.parent.hkl_list = [("edge", i + 1) for i in range(e)]

        data_dict = {"x": X_flat, "y": Y_flat}

        for i in range(e):
            (h_val, k_val, *rest) = self.parent.hkl_list[i]
            # Flatten out a, s, t
            a_i = flat_a[:, i]
            s_i = flat_s[:, i]
            t_i = flat_t[:, i]
            eta_i = flat_eta[:, i]
            width_i = flat_width[:, i]
            height_i = flat_height[:, i] 
            a_unc_i = flat_a_unc[:, i]
            s_unc_i = flat_s_unc[:, i]
            t_unc_i = flat_t_unc[:, i]
            eta_unc_i = flat_eta_unc[:, i]
             # <--- new

            # Build column names
            a_col_name = f"d_{h_val}{k_val}"
            s_col_name = f"s_{h_val}{k_val}"
            t_col_name = f"t_{h_val}{k_val}"
            eta_col_name = f"eta_{h_val}{k_val}"
            width_col_name = f"fwhm_{h_val}{k_val}"
            height_col = f"height_{h_val}{k_val}"
            a_unc_col = f"d_unc_{h_val}{k_val}"
            s_unc_col = f"s_unc_{h_val}{k_val}"
            t_unc_col = f"t_unc_{h_val}{k_val}"
            eta_unc_col = f"eta_unc_{h_val}{k_val}"
              # <--- new

            data_dict[a_col_name] = a_i
            data_dict[s_col_name] = s_i
            data_dict[t_col_name] = t_i
            data_dict[eta_col_name] = eta_i
            data_dict[width_col_name] = width_i
            data_dict[a_unc_col] = a_unc_i
            data_dict[s_unc_col] = s_unc_i
            data_dict[t_unc_col] = t_unc_i
            data_dict[eta_unc_col] = eta_unc_i
            data_dict[height_col] = height_i  # <--- new

        df = pd.DataFrame(data_dict)

        # Build metadata
        metadata = [
            ("box_width", self.box_width),
            ("box_height", self.box_height),
            ("step_x", self.step_x),
            ("step_y", self.step_y),
            ("roi_x_min", self.min_x),
            ("roi_x_max", self.max_x),
            ("roi_y_min", self.min_y),
            ("roi_y_max", self.max_y),
            ("number_of_edges", self.num_edges),
            ("directory", self.work_directory_short),
            ("fix_s", self.fix_s),
            ("fix_t", self.fix_t),
            ("fix_eta", self.fix_eta),
            ("flight_path", self.parent.flight_path),
            ("selected_phase", self.parent.phase_dropdown.currentText()),
        ]

        table = self.parent.bragg_table
        for row_idx in range(table.rowCount()):
            row_items = []
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                if cell_item is not None:
                    text_val = cell_item.text().replace(",", ";")
                else:
                    text_val = ""
                row_items.append(text_val)
            row_text = "|".join(row_items)
            metadata.append((f"bragg_table_row_{row_idx+1}", row_text))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_edges_ungridded_{timestamp}.csv"
        file_path = os.path.join(self.work_directory, filename)
        file_path_short = self.get_short_path(file_path, levels=2)

        try:
            with open(file_path, "w", newline="") as f:
                f.write("Metadata Name,Metadata Value\n")
                for key, value in metadata:
                    f.write(f"{key},{value}\n")
                f.write("\n")
                df.to_csv(f, index=False)

            self.message.emit(f"Ungridded results (edges) saved to {file_path_short}")
        except Exception as e:
            self.message.emit(f"Failed to save ungridded edges results: {e}")

    def save_results_to_csv(self):
        """
        Save the final arrays again, but now presumably after interpolation,
        also including edge_height.
        """
        import os
        import pandas as pd

        h, w, e = self.a_array.shape
        N = h * w

        flat_a = self.a_array.reshape(N, e)
        flat_s = self.s_array.reshape(N, e)
        flat_t = self.t_array.reshape(N, e)
        flat_eta = self.eta_array.reshape(N, e)
        flat_width = self.width_array.reshape(N, e)
        flat_height = self.height_array.reshape(N, e)
        flat_a_unc = self.a_unc_array.reshape(N, e)
        flat_s_unc = self.s_unc_array.reshape(N, e)
        flat_t_unc = self.t_unc_array.reshape(N, e)
        flat_eta_unc = self.eta_unc_array.reshape(N, e)        

        yy, xx = np.mgrid[0:h, 0:w]
        X_flat = xx.flatten()
        Y_flat = yy.flatten()

        data_dict = {"x": X_flat, "y": Y_flat}

        # if not hasattr(self.parent, 'hkl_list') or len(self.parent.hkl_list) != e:
        #     self.parent.hkl_list = [("edge", i + 1) for i in range(e)]

        for i, hkl in enumerate(self.hkl_list):
            # hkl is either (h,k,l) or the fallback ('edge', n)
            if len(hkl) == 3 and isinstance(hkl[0], (int, np.integer)):
                hkl_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"   # e.g. 2,2,1 -> "221"
            else:
                hkl_str = f"edge{i+1}"
        
            a_col_name     = f"d_{hkl_str}"
            s_col_name     = f"s_{hkl_str}"
            t_col_name     = f"t_{hkl_str}"
            eta_col_name   = f"eta_{hkl_str}"
            width_col_name = f"fwhm_{hkl_str}"
            height_col     = f"height_{hkl_str}"
            a_unc_col      = f"d_unc_{hkl_str}"
            s_unc_col      = f"s_unc_{hkl_str}"
            t_unc_col      = f"t_unc_{hkl_str}"
            eta_unc_col    = f"eta_unc_{hkl_str}"
            
            

        # for i in range(e):
            (h_val, k_val, *rest) = self.parent.hkl_list[i]
            a_i = flat_a[:, i]
            s_i = flat_s[:, i]
            t_i = flat_t[:, i]
            eta_i = flat_eta[:, i]
            width_i = flat_width[:, i]
            height_i = flat_height[:, i]
            a_unc_i = flat_a_unc[:, i]
            s_unc_i = flat_s_unc[:, i]
            t_unc_i = flat_t_unc[:, i]
            eta_unc_i = flat_eta_unc[:, i]
              # <--- new

            data_dict[a_col_name] = a_i
            data_dict[s_col_name] = s_i
            data_dict[t_col_name] = t_i
            data_dict[eta_col_name] = eta_i
            data_dict[width_col_name] = width_i
            data_dict[height_col] = height_i
            data_dict[a_unc_col] = a_unc_i
            data_dict[s_unc_col] = s_unc_i
            data_dict[t_unc_col] = t_unc_i
            data_dict[eta_unc_col] = eta_unc_i
            

        df = pd.DataFrame(data_dict)

        # Extended metadata
        metadata = [
            ("box_width", self.box_width),
            ("box_height", self.box_height),
            ("step_x", self.step_x),
            ("step_y", self.step_y),
            ("roi_x_min", self.min_x),
            ("roi_x_max", self.max_x),
            ("roi_y_min", self.min_y),
            ("roi_y_max", self.max_y),
            ("directory", self.work_directory_short),
            ("fix_s", self.fix_s),
            ("fix_t", self.fix_t),
            ("fix_eta", self.fix_eta),
            ("flight_path", self.parent.flight_path),
            ("selected_phase", self.parent.phase_dropdown.currentText()),
        ]
        if self.num_edges is not None:
            metadata.append(("number_of_edges", self.num_edges))

        table = self.parent.bragg_table
        for row_idx in range(table.rowCount()):
            row_items = []
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                if cell_item is not None:
                    text_val = cell_item.text().replace(",", ";")
                else:
                    text_val = ""
                row_items.append(text_val)
            row_text = "|".join(row_items)
            metadata.append((f"bragg_table_row_{row_idx+1}", row_text))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_edges_gridded_{timestamp}.csv"
        filepath = os.path.join(self.work_directory, filename)
        filepath_short = self.get_short_path(filepath, levels=2)

        try:
            with open(filepath, "w", newline="") as f:
                f.write("Metadata Name,Metadata Value\n")
                for key, value in metadata:
                    f.write(f"{key},{value}\n")
                f.write("\n")
                df.to_csv(f, index=False)

            self.message.emit(f"Gridded edges results saved to {filepath_short}")
        except Exception as e:
            self.message.emit(f"Failed to save gridded edges results: {e}")

   
    def interpolate_results(self):
        """
        Interpolate self.*_array and *_unc_array inside (min_x,max_x,min_y,max_y).
        When there are too few points or Qhull fails, fall back to 'nearest'.
        """
        # from scipy.interpolate import griddata
        from scipy.spatial.qhull import QhullError
        import warnings
    
        # ------------------------------------------------------------
        # Prepare common grids and area mask
        # ------------------------------------------------------------
        grid_x, grid_y = np.meshgrid(
            np.arange(self.image_width), np.arange(self.image_height)
        )
        area_mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        area_mask[self.min_y:self.max_y, self.min_x:self.max_x] = True
    
        # ------------------------------------------------------------
        # Helper with robust fallback
        # ------------------------------------------------------------
        def interpolate_2d(arr, edge_id, label):
            """Return a copy of arr where NaNs inside area_mask are filled."""
            mask_valid = ~np.isnan(arr) & area_mask
            n_points   = np.count_nonzero(mask_valid)
    
            if n_points < 4:                     # too few for Qhull
                # self.message.emit(
                #     f"Edge {edge_id}: Not enough {label} points ({n_points}) "
                #     "for linear interpolation – skipped.")
                return arr
    
            pts  = np.column_stack((grid_x[mask_valid], grid_y[mask_valid]))
            vals = arr[mask_valid]
    
            try:
                interp_vals = griddata(
                    pts, vals,
                    (grid_x[area_mask], grid_y[area_mask]),
                    method='linear')
            except (QhullError, ValueError) as err:
                # Fall back to safer 'nearest' method
                warnings.warn(str(err), RuntimeWarning, stacklevel=2)
                self.message.emit(
                    f"Edge {edge_id}: Linear interpolation failed "
                    f"({err.__class__.__name__}) – using nearest neighbour.")
                interp_vals = griddata(
                    pts, vals,
                    (grid_x[area_mask], grid_y[area_mask]),
                    method='nearest')
    
            new_arr = arr.copy()
            new_arr[area_mask] = interp_vals
            return new_arr
    
        # ------------------------------------------------------------
        # Interpolate each array (a, s, t, eta, height) edge‑by‑edge
        # ------------------------------------------------------------
        for i in range(self.num_edges):
            # ----- a -------------------------------------------------
            self.a_array[:, :, i]     = interpolate_2d(self.a_array[:, :, i],     i, "a")
            self.a_unc_array[:, :, i] = interpolate_2d(self.a_unc_array[:, :, i], i, "a‑unc")
    
            # ----- s -------------------------------------------------
            self.s_array[:, :, i]     = interpolate_2d(self.s_array[:, :, i],     i, "s")
            self.s_unc_array[:, :, i] = interpolate_2d(self.s_unc_array[:, :, i], i, "s‑unc")
    
            # ----- t -------------------------------------------------
            self.t_array[:, :, i]     = interpolate_2d(self.t_array[:, :, i],     i, "t")
            self.t_unc_array[:, :, i] = interpolate_2d(self.t_unc_array[:, :, i], i, "t‑unc")
    
            # ----- eta (bug‑fix: test eta_slice, not t_slice) --------
            self.eta_array[:, :, i]     = interpolate_2d(self.eta_array[:, :, i],     i, "eta")
            self.eta_unc_array[:, :, i] = interpolate_2d(self.eta_unc_array[:, :, i], i, "eta‑unc")
    
            # ----- height -------------------------------------------
            self.height_array[:, :, i]  = interpolate_2d(self.height_array[:, :, i], i, "height")
            self.width_array[:, :, i]  = interpolate_2d(self.width_array[:, :, i], i, "width")

        self.message.emit("Interpolation (edges) completed successfully.")


class BatchFitWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)
    message = pyqtSignal(str)
    current_box_changed = pyqtSignal(int, int, int, int)

    def __init__(
        self,
        parent,
        images,
        wavelengths,
        min_x,
        max_x,
        min_y,
        max_y,
        box_width,
        box_height,
        step_x,
        step_y,
        total_boxes,
        interpolation_enabled,
        work_directory,
        fix_s=False,
        fix_t=False,
        fix_eta=False
    ):
        super().__init__()
        self.parent = parent
        self.images = images
        self.wavelengths = wavelengths
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.box_width = box_width
        self.box_height = box_height
        self.step_x = step_x
        self.step_y = step_y
        self.total_boxes = total_boxes
        self.stop_requested = False
        self.interpolation_enabled = interpolation_enabled
        self.work_directory = work_directory
        self.fix_s = fix_s
        self.fix_t = fix_t
        self.fix_eta = fix_eta

        # All images same shape
        self.image_height, self.image_width = self.images[0].shape

        # We'll discover lattice parameters after first success
        self.param_arrays = {}      # e.g. { "a": 2D array, "b":..., etc. }
        self.param_unc_arrays = {}
        self.num_edges = None
        self.s_array = None
        self.s_unc_array = None
        self.t_array = None
        self.t_unc_array = None
        self.eta_array = None
        self.eta_unc_array = None

        # New array for edge heights:
        self.height_array = None
        self.width_array = None

    def run(self):
        box_counter = 0
        params_initialized = False

        for y in range(self.min_y, self.max_y - self.box_height + 1, self.step_y):
            for x in range(self.min_x, self.max_x - self.box_width + 1, self.step_x):
                if self.stop_requested:
                    self.message.emit("Batch fitting stopped by user.")
                    self.finished.emit("")
                    return

                box_counter += 1
                progress_percentage = int((box_counter / self.total_boxes) * 100)
                self.progress_updated.emit(progress_percentage)

                row_min = y
                row_max = y + self.box_height
                col_min = x
                col_max = x + self.box_width

                self.current_box_changed.emit(row_min, row_max, col_min, col_max)

                # Sum intensities in sub-area
                intensities = []
                for img in self.images:
                    sub_img = img[row_min:row_max, col_min:col_max]
                    intensities.append(sub_img.sum())
                self.parent.intensities = np.array(intensities)

                center_row = row_min + self.box_height // 2
                center_col = col_min + self.box_width // 2

                # Call fit_full_pattern_core
                try:
                    result_dict, error_msg = self.parent.fit_full_pattern_core(
                        fix_s=self.fix_s,
                        fix_t=self.fix_t,
                        fix_eta=self.fix_eta
                    )
                except Exception as e:
                    error_msg = f"Unexpected error during fitting: {e}"
                    result_dict = None

                if error_msg or not result_dict or not result_dict.get("success", False):
                    # Fill NaNs
                    self.assign_nan_to_pixel(center_row, center_col)
                    continue

                # -------------- Fit Succeeded --------------
                lattice_params = result_dict["lattice_params"]
                lattice_uncs   = result_dict["lattice_uncertainties"]

                fitted_s_dict = result_dict["fitted_s"]
                fitted_t_dict = result_dict["fitted_t"]
                fitted_eta_dict = result_dict["fitted_eta"]
                s_unc_dict     = result_dict["s_uncertainties"]
                t_unc_dict     = result_dict["t_uncertainties"]
                eta_unc_dict     = result_dict["eta_uncertainties"]
                bragg_edges    = result_dict.get("bragg_edges", [])
                ab_fits = result_dict['ab_fits']

                # Update parent's hkl_list
                if bragg_edges:
                    hkl_list = [edge["hkl"] for edge in bragg_edges]
                    self.parent.hkl_list = hkl_list
                else:
                    self.parent.hkl_list = sorted(fitted_s_dict.keys())

                # If first success => allocate arrays
                if not params_initialized:
                    for p_name in lattice_params.keys():
                        self.param_arrays[p_name] = np.full(
                            (self.image_height, self.image_width), np.nan
                        )
                        self.param_unc_arrays[p_name] = np.full(
                            (self.image_height, self.image_width), np.nan
                        )
                    self.num_edges = len(fitted_s_dict)
                    self.initialize_s_t_arrays()  # sets up s,t + height_array
                    params_initialized = True

                # 1) Store lattice param at center pixel
                for p_name, p_val in lattice_params.items():
                    self.param_arrays[p_name][center_row, center_col] = p_val
                for p_name, p_unc_val in lattice_uncs.items():
                    self.param_unc_arrays[p_name][center_row, center_col] = p_unc_val

                # 2) Store s,t in arrays
                hkl_list = self.parent.hkl_list
                for i, hkl in enumerate(hkl_list):
                    s_val = fitted_s_dict.get(hkl, np.nan)
                    t_val = fitted_t_dict.get(hkl, np.nan)
                    eta_val = fitted_eta_dict.get(hkl, np.nan)
                    s_unc_val = s_unc_dict.get(hkl, np.nan)
                    t_unc_val = t_unc_dict.get(hkl, np.nan)
                    eta_unc_val = eta_unc_dict.get(hkl, np.nan)
                    self.s_array[center_row, center_col, i] = s_val
                    self.s_unc_array[center_row, center_col, i] = s_unc_val
                    self.t_array[center_row, center_col, i] = t_val
                    self.t_unc_array[center_row, center_col, i] = t_unc_val
                    self.eta_array[center_row, center_col, i] = eta_val
                    self.eta_unc_array[center_row, center_col, i] = eta_unc_val

                # 3) Compute edge heights for each edge
                #    (region1 - region2) at the final x_edge => store in self.height_array
                self.compute_and_store_heights(center_row, center_col, bragg_edges, lattice_params, fitted_s_dict, fitted_t_dict, fitted_eta_dict, ab_fits)

        # Done => if never initialized => no success
        if not params_initialized:
            self.message.emit("No successful fits performed. No results to save.")
            self.finished.emit("No fits.")
            return

        # Save ungridded
        self.save_results_to_csv_ungrid()

        # Interpolate if needed
        if self.interpolation_enabled:
            self.interpolate_results()

        # Save gridded
        self.save_results_to_csv()

        gc.collect()
        self.finished.emit("Batch fitting completed.")

    def stop(self):
        self.stop_requested = True

    def initialize_s_t_arrays(self):
        """Initialize s,t arrays + height_array once we know num_edges."""
        H, W = self.image_height, self.image_width
        N = self.num_edges
        self.s_array = np.full((H, W, N), np.nan)
        self.s_unc_array = np.full((H, W, N), np.nan)
        self.t_array = np.full((H, W, N), np.nan)
        self.t_unc_array = np.full((H, W, N), np.nan)
        self.eta_array = np.full((H, W, N), np.nan)
        self.eta_unc_array = np.full((H, W, N), np.nan)

        # also a height_array:
        self.height_array = np.full((H, W, N), np.nan)
        self.width_array  = np.full((H, W, N), np.nan)

    def assign_nan_to_pixel(self, r, c):
        """Fill NaNs for all lattice params, s,t, & height at pixel (r,c)."""
        for p_name in self.param_arrays.keys():
            self.param_arrays[p_name][r, c] = np.nan
            self.param_unc_arrays[p_name][r, c] = np.nan

        if self.s_array is not None:
            self.s_array[r, c, :] = np.nan
            self.s_unc_array[r, c, :] = np.nan
        if self.t_array is not None:
            self.t_array[r, c, :] = np.nan
            self.t_unc_array[r, c, :] = np.nan
            
        if self.eta_array is not None:
            self.eta_array[r, c, :] = np.nan
            self.eta_unc_array[r, c, :] = np.nan

        if hasattr(self, "height_array") and self.height_array is not None:
            self.height_array[r, c, :] = np.nan
            
        if hasattr(self, "width_array") and self.width_array is not None:
            self.width_array[r, c, :]  = np.nan

    def compute_and_store_heights(self, row_c, col_c, bragg_edges, lattice_params, fitted_s_dict, fitted_t_dict, fitted_eta_dict, ab_fits):
        """
        For each Bragg edge in bragg_edges, compute the "edge height"
        f1(x_edge) - f2(x_edge), store in self.height_array[row_c, col_c, i].
        """
        # If there's no edge, skip
        if not bragg_edges or not self.num_edges:
            return

        # We'll assume the order in bragg_edges matches self.parent.hkl_list
        # or at least that i lines up with each edge
        from math import exp, sqrt

        def d_spacing(h, k, l, structure, lat):
            if structure == "cubic":
                a_val = lat["a"]
                return a_val / sqrt(h**2 + k**2 + l**2)
            elif structure == "tetragonal":
                a_val = lat["a"]
                c_val = lat["c"]
                return 1.0 / sqrt((h**2 + k**2)/(a_val**2) + (l**2)/(c_val**2))
            elif structure == "hexagonal":
                a_val = lat["a"]
                c_val = lat["c"]
                return 1.0 / sqrt((4.0/3.0)*((h**2 + h*k + k**2)/(a_val**2)) + (l**2)/(c_val**2))
            elif structure == "orthorhombic":
                a_val = lat["a"]
                b_val = lat["b"]
                c_val = lat["c"]
                return 1.0 / sqrt((h**2)/(a_val**2)+(k**2)/(b_val**2)+(l**2)/(c_val**2))
            return float("nan")

        structure_type = getattr(self.parent, "structure_type", "cubic")

        for i, edge in enumerate(bragg_edges):
            (h, k, l) = edge["hkl"]
            # a0   = edge["a0"]
            # b0   = edge["b0"]
            # a_hk = edge["a_hkl"]
            # b_hk = edge["b_hkl"]

            # Compute x_edge = 2 * d_hkl from final lattice param
            d_val = d_spacing(h, k, l, structure_type, lattice_params)
            # if not d_val or d_val <= 0:
            #     self.height_array[row_c, col_c, i] = np.nan
            #     continue
            try:
                a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_fits[edge["hkl"]]
            except KeyError:
                # safety fallback – skip if not present
                self.height_array[row_c, col_c, i] = np.nan
                self.width_array[row_c,  col_c, i] = np.nan
                continue

            x_edge = 2.0 * d_val
           
            
           # ---------------- width  (FWHM of derivative) --------------
            s = fitted_s_dict.get(edge["hkl"],  np.nan)
            t = fitted_t_dict.get(edge["hkl"],  np.nan)
            η = fitted_eta_dict.get(edge["hkl"], np.nan)
    
            span  = 0.2                 # Å on each side of x_edge
            xx    = np.linspace(x_edge-span, x_edge+span, 14000)
            yy    = fitting_function_3(
                        xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                        s,  t,  η, [],             # empty hkl_list
                        xx.min(), xx.max(),
                        "cubic", {"a": x_edge/2})  # dummy lattice
    
            try:
                dy = np.gradient(yy, xx)
                half = dy.max() / 2
                left = xx[dy >= half][0]
                right = xx[dy >= half][-1]
                edge_width = right - left
                edge_height = yy.max() - yy.min()
            except:
                edge_width = np.nan
                edge_height = np.nan
            
    
            self.width_array[row_c, col_c, i] = edge_width
            self.height_array[row_c, col_c, i] = edge_height

    def save_results_to_csv_ungrid(self):
        """
        Save 'ungridded' results by flattening all arrays + adding extended metadata.
        Now also includes the flattened 'height_array'.
        """
        import datetime, os
        import pandas as pd

        H, W = self.image_height, self.image_width
        if self.num_edges is None:
            return

        N = H * W
        # Lattice param flatten
        X_coords = np.tile(np.arange(self.image_width), self.image_height)
        Y_coords = np.repeat(np.arange(self.image_height), self.image_width)
        df = pd.DataFrame({"x": X_coords, "y": Y_coords})

        # Flatten param_arrays
        for p_name, arr_2d in self.param_arrays.items():
            df[p_name] = arr_2d.flatten()
            df[f"{p_name}_unc"] = self.param_unc_arrays[p_name].flatten()

        # Flatten s/t
        flat_s    = self.s_array.reshape(N, self.num_edges)
        flat_s_unc= self.s_unc_array.reshape(N, self.num_edges)
        flat_t    = self.t_array.reshape(N, self.num_edges)
        flat_t_unc= self.t_unc_array.reshape(N, self.num_edges)
        flat_eta    = self.eta_array.reshape(N, self.num_edges)
        flat_eta_unc= self.eta_unc_array.reshape(N, self.num_edges)
        flat_width = self.width_array.reshape(N, self.num_edges)


        # Flatten height
        flat_height = None
        if self.height_array is not None:
            flat_height = self.height_array.reshape(N, self.num_edges)

        # Build columns
        hkl_list = getattr(self.parent, "hkl_list", [])
        if len(hkl_list) != self.num_edges:
            # fallback
            hkl_list = [("edge", i+1, 0) for i in range(self.num_edges)]

        for i, hkl in enumerate(hkl_list):
            hkl_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"
            df[f"s_{hkl_str}"] = flat_s[:, i]
            
            df[f"t_{hkl_str}"] = flat_t[:, i]
            
            df[f"eta_{hkl_str}"] = flat_eta[:, i]
           
            df[f"fwhm_{hkl_str}"] = flat_width[:, i]


            # also store height
            if flat_height is not None:
                df[f"height_{hkl_str}"] = flat_height[:, i]
            df[f"s_unc_{hkl_str}"] = flat_s_unc[:, i]
            df[f"t_unc_{hkl_str}"] = flat_t_unc[:, i]
            df[f"eta_unc_{hkl_str}"] = flat_eta_unc[:, i]

        # Build metadata
        metadata = [
            ("box_width", self.box_width),
            ("box_height", self.box_height),
            ("step_x", self.step_x),
            ("step_y", self.step_y),
            ("roi_x_min", self.min_x),
            ("roi_x_max", self.max_x),
            ("roi_y_min", self.min_y),
            ("roi_y_max", self.max_y),
            ("directory", self.work_directory),
            ("fix_s", self.fix_s),
            ("fix_t", self.fix_t),
            ("fix_eta", self.fix_eta),
            ("flight_path", self.parent.flight_path),
            ("selected_phase", self.parent.phase_dropdown.currentText()),
            ("number_of_edges", self.num_edges),
        ]

        # Bragg table rows
        table = self.parent.bragg_table
        for row_idx in range(table.rowCount()):
            row_items = []
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                val = cell_item.text().replace(",",";") if cell_item else ""
                row_items.append(val)
            row_text = "|".join(row_items)
            metadata.append((f"bragg_table_row_{row_idx+1}", row_text))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_ungridded_{timestamp}.csv"
        file_path = os.path.join(self.work_directory, filename)

        try:
            with open(file_path, "w", newline="") as f:
                f.write("Metadata Name,Metadata Value\n")
                for key, value in metadata:
                    f.write(f"{key},{value}\n")
                f.write("\n")
                df.to_csv(f, index=False)

            self.message.emit(f"Ungridded results saved to {file_path}")
        except Exception as e:
            self.message.emit(f"Failed to save ungridded results: {e}")

    def save_results_to_csv(self):
        """
        Save gridded results after interpolation, also including the height_array.
        """
        import datetime, os
        import pandas as pd

        H, W = self.image_height, self.image_width
        if self.num_edges is None:
            return

        N = H * W
        X_coords = np.tile(np.arange(self.image_width), self.image_height)
        Y_coords = np.repeat(np.arange(self.image_height), self.image_width)
        df = pd.DataFrame({"x": X_coords, "y": Y_coords})

        # flatten param_arrays
        for p_name, arr_2d in self.param_arrays.items():
            df[p_name] = arr_2d.flatten()
            df[f"{p_name}_unc"] = self.param_unc_arrays[p_name].flatten()

        flat_s    = self.s_array.reshape(N, self.num_edges)
        flat_s_unc= self.s_unc_array.reshape(N, self.num_edges)
        flat_t    = self.t_array.reshape(N, self.num_edges)
        flat_t_unc= self.t_unc_array.reshape(N, self.num_edges)
        flat_eta    = self.eta_array.reshape(N, self.num_edges)
        flat_eta_unc= self.eta_unc_array.reshape(N, self.num_edges)
        flat_width = self.width_array.reshape(N, self.num_edges)

        flat_height = None
        if self.height_array is not None:
            flat_height = self.height_array.reshape(N, self.num_edges)

        # columns
        hkl_list = getattr(self.parent, "hkl_list", [])
        if len(hkl_list) != self.num_edges:
            hkl_list = [("edge", i+1, 0) for i in range(self.num_edges)]

        for i, hkl in enumerate(hkl_list):
            hkl_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"
            df[f"s_{hkl_str}"] = flat_s[:, i]
            
            df[f"t_{hkl_str}"] = flat_t[:, i]
            
            df[f"eta_{hkl_str}"] = flat_eta[:, i]
            
            df[f"fwhm_{hkl_str}"] = flat_width[:, i]


            # add height
            if flat_height is not None:
                df[f"height_{hkl_str}"] = flat_height[:, i]
            df[f"s_unc_{hkl_str}"] = flat_s_unc[:, i]
            df[f"t_unc_{hkl_str}"] = flat_t_unc[:, i]
            df[f"eta_unc_{hkl_str}"] = flat_eta_unc[:, i]

        # metadata
        metadata = [
            ("box_width", self.box_width),
            ("box_height", self.box_height),
            ("step_x", self.step_x),
            ("step_y", self.step_y),
            ("roi_x_min", self.min_x),
            ("roi_x_max", self.max_x),
            ("roi_y_min", self.min_y),
            ("roi_y_max", self.max_y),
            ("directory", self.work_directory),
            ("fix_s", self.fix_s),
            ("fix_t", self.fix_t),
            ("fix_eta", self.fix_eta),
            ("flight_path", self.parent.flight_path),
            ("selected_phase", self.parent.phase_dropdown.currentText()),
            ("number_of_edges", self.num_edges),
        ]

        table = self.parent.bragg_table
        for row_idx in range(table.rowCount()):
            row_items = []
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                val = cell_item.text().replace(",",";") if cell_item else ""
                row_items.append(val)
            row_text = "|".join(row_items)
            metadata.append((f"bragg_table_row_{row_idx+1}", row_text))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_gridded_{timestamp}.csv"
        file_path = os.path.join(self.work_directory, filename)

        try:
            with open(file_path, "w", newline="") as f:
                f.write("Metadata Name,Metadata Value\n")
                for key, value in metadata:
                    f.write(f"{key},{value}\n")
                f.write("\n")
                df.to_csv(f, index=False)

            self.message.emit(f"Gridded results saved to {file_path}")
        except Exception as e:
            self.message.emit(f"Failed to save gridded results: {e}")

    def interpolate_results(self):
        """
        Interpolate the arrays to fill NaNs, including the height_array.
        """
        H, W = self.image_height, self.image_width
        from scipy.interpolate import griddata

        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        area_mask = np.zeros((H, W), dtype=bool)
        area_mask[self.min_y:self.max_y, self.min_x:self.max_x] = True

        def interpolate_2d(array_2d):
            mask_valid = ~np.isnan(array_2d) & area_mask
            if not np.any(mask_valid):
                return array_2d
            pts = np.column_stack((grid_x[mask_valid], grid_y[mask_valid]))
            vals = array_2d[mask_valid]
            interp_area = griddata(pts, vals, (grid_x[area_mask], grid_y[area_mask]), method='linear')
            array_new = array_2d.copy()
            array_new[area_mask] = interp_area
            return array_new

        # lattice param arrays
        for p_name in self.param_arrays.keys():
            arr_2d = self.param_arrays[p_name]
            arr_2d_unc = self.param_unc_arrays[p_name]
            self.param_arrays[p_name]     = interpolate_2d(arr_2d)
            self.param_unc_arrays[p_name] = interpolate_2d(arr_2d_unc)

        # s,t arrays + height array
        if self.num_edges is not None:
            for i in range(self.num_edges):
                s_slice = self.s_array[:, :, i]
                t_slice = self.t_array[:, :, i]
                s_unc_slice = self.s_unc_array[:, :, i]
                t_unc_slice = self.t_unc_array[:, :, i]
                self.s_array[:, :, i]     = interpolate_2d(s_slice)
                self.t_array[:, :, i]     = interpolate_2d(t_slice)
                self.s_unc_array[:, :, i] = interpolate_2d(s_unc_slice)
                self.t_unc_array[:, :, i] = interpolate_2d(t_unc_slice)
                eta_slice = self.eta_array[:, :, i]
                eta_unc_slice = self.eta_unc_array[:, :, i]
                self.eta_array[:, :, i]     = interpolate_2d(eta_slice)
                self.eta_unc_array[:, :, i] = interpolate_2d(eta_unc_slice)
                self.width_array[:, :, i] = interpolate_2d(self.width_array[:, :, i])


            # Also do height_array
            if self.height_array is not None:
                for i in range(self.num_edges):
                    h_slice = self.height_array[:, :, i]
                    self.height_array[:, :, i] = interpolate_2d(h_slice)

        self.message.emit("Interpolation completed successfully.")

class ImageLoadWorker(QThread):
    progress_updated = pyqtSignal(int)  # Emits progress percentage
    run_loaded = pyqtSignal(str, dict)   # Emits folder_path and loaded images
    finished = pyqtSignal()              # Emits when loading is finished
    message = pyqtSignal(str)            # Emits messages for user feedback

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            normalized_path = os.path.normpath(self.folder_path)
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) >= 2:
                short_path = os.path.join(path_parts[-2], path_parts[-1])
            else:
                short_path = normalized_path
            
            # short_path = self.get_short_path(self.folder_path, levels=2)
            # List all FITS files in the folder
            fits_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.fits', '.fit'))]
            if not fits_files:
                self.message.emit(f"No FITS files found in folder: \\{short_path}")
                self.finished.emit()
                return

            # Sort files to ensure suffixes are aligned
            fits_files.sort()

            total_files = len(fits_files)
            run_dict = {}

            for idx, file in enumerate(fits_files):
                # Extract suffix as per naming convention
                basename = os.path.splitext(file)[0]
                parts = basename.split('_')
                if len(parts) < 2:
                    self.message.emit(f"Filename '{file}' does not contain an underscore-separated suffix. Skipping.")
                    continue
                suffix = parts[-1]

                # Validate suffix: must be a five-digit number between 00000 and 02924
                if not suffix.isdigit() or len(suffix) != 5:
                    self.message.emit(f"Suffix '{suffix}' in filename '{file}' is not a valid five-digit number. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue
                # suffix_int = int(suffix)
                # if suffix_int < 0 or suffix_int > 2924:
                #     self.message.emit(f"Suffix '{suffix}' in filename '{file}' is outside the allowed range _00000 to _02924. Skipping.")
                #     continue

                # Construct the full path
                file_path = os.path.join(self.folder_path, file)
                # Load the image data using fits.getdata to ensure the file is closed immediately
                try:
                    image_data = fits.getdata(file_path)
                    if image_data is None:
                        self.message.emit(f"No image data found in file '{file}'. Skipping.")
                        continue
                except Exception as e:
                    self.message.emit(f"Failed to read data from '{file}': {e}. Skipping.")
                    continue

                # Handle duplicate suffixes
                if suffix in run_dict:
                    self.message.emit(f"Duplicate suffix '{suffix}' found in file '{file}'. Previous image will be overwritten.")
                run_dict[suffix] = image_data

                # Update progress
                progress = int(((idx + 1) / total_files) * 100)
                self.progress_updated.emit(progress)

            if not run_dict:
                self.message.emit(f"No valid FITS images with suffixes _00000 to _02924 were found in folder: {self.folder_path}")
            else:
                self.message.emit(f"Successfully loaded {len(run_dict)} images from \\{short_path}")
                # Emit both folder_path and run_dict
                self.run_loaded.emit(self.folder_path, run_dict)

            # Explicitly call garbage collector to ensure all file handles are released
            gc.collect()

        except Exception as e:
            self.message.emit(f"Error loading images from \\{short_path}: {e}")

        self.finished.emit()


# Create a Matplotlib canvas to embed in the PyQt5 window
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

            
# Dialog class for individual fit visualization
class FitVisualizationDialog(QDialog):
    """
    Shows the full‑pattern fit plus dashed Region‑3 edge fits.
    A second text box now includes d‑spacing for each edge.
    """
    def __init__(self, x_pos, y_pos, box_width, box_height,
                 parameters, parent=None):
        super().__init__(parent)
        self.x_pos      = x_pos
        self.y_pos      = y_pos
        self.box_width  = box_width
        self.box_height = box_height
        self.parameters = parameters

        # ───── Window & widgets ───────────────────────────────────
        self.setWindowTitle(f"Full‑Pattern Fit at ({x_pos},{y_pos})")
        self.setGeometry(200, 200, 1200, 1000)

        main_layout = QVBoxLayout(self)

        self.canvas  = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_plot)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

        self.plot_fit()                 # ── do the drawing

    # ------------------------------------------------------------------
    def plot_fit(self):
        """Draw experimental data, full‑pattern model and edge curves."""
        ax = self.canvas.axes
        ax.clear()

        p = self.parameters
        x_exp, y_exp    = p['x_exp'],  p['y_exp']
        model_x, model_y= p['model_x'], p['model_y']

        # Experimental data & global pattern
        ax.plot(x_exp, y_exp, 'bo', ms=4, label="Experimental")
        if len(model_x):
            ax.plot(model_x, model_y, 'k-', alpha=0.3, lw=10, 
                    # label="Fitted pattern"
                    )

        # Individual Region‑3 edge fits (dashed)
        for ef in p.get('edge_fits', []):
            ax.plot(ef['x'], ef['fit'], '--', lw=4,
                    # label=f"Fitted Edge {ef['hkl']}"
                    )

        # Axes cosmetics
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Summed Intensity")
        ax.set_title(f"ROI centre ({self.x_pos},{self.y_pos})  •  "
                     f"Box {self.box_width}×{self.box_height}")
        # ax.legend()

        # ───── info box with lattice, d, s, t, η ────────────────
        info_lines = ["Lattice parameters:"]
        for name, val in p["lattice_params"].items():
            unc = p["lattice_uncertainties"].get(name, 0.0)
            info_lines.append(f"   {name} = {val:.5f} ± {unc:.5f}")

        # ------------------------------------------------------------------
        # 1)  PATTERN‑FIT (global) PARAMETERS
        # ------------------------------------------------------------------

        bulk = dict(
            d   = p.get("fitted_d", {}),
            du  = p.get("d_uncertainties", {}),
            s   = p.get("fitted_s", {}),
            su  = p.get("s_uncertainties", {}),
            t   = p.get("fitted_t", {}),
            tu  = p.get("t_uncertainties", {}),
            e   = p.get("fitted_eta", {}),
            eu  = p.get("eta_uncertainties", {}),
        )
        
        if bulk["s"]:
            info_lines.append("")
            info_lines.append("Pattern fit: s / t / η  (± σ)")


        def _nan_if_none(v):
            return float("nan") if v is None else v

        for hkl in sorted(bulk["s"].keys()):
            d,du = bulk["d"].get(hkl, float("nan")),  bulk["du"].get(hkl, 0.0)
            s,su = bulk["s"][hkl],                    bulk["su"].get(hkl, 0.0)
            t,tu = bulk["t"][hkl],                    bulk["tu"].get(hkl, 0.0)
            e,eu = bulk["e"][hkl],                    bulk["eu"].get(hkl, 0.0)

            d,du,s,su,t,tu,e,eu = map(_nan_if_none,
                                      (d,du,s,su,t,tu,e,eu))

            info_lines.append(
                f"   hkl{hkl}: "
                # f"d={d:.6f}±{du:.6f}, "
                f"s={s:.6f}±{su:.6f}, "
                f"t={t:.6f}±{tu:.6f}, "
                f"η={e:.6f}±{eu:.6f}"
            )

        # ------------------------------------------------------------------
        # 2)  REGION‑3 EDGE‑FIT PARAMETERS  (if available)
        # ------------------------------------------------------------------
        edge_fits = p.get("edge_fits", [])
        if edge_fits:
            info_lines.append("")
            info_lines.append("Edge fit: d / s / t / η  (± σ)")

            for ef in edge_fits:
                hkl = ef["hkl"]
                d   = _nan_if_none(ef.get("d_fit"))
                du  = _nan_if_none(ef.get("d_unc"))
                s   = _nan_if_none(ef.get("s_fit"))
                su  = _nan_if_none(ef.get("s_unc"))
                t   = _nan_if_none(ef.get("t_fit"))
                tu  = _nan_if_none(ef.get("t_unc"))
                e   = _nan_if_none(ef.get("eta_fit"))
                eu  = _nan_if_none(ef.get("eta_unc"))

                info_lines.append(
                    f"   hkl{hkl}: "
                    f"d={d:.6f}±{du:.6f}, "
                    f"s={s:.6f}±{su:.6f}, "
                    f"t={t:.6f}±{tu:.6f}, "
                    f"η={e:.6f}±{eu:.6f}"
                )

        # ------------------------------------------------------------------
        #  Render the text box
        # ------------------------------------------------------------------
        ax.text(0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=18,
                bbox=dict(fc="white", alpha=0.7, ec="none"))

        self.canvas.draw()

    # ------------------------------------------------------------------
    def save_plot(self):
        """
        Save a CSV containing
            wavelength | intensity_exp | intensity_fullfit | intensity_edge_...
        All columns have identical length; NaN is used where a curve is
        undefined on the experimental wavelength grid.
        """
        # ---- prompt for output filename ----------------------------
        opts = QFileDialog.Options()
        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save data as CSV",
            "",
            "CSV Files (*.csv)",
            options=opts
        )
        if not csv_path:                # user cancelled
            return
    
        import numpy as np, pandas as pd
        p = self.parameters
    
        # ---------- base grid (may contain duplicates) --------------
        x_exp = np.asarray(p["x_exp"]).ravel()
        y_exp = np.asarray(p["y_exp"]).ravel()
        n_pts = x_exp.size
    
        # start with experimental data
        data_dict = {
            "wavelength":      x_exp,
            "intensity_exp":   y_exp,
            "intensity_fullfit": np.full(n_pts, np.nan, dtype=float),
        }
    
        # ---------- put full‑pattern fit on the grid ----------------
        x_fit = np.asarray(p["model_x"]).ravel()
        y_fit = np.asarray(p["model_y"]).ravel()
    
        # for every point in x_fit, place y_fit value where x_exp == x_fit
        for xi, yi in zip(x_fit, y_fit):
            mask = x_exp == xi
            data_dict["intensity_fullfit"][mask] = yi
    
        # ---------- each Region‑3 edge curve ------------------------
        for ef in p.get("edge_fits", []):
            # column label
            hkl_lbl = (
                "_".join(map(str, ef["hkl"]))
                if isinstance(ef["hkl"], tuple) else str(ef["hkl"])
            )
            col = np.full(n_pts, np.nan, dtype=float)
    
            x_edge = np.asarray(ef["x"]).ravel()
            y_edge = np.asarray(ef["fit"]).ravel()
    
            for xi, yi in zip(x_edge, y_edge):
                col[x_exp == xi] = yi
    
            data_dict[f"intensity_edge_{hkl_lbl}"] = col
    
        # ---------- create DataFrame & write CSV --------------------
        df = pd.DataFrame(data_dict)
        try:
            df.to_csv(csv_path, index=False)
            QMessageBox.information(
                self, "CSV saved",
                f"Data successfully written to:\n{csv_path}"
            )
        except Exception as err:
            QMessageBox.warning(
                self, "Save failed",
                f"Could not save CSV file:\n{err}"
            )



class AdjustmentsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)                      # keep the parent

        self.setWindowTitle("Adjust Image")
        self.parent = parent

        # ─────────────── Build the GUI ───────────────
        layout = QVBoxLayout()

        # ➊ Contrast
        contrast_layout = QHBoxLayout()
        contrast_label   = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 200)
        self.contrast_slider.setValue(int(parent.contrast_slider_value))
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setRange(0.01, 2.00)
        self.contrast_spinbox.setSingleStep(0.01)
        self.contrast_spinbox.setValue(parent.contrast_slider_value / 100.0)
        self.contrast_spinbox.valueChanged.connect(self.update_contrast_spinbox)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_spinbox)
        layout.addLayout(contrast_layout)

        # ➋ Brightness
        brightness_layout = QHBoxLayout()
        brightness_label   = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(int(parent.min_slider_value), int(parent.max_slider_value))
        self.brightness_slider.setValue(parent.brightness_slider_value)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.brightness_spinbox = QDoubleSpinBox()
        self.brightness_spinbox.setRange(parent.min_slider_value, parent.max_slider_value)
        self.brightness_spinbox.setSingleStep(1.0)
        self.brightness_spinbox.setValue(parent.brightness_slider_value)
        self.brightness_spinbox.valueChanged.connect(self.update_brightness_spinbox)
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_spinbox)
        layout.addLayout(brightness_layout)

        # ➌ Minimum intensity
        min_layout = QHBoxLayout()
        min_label   = QLabel("Minimum:")
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setDecimals(4)
        self.min_spinbox.setRange(-np.inf, np.inf)            # set dynamically
        self.min_spinbox.setValue(parent.min_slider_value)
        self.min_spinbox.valueChanged.connect(self.update_min)
        min_layout.addWidget(min_label)
        min_layout.addWidget(self.min_spinbox)
        layout.addLayout(min_layout)

        # ➍ Maximum intensity
        max_layout = QHBoxLayout()
        max_label   = QLabel("Maximum:")
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setDecimals(4)
        self.max_spinbox.setRange(-np.inf, np.inf)            # set dynamically
        self.max_spinbox.setValue(parent.max_slider_value)
        self.max_spinbox.valueChanged.connect(self.update_max)
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.max_spinbox)
        layout.addLayout(max_layout)

        # Dynamic ranges based on the current image
        self.set_dynamic_ranges()

        self.setLayout(layout)

        # ─────────────── Size & position ───────────────
        #
        # 1. Shrink-wrap to the minimal size that fits the widgets.
        self.adjustSize()
        # 2. Add a small margin so the controls aren’t flush with the edges.
        margin_w, margin_h = 40, 20
        self.setFixedSize(self.width() + margin_w,
                          self.height() + margin_h)
        # 3. Centre the dialog on top of the parent window.
        if parent is not None:
            parent_rect = parent.geometry()
            self.move(parent_rect.center() - self.rect().center())


    def set_dynamic_ranges(self):
        """
        Adjust the ranges of min and max spinboxes based on image data.
        """
        if self.parent.auto_vmin is not None and self.parent.auto_vmax is not None:
            # Set the range wider than the current image data for flexibility
            range_min = self.parent.auto_vmin - (self.parent.auto_vmin * 0.5)
            range_max = self.parent.auto_vmax + (self.parent.auto_vmax * 0.5)
            self.min_spinbox.setRange(range_min, self.parent.auto_vmax)
            self.max_spinbox.setRange(self.parent.auto_vmin, range_max)
            # Initialize spinboxes to auto-adjusted values
            self.min_spinbox.setValue(self.parent.current_vmin)
            self.max_spinbox.setValue(self.parent.current_vmax)

    def update_contrast(self, value):
        """
        Update contrast based on slider and spinbox synchronization.
        """
        contrast_value = value / 100.0
        self.contrast_spinbox.blockSignals(True)
        self.contrast_spinbox.setValue(contrast_value)
        self.contrast_spinbox.blockSignals(False)
        self.parent.contrast_slider_value = value
        self.parent.display_image()

    def update_contrast_spinbox(self, value):
        """
        Update contrast based on spinbox and slider synchronization.
        """
        slider_value = int(value * 100)
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(slider_value)
        self.contrast_slider.blockSignals(False)
        self.parent.contrast_slider_value = slider_value
        self.parent.display_image()

    def update_brightness(self, value):
        """
        Update brightness based on slider and spinbox synchronization.
        """
        brightness_value = value
        self.brightness_spinbox.blockSignals(True)
        self.brightness_spinbox.setValue(brightness_value)
        self.brightness_spinbox.blockSignals(False)
        self.parent.brightness_slider_value = value
        self.parent.display_image()

    def update_brightness_spinbox(self, value):
        """
        Update brightness based on spinbox and slider synchronization.
        """
        slider_value = int(value)
        self.brightness_slider.blockSignals(True)
        self.brightness_slider.setValue(slider_value)
        self.brightness_slider.blockSignals(False)
        self.parent.brightness_slider_value = slider_value
        self.parent.display_image()

    def update_min(self, value):
        """
        Update minimum intensity and ensure it doesn't exceed the current maximum.
        """
        if value >= self.parent.current_vmax:
            QMessageBox.warning(self, "Invalid Minimum", "Minimum value cannot exceed or equal the maximum value.")
            self.min_spinbox.setValue(self.parent.current_vmin)
            return
        self.parent.min_slider_value = value
        self.parent.display_image()

    def update_max(self, value):
        """
        Update maximum intensity and ensure it doesn't fall below the current minimum.
        """
        if value <= self.parent.current_vmin:
            QMessageBox.warning(self, "Invalid Maximum", "Maximum value cannot be less than or equal to the minimum value.")
            self.max_spinbox.setValue(self.parent.current_vmax)
            return
        self.parent.max_slider_value = value
        self.parent.display_image()  
        

class ParameterPlotDialog(QDialog):
    def __init__(self, X_unique, Y_unique, Z, parameter_name, metadata, csv_filename, work_directory=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{parameter_name} Plot")
        self.setGeometry(200, 200, int(screen_width * 0.5), int(screen_height * 0.8))
        self.work_directory = work_directory
        self.parameter_name = parameter_name
        self.metadata = metadata  # Store metadata
        self.csv_filename = csv_filename  # Store CSV filename

        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.canvas = MplCanvas(self, width=9, height=7, dpi=100)

        # Layouts
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        # Color bar min and max inputs
        self.color_min_input = QLineEdit()
        self.color_max_input = QLineEdit()
        self.d0_input = QLineEdit()
        self.color_min_input.setPlaceholderText("Color Bar Min")
        self.color_max_input.setPlaceholderText("Color Bar Max")
        self.d0_input.setPlaceholderText("d0")
        self.color_min_input.editingFinished.connect(self.update_plot)
        self.color_max_input.editingFinished.connect(self.update_plot)

        control_layout.addWidget(QLabel("Color Bar Min:"))
        control_layout.addWidget(self.color_min_input)
        control_layout.addWidget(QLabel("Color Bar Max:"))
        control_layout.addWidget(self.color_max_input)
        control_layout.addWidget(QLabel("d0:"))
        control_layout.addWidget(self.d0_input)

        # Add a button to calculate strain
        self.calculate_strain_button = QPushButton("Calculate Strain")
        self.calculate_strain_button.clicked.connect(self.calculate_strain)
        control_layout.addWidget(self.calculate_strain_button)

        # Add input fields for x_min, x_max, y_min, y_max
        self.x_min_input = QLineEdit()
        self.x_max_input = QLineEdit()
        self.y_min_input = QLineEdit()
        self.y_max_input = QLineEdit()
        self.x_min_input.setPlaceholderText("x min")
        self.x_max_input.setPlaceholderText("x max")
        self.y_min_input.setPlaceholderText("y min")
        self.y_max_input.setPlaceholderText("y max")

        control_layout.addWidget(QLabel("x min:"))
        control_layout.addWidget(self.x_min_input)
        control_layout.addWidget(QLabel("x max:"))
        control_layout.addWidget(self.x_max_input)
        control_layout.addWidget(QLabel("y min:"))
        control_layout.addWidget(self.y_min_input)
        control_layout.addWidget(QLabel("y max:"))
        control_layout.addWidget(self.y_max_input)

        # Add a button to calculate the mean value
        self.calculate_mean_button = QPushButton("Calculate Mean")
        self.calculate_mean_button.clicked.connect(self.calculate_mean)
        control_layout.addWidget(self.calculate_mean_button)

        self.unit_switch_checkbox = QCheckBox("Display in mm")
        self.unit_switch_checkbox.stateChanged.connect(self.toggle_units)
        control_layout.addWidget(self.unit_switch_checkbox)

        # Add a toggle button to enable/disable point selection
        self.toggle_select_button = QPushButton("Select Points")
        self.toggle_select_button.setCheckable(True)
        self.toggle_select_button.setStyleSheet("background-color: none")
        self.toggle_select_button.clicked.connect(self.toggle_select_mode)
        control_layout.addWidget(self.toggle_select_button)
        
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.apply_filter)
        control_layout.addWidget(self.filter_button)

        # Add a button to save as FITS
        self.save_fits_button = QPushButton("Save as FITS")
        self.save_fits_button.clicked.connect(self.save_as_fits)
        control_layout.addWidget(self.save_fits_button)

        self.unit_in_mm = False  # Default to pixels

        # Create a plot area widget containing the canvas and labels
        plot_widget = QWidget()
        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout = QVBoxLayout(plot_widget)
        # Add the canvas to the plot_layout
        plot_layout.addWidget(self.canvas)
        # Add labels to display coordinates and mean value
        self.coord_label = QLabel("x: -, y: -, value: -")
        self.coord_label.setFixedHeight(40)
        self.mean_label = QLabel("Mean Value: -")
        self.mean_label.setFixedHeight(40)
        # Create a horizontal layout for the labels
        label_layout = QHBoxLayout()
        label_layout.addWidget(self.coord_label)
        label_layout.addWidget(self.mean_label)
        # Add the label layout to the plot_layout
        plot_layout.addLayout(label_layout)

        # Create a splitter to hold metadata and plot area
        display_splitter = QSplitter(Qt.Horizontal)

        # Metadata display area
        metadata_widget = QWidget()
        metadata_layout = QVBoxLayout()
        metadata_widget.setLayout(metadata_layout)
        
                # **Add filename label**
        # filename_label = QLabel(f"Loaded File: {os.path.basename(self.csv_filename)}")
        # filename_label.setStyleSheet("font-weight: bold;")
        
        # metadata_label = QLabel("Metadata:")
        self.metadata_display = QTextEdit()
        self.metadata_display.setReadOnly(True)
        self.metadata_display.setPlainText(self.format_metadata(self.metadata, self.csv_filename))
        self.metadata_display.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # metadata_layout.addWidget(metadata_label)
        # metadata_layout.addWidget(filename_label)
        metadata_layout.addWidget(self.metadata_display)

        # Add widgets to splitter
        display_splitter.addWidget(metadata_widget)
        display_splitter.addWidget(plot_widget)
        display_splitter.setSizes([200, 600])  # Adjust sizes to approximate 1/4 and 3/4 widths

        # Layout adjustments
        main_layout.addLayout(control_layout)

        # Add Matplotlib's Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        # Add the splitter to main_layout
        main_layout.addWidget(display_splitter)
        self.setLayout(main_layout)

        # Store data
        self.X_unique = X_unique
        self.Y_unique = Y_unique
        self.Z = Z

        # Initialize colorbar and mesh reference
        self.colorbar = None
        self.mesh = None  # Initialize mesh to None

        # Variables to store clicked points
        self.click_coords = []

        # Initialize scatter for markers
        self.marker_scatter = self.canvas.axes.scatter([], [], c='red', marker='o')

        # Selection mode flag
        self.select_mode_enabled = False

        # Initialize interaction attributes
        self._press_event = None
        self._is_panning = False
        self._last_mouse_pos = None  # You can set it to (0, 0) if you prefer

        # Initial plot
        self.plot_parameter()

        # Connect events
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion_and_coordinates)

        # Keep a reference to line profile dialogs to prevent garbage collection
        self.line_profile_dialogs = []
        
    def apply_filter(self):
            """
            Prompts the user to load a FITS file as a mask and applies it to the parameter map Z.
            Then opens a new ParameterPlotDialog showing the filtered result.
            """
            # Open a file dialog to select the FITS file
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select FITS Mask File", "",
                "FITS Files (*.fits);;All Files (*)", options=options
            )
            if not file_path:
                return  # User canceled
    
            try:
                # Load the mask from the FITS file
                with fits.open(file_path) as hdulist:
                    mask_data = hdulist[0].data  # Assuming the mask is in the primary HDU
    
                if mask_data is None:
                    raise ValueError("No data found in the selected FITS file.")
    
                # Check shape compatibility
                if mask_data.shape != self.Z.shape:
                    raise ValueError(
                        f"Mask shape {mask_data.shape} does not match parameter map shape {self.Z.shape}."
                    )
    
                # Perform element-wise multiplication
                filtered_result = self.Z * mask_data
                filtered_result[filtered_result == 0] = np.nan
    
                # Open a new ParameterPlotDialog to display the filtered data
                filtered_dialog = ParameterPlotDialog(
                    self.X_unique,
                    self.Y_unique,
                    filtered_result,
                    f"Filtered {self.parameter_name}",
                    metadata=self.metadata,
                    csv_filename=self.csv_filename,
                    work_directory=self.work_directory,
                    parent=self.parent()
                )
                filtered_dialog.show()
    
                # Keep a reference so the dialog is not garbage-collected
                if not hasattr(self, 'child_dialogs'):
                    self.child_dialogs = []
                self.child_dialogs.append(filtered_dialog)
    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply filter:\n{e}")


    def format_metadata(self, metadata, filename):
        """
        Formats the metadata dictionary into a string for display,
        including the filename at the top.
        """
        metadata_text = f"Loaded File: {os.path.basename(filename)}\n\n"
        metadata_entries = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
        metadata_text += metadata_entries
        return metadata_text

        
    def calculate_edges(self, centers):
        # Calculate edges from centers
        # The edges are halfway between adjacent centers
        edges = (centers[:-1] + centers[1:]) / 2
        # Extrapolate the first and last edges
        first_edge = centers[0] - (edges[0] - centers[0])
        last_edge = centers[-1] + (centers[-1] - edges[-1])
        edges = np.concatenate(([first_edge], edges, [last_edge]))
        return edges

        
    def toggle_units(self):
        self.unit_in_mm = self.unit_switch_checkbox.isChecked()
        # Update placeholders for input fields
        if self.unit_in_mm:
            x_min_placeholder = "x min (mm)"
            x_max_placeholder = "x max (mm)"
            y_min_placeholder = "y min (mm)"
            y_max_placeholder = "y max (mm)"
        else:
            x_min_placeholder = "x min (pixels)"
            x_max_placeholder = "x max (pixels)"
            y_min_placeholder = "y min (pixels)"
            y_max_placeholder = "y max (pixels)"
        self.x_min_input.setPlaceholderText(x_min_placeholder)
        self.x_max_input.setPlaceholderText(x_max_placeholder)
        self.y_min_input.setPlaceholderText(y_min_placeholder)
        self.y_max_input.setPlaceholderText(y_max_placeholder)
        # Re-plot the parameter map with the new units
        self.plot_parameter()

    
    def save_as_fits(self):
        # ------------------------------------------------------------------
        # 1. Ask user for a file name
        # ------------------------------------------------------------------
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Map as FITS", "",
            "FITS Files (*.fits);;All Files (*)", options=options
        )
        if not file_path:           # user hit “Cancel”
            return
    
        try:
            # ------------------------------------------------------------------
            # 2. Prepare data (resize → float32 → flip vertically)
            # ------------------------------------------------------------------
            resized_Z = self.resize_parameter_map(self.Z, (512, 512)).astype(np.float32)
    
            # Flip rows so that (row 0, col 0) becomes the *bottom-left* pixel
            fits_Z = np.flipud(resized_Z)          # or resized_Z[::-1, :]
    
            # ------------------------------------------------------------------
            # 3. Build FITS header  (optional: note we flipped Y)
            # ------------------------------------------------------------------
            header = fits.Header()
            header["PARAM"]  = self.parameter_name
            header["COMMENT"] = "Parameter map saved from the application"
            header["HISTORY"] = "Array flipped in Y so FITS shows same orientation as GUI"
    
            # ------------------------------------------------------------------
            # 4. Write file
            # ------------------------------------------------------------------
            fits.PrimaryHDU(data=fits_Z, header=header).writeto(file_path, overwrite=True)
    
            QMessageBox.information(
                self, "Success",
                f"Parameter map saved successfully to {file_path}."
            )
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FITS file:\n{e}")


    def resize_parameter_map(self, Z, new_shape):
        # Calculate the zoom factors
        zoom_factors = (new_shape[0] / Z.shape[0], new_shape[1] / Z.shape[1])
        # Use scipy's zoom function to resize the array
        resized_Z = zoom(Z, zoom_factors, order=1)  # order=1 for bilinear interpolation
        return resized_Z
 
       
    def plot_parameter(self):
        # Remove colorbar first
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None  # Reset to None
    
        self.canvas.axes.clear()
    
        if self.unit_in_mm:
            x_factor = 0.055  # Conversion factor from pixels to mm
            y_factor = 0.055
            x_label = 'X Position (mm)'
            y_label = 'Y Position (mm)'
        else:
            x_factor = 1.0
            y_factor = 1.0
            x_label = 'X Position (pixels)'
            y_label = 'Y Position (pixels)'
    
        # Convert centers to the selected unit
        x_centers = self.X_unique * x_factor
        y_centers = self.Y_unique * y_factor
    
        # Calculate edges
        x_edges = self.calculate_edges(x_centers)
        y_edges = self.calculate_edges(y_centers)
    
        # Use pcolormesh with the edges
        self.mesh = self.canvas.axes.pcolormesh(
            x_edges, y_edges, self.Z, shading='flat'
        )
    
        # Set equal aspect ratio
        self.canvas.axes.set_aspect('equal', adjustable='box')
    
        # Set axis limits based on data
        self.canvas.axes.set_xlim(x_edges[0], x_edges[-1])
        self.canvas.axes.set_ylim(y_edges[0], y_edges[-1])
    
        # Invert the y-axis to have origin at the upper-left
        self.canvas.axes.invert_yaxis()
    
        # Create a new colorbar
        self.colorbar = self.canvas.fig.colorbar(
            self.mesh, ax=self.canvas.axes
        )
    
        # Set color limits
        cmin_text = self.color_min_input.text()
        cmax_text = self.color_max_input.text()
        if not cmin_text:
            cmin = np.nanmin(self.Z)
            self.color_min_input.setText(f"{cmin:.4f}")
        else:
            cmin = float(cmin_text)
        if not cmax_text:
            cmax = np.nanmax(self.Z)
            self.color_max_input.setText(f"{cmax:.4f}")
        else:
            cmax = float(cmax_text)
        self.mesh.set_clim(vmin=cmin, vmax=cmax)
    
        self.canvas.axes.set_title(self.parameter_name)
        self.canvas.axes.set_xlabel(x_label)
        self.canvas.axes.set_ylabel(y_label)
        self.canvas.draw()


    def update_plot(self):
        if self.mesh is None:
            return  # Exit if mesh is not yet created
        try:
            cmin = float(self.color_min_input.text())
            cmax = float(self.color_max_input.text())

            if cmin >= cmax:
                raise ValueError("Minimum must be less than Maximum.")

            # Update color limits
            self.mesh.set_clim(vmin=cmin, vmax=cmax)

            # Remove and re-add the colorbar to update it
            if self.colorbar is not None:
                self.colorbar.remove()
            self.colorbar = self.canvas.fig.colorbar(
                self.mesh, ax=self.canvas.axes
            )

            self.canvas.draw()
        except ValueError:
            # QMessageBox.warning(self, "Invalid Input", f"Error: {e}")
            # Optionally reset inputs to default or previous valid values
            pass 
        
    def toggle_select_mode(self):
        self.select_mode_enabled = self.toggle_select_button.isChecked()
        if self.select_mode_enabled:
            self.toggle_select_button.setText("Selection Mode: ON")
            self.toggle_select_button.setStyleSheet("background-color: lightgreen")
            # Reset temporary selections for a new pair but keep previous selections intact.
            self.click_coords = []
            self.first_point_plot = None
            # Initialize counters and lists if they don't exist yet.
            if not hasattr(self, 'selection_counter'):
                self.selection_counter = 0
            if not hasattr(self, 'all_start_annotations'):
                self.all_start_annotations = []
            if not hasattr(self, 'all_end_annotations'):
                self.all_end_annotations = []
            if not hasattr(self, 'all_selection_lines'):
                self.all_selection_lines = []
        else:
            self.toggle_select_button.setText("Select Points")
            self.toggle_select_button.setStyleSheet("background-color: none")
            # Remove all permanent annotations and lines
            if hasattr(self, 'all_start_annotations'):
                for annot in self.all_start_annotations:
                    annot.remove()
                self.all_start_annotations = []
            if hasattr(self, 'all_end_annotations'):
                for annot in self.all_end_annotations:
                    annot.remove()
                self.all_end_annotations = []
            if hasattr(self, 'all_selection_lines'):
                for line, path_annot in self.all_selection_lines:
                    line.remove()
                    path_annot.remove()
                self.all_selection_lines = []
            # Remove any remaining markers with labels 'selection_marker' or 'selection_line'
            for line in self.canvas.axes.get_lines():
                if line.get_label() in ['selection_marker', 'selection_line']:
                    line.remove()
            # Reset temporary state and counter
            self.click_coords = []
            self.first_point_plot = None
            self.selection_counter = 0
            self.canvas.draw()
    
    def on_release(self, event):
        if self.select_mode_enabled and event.button == 1 and event.inaxes == self.canvas.axes:
            if not self._is_panning:
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return
                # Convert coordinates to pixels if needed for internal processing.
                if self.unit_in_mm:
                    x_pixel = x / 0.055
                    y_pixel = y / 0.055
                else:
                    x_pixel = x
                    y_pixel = y
    
                self.click_coords.append((x_pixel, y_pixel))
    
                if len(self.click_coords) == 1:
                    # First point selected: store and annotate as "start"
                    self.first_point_plot = (x, y)
                    current_pair_number = self.selection_counter + 1
                    self.canvas.axes.plot(x, y, 'ro', label='selection_marker')
                    start_annot = self.canvas.axes.annotate(
                        f'start {current_pair_number}', (x, y),
                        textcoords="offset points", xytext=(8, 8),
                        color='red', fontsize=14
                    )
                    self.all_start_annotations.append(start_annot)
                elif len(self.click_coords) == 2:
                    # Second point selected: annotate as "end" and draw connecting line.
                    current_pair_number = self.selection_counter + 1
                    self.canvas.axes.plot(x, y, 'ro', label='selection_marker')
                    end_annot = self.canvas.axes.annotate(
                        f'end {current_pair_number}', (x, y),
                        textcoords="offset points", xytext=(8, 8),
                        color='red', fontsize=14
                    )
                    self.all_end_annotations.append(end_annot)
                    # Draw a red line connecting the two points.
                    selection_line, = self.canvas.axes.plot(
                        [self.first_point_plot[0], x],
                        [self.first_point_plot[1], y],
                        'r-', lw=2, label='selection_line'
                    )
                    # Annotate the line (path) at its midpoint.
                    mid_x = (self.first_point_plot[0] + x) / 2
                    mid_y = (self.first_point_plot[1] + y) / 2
                    path_annot = self.canvas.axes.annotate(
                        f'path {current_pair_number}', (mid_x, mid_y),
                        textcoords="offset points", xytext=(8, 8),
                        color='red', fontsize=14
                    )
                    self.all_selection_lines.append((selection_line, path_annot))
                    self.canvas.draw()
                    # Optionally, process this pair (e.g., extract line profile)
                    self.extract_line_profile()
                    # Reset temporary state for the next pair.
                    self.click_coords = []
                    self.first_point_plot = None
                    self.selection_counter += 1
        # Reset panning variables
        self._is_panning = False
        self._press_event = None


    def on_press(self, event):
        if self.select_mode_enabled and event.button == 1 and event.inaxes == self.canvas.axes:
            self._is_panning = False
            self._press_event = event
            if self.unit_in_mm:
                self._last_mouse_pos = (event.xdata / 0.055, event.ydata / 0.055)
            else:
                self._last_mouse_pos = (event.xdata, event.ydata)
        else:
            self._press_event = None


    def on_motion_and_coordinates(self, event):
        if event.inaxes != self.canvas.axes:
            return
    
        if self._press_event is None:
            # No button pressed; update coordinates
            self.update_coordinates(event)
            return
    
        if self.select_mode_enabled and (event.button == 1 or self._is_panning):
            dx = event.xdata - self._press_event.xdata
            dy = event.ydata - self._press_event.ydata
            movement = np.hypot(dx, dy)
            if movement > 0.01:
                self._is_panning = True
    
            if self._is_panning:
                dx = event.xdata - self._last_mouse_pos[0]
                dy = event.ydata - self._last_mouse_pos[1]
                cur_xlim = self.canvas.axes.get_xlim()
                cur_ylim = self.canvas.axes.get_ylim()
                self.canvas.axes.set_xlim(
                    cur_xlim[0] - dx, cur_xlim[1] - dx
                )
                self.canvas.axes.set_ylim(
                    cur_ylim[0] - dy, cur_ylim[1] - dy
                )
                self._last_mouse_pos = (event.xdata, event.ydata)
                self.canvas.draw()
        else:
            self.update_coordinates(event)

    def calculate_strain(self):
        # Get d0 value from input
        try:
            d0 = float(self.d0_input.text())
            if d0 <= 0:
                raise ValueError("d0 must be greater than 0.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for d0.")
            return

        # Calculate strain as x_hkl / d0 at each pixel
        x_hkl_data = self.Z  # Assuming self.Z contains x_hkl data
        strain = (x_hkl_data / d0 - 1) * 1e6  # Multiply by 1e6 to get strain in microstrains

        # Open a new ParameterPlotDialog to display the strain data
        strain_dialog = ParameterPlotDialog(
            self.X_unique,
            self.Y_unique,
            strain,
            'Strain',
            metadata=self.metadata,            # Pass the metadata
            csv_filename=self.csv_filename,    # Pass the CSV filename
            work_directory=self.work_directory,
            parent=self.parent()
        )
        strain_dialog.show()

        # Keep a reference to prevent garbage collection
        if not hasattr(self, 'child_dialogs'):
            self.child_dialogs = []
        self.child_dialogs.append(strain_dialog)

    def calculate_mean(self):
        try:
            # Get x_min, x_max, y_min, y_max from input
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())
    
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("Minimum values must be less than maximum values.")
    
            # Convert to pixels if unit is mm
            if self.unit_in_mm:
                x_min_pixel = x_min / 0.055
                x_max_pixel = x_max / 0.055
                y_min_pixel = y_min / 0.055
                y_max_pixel = y_max / 0.055
            else:
                x_min_pixel = x_min
                x_max_pixel = x_max
                y_min_pixel = y_min
                y_max_pixel = y_max
    
            # Ensure coordinates are within bounds
            x_min_pixel = max(min(x_min_pixel, self.X_unique[-1]), self.X_unique[0])
            x_max_pixel = max(min(x_max_pixel, self.X_unique[-1]), self.X_unique[0])
            y_min_pixel = max(min(y_min_pixel, self.Y_unique[-1]), self.Y_unique[0])
            y_max_pixel = max(min(y_max_pixel, self.Y_unique[-1]), self.Y_unique[0])
    
            # Map x and y to array indices using searchsorted
            ixmin = np.searchsorted(self.X_unique, x_min_pixel, side='left')
            ixmax = np.searchsorted(self.X_unique, x_max_pixel, side='right')
            iymin = np.searchsorted(self.Y_unique, y_min_pixel, side='left')
            iymax = np.searchsorted(self.Y_unique, y_max_pixel, side='right')
    
            # Ensure indices are within bounds
            ixmin = max(ixmin, 0)
            ixmax = min(ixmax, self.Z.shape[1])
            iymin = max(iymin, 0)
            iymax = min(iymax, self.Z.shape[0])
    
            # Extract the subarray of Z values within the selected area
            Z_selected = self.Z[iymin:iymax, ixmin:ixmax]
    
            if Z_selected.size == 0:
                QMessageBox.warning(self, "No Data", "The selected area contains no data.")
                return
    
            # Calculate the mean value
            mean_value = np.nanmean(Z_selected)
    
            # Update the mean label
            unit = 'Å' if self.parameter_name != 'Strain' else 'µε'  # µε for microstrain
            self.mean_label.setText(f"Mean Value: {mean_value:.6f} {unit}")
    
            # Optionally, draw a rectangle on the plot to show the selected area
            if hasattr(self, 'selection_rectangle'):
                self.selection_rectangle.remove()
    

            rect_x = x_min
            rect_y = y_min
            rect_width = x_max - x_min
            rect_height = y_max - y_min

    
            self.selection_rectangle = Rectangle(
                (rect_x, rect_y),
                rect_width,
                rect_height,
                edgecolor='red',
                facecolor='none',
                linewidth=2
            )
            self.canvas.axes.add_patch(self.selection_rectangle)
            self.canvas.draw()
    
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error in input values:\n{e}")


    def zoom(self, event):
        # Zooming is handled by the Navigation Toolbar
        pass  # No need to implement custom zoom

    def update_coordinates(self, event):
        x, y = event.xdata, event.ydata
        # Find the closest grid point in the data
        if x is not None and y is not None:
            if self.unit_in_mm:
                # Convert x and y back to pixels for indexing
                x_pixel = x / 0.055
                y_pixel = y / 0.055
            else:
                x_pixel = x
                y_pixel = y
    
            ix = np.abs(self.X_unique - x_pixel).argmin()
            iy = np.abs(self.Y_unique - y_pixel).argmin()
            z_value = (
                self.Z[iy, ix]
                if 0 <= iy < self.Z.shape[0] and 0 <= ix < self.Z.shape[1]
                else None
            )
    
            if self.parameter_name != 'Strain':
                value_str = f"{z_value:.6f} Å" if z_value is not None else "Out of range"
            else:
                value_str = f"{z_value:.0f} µε" if z_value is not None else "Out of range"
    
            if self.unit_in_mm:
                self.coord_label.setText(
                    f"x: {x:.2f} mm, y: {y:.2f} mm, value: {value_str}"
                )
            else:
                self.coord_label.setText(
                    f"x: {x:.0f}, y: {y:.0f}, value: {value_str}"
                )
        else:
            self.coord_label.setText("x: -, y: -, value: -")


    def extract_line_profile(self):
        # Get the two points
        (x0, y0), (x1, y1) = self.click_coords
        # Number of points along the line
        num_points = 500
        # Generate line coordinates in pixels
        x_coords_pixel = np.linspace(x0, x1, num_points)
        y_coords_pixel = np.linspace(y0, y1, num_points)
    
        # Convert coordinates to mm for plotting if necessary
        if self.unit_in_mm:
            x_coords_plot = x_coords_pixel * 0.055
            y_coords_plot = y_coords_pixel * 0.055
        else:
            x_coords_plot = x_coords_pixel
            y_coords_plot = y_coords_pixel
    
        # Extract z values along the line
        z_values = self.interpolate_z_values(x_coords_pixel, y_coords_pixel)
    
        # Plot the line profile in a new window
        self.plot_line_profile(x_coords_plot, y_coords_plot, z_values)


    def interpolate_z_values(self, x_coords_pixel, y_coords_pixel):
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (self.Y_unique, self.X_unique), self.Z,
            bounds_error=False, fill_value=np.nan
        )
        # Prepare coordinate pairs
        points = np.array([y_coords_pixel, x_coords_pixel]).T  # Note the order: (y, x)
        # Interpolate z values
        z_values = interpolator(points)
        return z_values


    def plot_line_profile(self, x_coords, y_coords, z_values):
        # Create a new dialog to display the line profile
        line_profile_dialog = LineProfileDialog(
            x_coords, y_coords, z_values, parent=self
        )
        # Show the dialog non-modally and keep a reference to prevent garbage collection
        self.line_profile_dialogs.append(line_profile_dialog)
        line_profile_dialog.show()

class LineProfileDialog(QDialog):
    def __init__(self, x_coords, y_coords, z_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line Profile")
        self.setGeometry(200, 200, int(window_width*0.7), int(window_height*0.7))

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Store the data for saving
        self.distances = np.sqrt((x_coords - x_coords[0])**2 + (y_coords - y_coords[0])**2)
        self.z_values = z_values


        # Matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Add stretch to push the buttons to the right

        # Save Data button
        self.save_data_button = QPushButton("Save Data")
        self.save_data_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_data_button)

        # Save Plot button (optional)
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_plot_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Plot the line profile
        self.canvas.axes.plot(self.distances, self.z_values)
        self.canvas.axes.set_xlabel("Distance")
        self.canvas.axes.set_ylabel("Value")
        self.canvas.axes.set_title("Line Profile")
        self.canvas.draw()

    def save_data(self):
        """
        Opens a file dialog to save the distance and z_values data to a text file.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                # Save the data to the selected file
                data = np.column_stack((self.distances, self.z_values))
                header = "Distance\tValue"
                np.savetxt(file_name, data, delimiter='\t', header=header, comments='')
                QMessageBox.information(self, "Save Data", f"Data saved successfully to:\n{file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Data", f"Failed to save data:\n{e}")

    def save_plot(self):
        """
        Opens a file dialog to save the current plot.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                self.canvas.figure.savefig(file_name)
                QMessageBox.information(self, "Save Plot", f"Plot saved successfully to:\n{file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Save Plot", f"Failed to save plot:\n{e}")

# Dialog class for Open Beam Intensity vs Wavelength Plot
class OpenBeamPlotDialog(QDialog):
    def __init__(self, wavelengths, intensities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Beam Intensity vs Wavelength")
        self.setGeometry(200, 200, int(window_width*0.5), int(window_height*0.5))
        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=7, height=5, dpi=100)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Plotting
        self.canvas.axes.plot(wavelengths, intensities, 'o-', color='green')
        self.canvas.axes.set_xlabel("Wavelength (Å)")
        self.canvas.axes.set_ylabel("Summed Intensity")
        self.canvas.axes.set_title("Open Beam Intensity vs Wavelength")
        self.canvas.draw()


class OpenBeamLoadWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)
    open_beam_loaded = pyqtSignal(str, dict)  # Dictionary: {suffix: open_beam_image (2D np.array)}

    def __init__(self, folder_path, area_x=(10, 500), area_y=(10, 500)):
        """
        Note:
          In the revised version the entire open beam images are loaded.
          The parameters area_x and area_y are no longer used to compute a single summed value,
          but you can still use them to enforce a minimum image size if desired.
        """
        super().__init__()
        self.folder_path = folder_path
        self.area_x = area_x  # Not used in the new approach, but kept for compatibility.
        self.area_y = area_y

    def run(self):
        try:
            # List all FITS files in the folder
            normalized_path = os.path.normpath(self.folder_path)
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) >= 2:
                short_path = os.path.join(path_parts[-2], path_parts[-1])
            else:
                short_path = normalized_path
            # short_path = self.get_short_path(self.folder_path, levels=2)

            fits_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.fits', '.fit'))]
            if not fits_files:
                self.message.emit(f"No FITS files found in folder: \\{short_path}")
                self.finished.emit()
                return

            # Sort files so that the suffixes (assumed to be in the filename) are aligned
            fits_files.sort()
            total_files = len(fits_files)
            open_beam_images = {}

            for idx, file in enumerate(fits_files):
                basename = os.path.splitext(file)[0]
                parts = basename.split('_')
                if len(parts) < 2:
                    self.message.emit(f"Filename '{file}' does not contain an underscore-separated suffix. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue

                suffix = parts[-1]
                # Validate suffix: must be a five-digit number
                if not suffix.isdigit() or len(suffix) != 5:
                    self.message.emit(f"Suffix '{suffix}' in filename '{file}' is not a valid five-digit number. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue

                file_path = os.path.join(self.folder_path, file)
                try:
                    image_data = fits.getdata(file_path)
                    if image_data is None:
                        self.message.emit(f"No image data found in file '{file}'. Skipping.")
                        progress = int(((idx + 1) / total_files) * 100)
                        self.progress_updated.emit(progress)
                        continue
                except Exception as e:
                    self.message.emit(f"Failed to read data from '{file}': {e}. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue

                # Convert image to float32 for processing
                image_data = image_data.astype(np.float32)
                h, w = image_data.shape

                # Check that the image is large enough for a 21x21 window
                # and that it contains a safe margin (5 pixels) along each edge.
                if h < 31 or w < 31:
                    self.message.emit(f"Image '{file}' dimensions ({h}x{w}) are too small for the required 21x21 window and 5-pixel edge margin. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue

                # Store the full open beam image in the dictionary (keyed by its suffix)
                open_beam_images[suffix] = image_data

                progress = int(((idx + 1) / total_files) * 100)
                self.progress_updated.emit(progress)

            if not open_beam_images:
                self.message.emit(f"No valid open beam images were loaded from \\{short_path}")
            else:
                self.message.emit(f"Successfully loaded {len(open_beam_images)} open beam images from \\{short_path}")
                self.open_beam_loaded.emit(self.folder_path, open_beam_images)

            gc.collect()

        except Exception as e:
            self.message.emit(f"Error loading open beam images from \\{short_path}: {e}")

        self.finished.emit()


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
        self.report_path = os.path.join(output_folder, f"{base_name}_outlier_report.csv")

    # ────────────────────────────────────────────────────────────────────────────
    # Main thread entry
    # ────────────────────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._prepare_output()

            total_images   = sum(len(run["images"]) for run in self.image_runs)
            processed      = 0
            total_cleaned  = 0

            for run_idx, data_run in enumerate(self.image_runs, start=1):
                if not self._is_running:
                    self.message.emit("Process stopped by user")
                    break

                self._copy_related_files(run_idx, data_run)

                for suffix, img_data in data_run["images"].items():
                    if not self._is_running:
                        break

                    try:
                        cleaned_pixels = self._clean_one_frame(suffix, img_data)
                        total_cleaned += cleaned_pixels
                        processed     += 1
                        self._update_progress(processed, total_images)

                    except Exception as exc:
                        self.message.emit(f"[ERROR] Frame {suffix}: {exc}")

            self.message.emit(
                f"[SUCCESS] Processed {processed} frame(s) • "
                f"Total cleaned pixels: {total_cleaned} • "
                f"Report: {os.path.basename(self.report_path)}"
            )
        except Exception as exc:
            self.message.emit(f"[FATAL] {exc}")
        finally:
            self.finished.emit()

    # ────────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ────────────────────────────────────────────────────────────────────────────
    def _prepare_output(self):
        """Ensure output folder exists and create a fresh report file."""
        os.makedirs(self.output_folder, exist_ok=True)
        try:
            with open(self.report_path, "w", encoding="utf-8") as fh:
                fh.write("frame_idx,pixel_x,pixel_y,outlier_value,replace_value\n")
        except OSError as exc:
            raise RuntimeError(f"Cannot create report file: {exc}") from exc

    def _clean_one_frame(self, suffix: str, img_data: np.ndarray) -> int:
        """Return number of pixels cleaned for this frame."""
        img = img_data.astype(np.float32, copy=True)
        invalid_mask = (img <= 0) | np.isnan(img) | np.isinf(img)
        bad_pixels   = np.argwhere(invalid_mask)

        if bad_pixels.size == 0:
            self.message.emit(f"Frame {suffix}: no outliers detected")
            return 0

        records = []
        cleaned = 0

        for y, x in bad_pixels:
            original = img[y, x]

            y0, y1 = max(0, y - 2), min(img.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(img.shape[1], x + 3)

            nbhood      = img[y0:y1, x0:x1]
            valid_mask  = (nbhood > 0) & np.isfinite(nbhood)
            valid_values = nbhood[valid_mask]

            if valid_values.size:          # We have something to average
                replacement = float(np.mean(valid_values))
                img[y, x]   = replacement
            else:
                replacement = np.nan       # Do not inject a hard zero
                # Leave img[y, x] unchanged to flag it downstream

            records.append(
                f"{suffix},{x},{y},{original:.4f},{replacement:.4f}"
            )
            cleaned += 1

        # Append to CSV
        try:
            with open(self.report_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(records) + "\n")
        except OSError as exc:
            self.message.emit(f"[WARNING] Could not append to report: {exc}")

        # Write FITS
        out_fits = os.path.join(self.output_folder, f"{self.base_name}_{suffix}.fits")
        try:
            fits.writeto(out_fits, img, overwrite=True)
        except Exception as exc:  # astropy throws its own subclass of OSError
            raise RuntimeError(f"Cannot write FITS {out_fits}: {exc}") from exc

        return cleaned

    def _update_progress(self, processed: int, total: int) -> None:
        progress = int((processed / total) * 100) if total else 0
        self.progress_updated.emit(progress)

    def _copy_related_files(self, run_idx: int, data_run: dict) -> None:
        """Copy *_Spectra.txt and *_ShutterCount.txt (if present)"""
        spectra_suffix      = "_Spectra.txt"
        shuttercount_suffix = "_ShutterCount.txt"

        def _copy_if_exists(suffix: str):
            for f in os.listdir(data_run["folder_path"]):
                if f.endswith(suffix):
                    src = os.path.join(data_run["folder_path"], f)
                    dst = os.path.join(self.output_folder, f"Run{run_idx}_{f}")
                    try:
                        shutil.copy2(src, dst)
                        self.message.emit(f"Copied {f} → {os.path.basename(dst)}")
                    except OSError as exc:
                        self.message.emit(f"[WARNING] Could not copy {f}: {exc}")
                    break          # only the first match

        _copy_if_exists(spectra_suffix)
        _copy_if_exists(shuttercount_suffix)

    # --------------------------------------------------------------------
    def stop(self):
        self._is_running = False


# import logging
from PyQt5.QtCore import QThread, pyqtSignal

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
            self._prepare_output()
            self._sum_images()
            if self._is_running:
                self._sum_and_save_shuttercounts()
            if self._is_running:
                self._copy_spectra_files()
        except Exception as exc:
            self.message.emit(f"[FATAL] Summation aborted: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    # ───────────────────────────────────────────────────────────────
    # Phase 0  –  Ensure output folder & log shortcut
    # ───────────────────────────────────────────────────────────────
    def _prepare_output(self):
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot create output folder '{self.output_folder}': {exc}") from exc

        parts = os.path.normpath(self.output_folder).split(os.sep)
        self._short_path = os.path.join(*parts[-2:]) if len(parts) >= 2 else self.output_folder

    # ───────────────────────────────────────────────────────────────
    # Phase 1  –  Image summation
    # ───────────────────────────────────────────────────────────────
    def _sum_images(self):
        if not self.summation_image_runs:
            self.message.emit("No summation runs supplied – nothing to do.")
            return
        
        first_keys  = set(self.summation_image_runs[0]["images"].keys())
        first_count = len(first_keys)
        bad_runs    = []
    
        for idx, run in enumerate(self.summation_image_runs[1:], start=2):
            keys = set(run["images"].keys())
            if len(keys) != first_count:
                bad_runs.append(
                    f"run {idx} has {len(keys)} image(s) vs {first_count}"
                )
            elif keys != first_keys:
                extra   = keys - first_keys
                missing = first_keys - keys
                detail  = []
                if extra:   detail.append(f"extra: {sorted(extra)[:3]}…")
                if missing: detail.append(f"missing: {sorted(missing)[:3]}…")
                bad_runs.append(f"run {idx} suffix mismatch ({'; '.join(detail)})")
    
        if bad_runs:
            self.message.emit(
                "❌  Image‑count / suffix mismatch detected – "
                "aborting summation:\n    " + "\n    ".join(bad_runs)
            )
            return  # hard abort – do not touch FITS files
        
        suffixes = sorted(first_keys)
        total    = len(suffixes)

        # Find common suffixes
        common_suffixes = set(self.summation_image_runs[0]["images"])
        for run in self.summation_image_runs[1:]:
            common_suffixes.intersection_update(run["images"].keys())

        if not common_suffixes:
            self.message.emit("No common suffixes across runs – skipping image summation.")
            return

        suffixes = sorted(common_suffixes)
        total    = len(suffixes)

        self.message.emit("--- Starting image summation ---")
        self.progress_updated.emit(0)

        for idx, suffix in enumerate(suffixes, start=1):
            if not self._is_running:
                self.message.emit("Image summation cancelled by user.")
                break

            summed   = None
            for run in self.summation_image_runs:
                if not self._is_running:
                    break
                try:
                    img = run["images"][suffix]
                    img = img.astype(np.float32, copy=False)

                    if summed is None:
                        summed = img.copy()
                    else:
                        if summed.shape != img.shape:
                            self.message.emit(
                                f"[WARNING] Shape mismatch for suffix '{suffix}' in "
                                f"run '{run['folder_path']}'. Skipping this suffix."
                            )
                            summed = None
                            break
                        summed += img
                except KeyError:
                    self.message.emit(
                        f"[WARNING] Run '{run['folder_path']}' missing suffix '{suffix}' – skipped."
                    )
                    summed = None
                    break
                except Exception as exc:
                    self.message.emit(
                        f"[ERROR] Problem reading suffix '{suffix}' in run "
                        f"'{run['folder_path']}': {exc}"
                    )
                    summed = None
                    break

            # Write FITS
            if self._is_running and summed is not None:
                out_name = f"{self.base_name}_Summed_{suffix}.fits"
                out_path = os.path.join(self.output_folder, out_name)
                try:
                    fits.writeto(out_path, summed, overwrite=True)
                    # self.message.emit(f"Saved summed image '{out_name}'.")
                except Exception as exc:
                    self.message.emit(f"[ERROR] Could not save '{out_name}': {exc}")

            self.progress_updated.emit(int((idx / total) * 100))

    # ───────────────────────────────────────────────────────────────
    # Phase 2  –  ShutterCount summation
    # ───────────────────────────────────────────────────────────────
    def _sum_and_save_shuttercounts(self):
        if not self._is_running:
            return

        shutter_lists = []
        for run_idx, run in enumerate(self.summation_image_runs, start=1):
            if not self._is_running:
                break

            combined = None
            for folder in run.get("run_folders", [run["folder_path"]]):
                if not self._is_running:
                    break

                sc_file = self._find_file(folder, "_ShutterCount.txt")
                if sc_file:
                    try:
                        data = np.loadtxt(sc_file, dtype=np.float32)
                        if data.ndim != 2 or data.shape[1] != 2:
                            self.message.emit(
                                f"[WARNING] Run {run_idx}: Bad ShutterCount format in '{sc_file}'."
                            )
                            continue
                        counts = data[:, 1]
                    except Exception as exc:
                        self.message.emit(
                            f"[WARNING] Run {run_idx}: Cannot read '{sc_file}': {exc}"
                        )
                        counts = np.zeros(256, np.float32)
                else:
                    self.message.emit(
                        f"Run {run_idx}: No ShutterCount in '{folder}'. Using zeros."
                    )
                    counts = np.zeros(256, np.float32)

                combined = counts if combined is None else (
                    combined + counts if len(combined) == len(counts) else combined
                )

            shutter_lists.append(combined if combined is not None
                                 else np.zeros(256, np.float32))

        if not shutter_lists:
            self.message.emit("No ShutterCount data found in any run.")
            return

        base_len = len(shutter_lists[0])
        if any(len(arr) != base_len for arr in shutter_lists):
            self.message.emit("[WARNING] ShutterCount length mismatch – skipping save.")
            return

        summed   = np.sum(shutter_lists, axis=0)
        out_data = np.column_stack((np.arange(base_len), summed))
        out_name = f"{self.base_name}_summed_ShutterCount.txt"
        out_path = os.path.join(self.output_folder, out_name)

        if not self._is_running:
            return
        try:
            np.savetxt(out_path, out_data, fmt="%d\t%d")
            self.message.emit(f"Saved summed ShutterCount '{out_name}'.")
        except OSError as exc:
            self.message.emit(f"[ERROR] Could not write '{out_name}': {exc}")

    # ───────────────────────────────────────────────────────────────
    # Phase 3  –  Copy Spectra files
    # ───────────────────────────────────────────────────────────────
    def _copy_spectra_files(self):
        if not self._is_running:
            return

        for run_idx, run in enumerate(self.summation_image_runs, start=1):
            if not self._is_running:
                break

            copied = False
            for folder in run.get("run_folders", [run["folder_path"]]):
                spectra = self._find_file(folder, "_Spectra.txt")
                if spectra:
                    dest_name = f"{self.base_name}_{run_idx}_Spectra.txt"
                    dest_path = os.path.join(self.output_folder, dest_name)
                    try:
                        shutil.copyfile(spectra, dest_path)
                        self.message.emit(f"Copied Spectra → '{dest_name}'.")
                        copied = True
                    except OSError as exc:
                        self.message.emit(
                            f"[WARNING] Could not copy Spectra from '{folder}': {exc}"
                        )
                    break           # only first spectra per run
            if not copied:
                self.message.emit(f"Run {run_idx}: No Spectra file found.")

        self.message.emit(
            f"Summation complete – files written to <b>'\\{self._short_path}\\'</b> "
            f"with base name '{self.base_name}'."
        )
        self.progress_updated.emit(100)

    # ───────────────────────────────────────────────────────────────
    # Utility helpers
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _find_file(folder: str, suffix: str) -> str | None:
        try:
            for f in os.listdir(folder):
                if f.endswith(suffix):
                    return os.path.join(folder, f)
        except OSError:
            return None
        return None

class OverlapCorrectionWorker(QThread):
    progress_updated = pyqtSignal(int)  # Emits progress percentage
    finished = pyqtSignal()             # Emits when processing is finished
    message = pyqtSignal(str)           # Emits messages for user feedback

    def __init__(self, run, base_name, output_folder):
        """
        Initialize the OverlapCorrectionWorker.

        Args:
            run (dict): Dictionary containing 'folder_path', 'images', 'spectra', 'shutter_count'.
            base_name (str): Base name for the output files.
            output_folder (str): Folder where corrected images will be saved.
        """
        super().__init__()
        self.run = run
        self.base_name = base_name
        self.output_folder = output_folder
        self._is_running = True

    def run(self):
        """
        Execute the Overlap Correction process.
        """
        try:
            folder_path = self.run['folder_path']
            images_dict = self.run['images']
            spectra_data = self.run['spectra']
            shutter_count_data = self.run['shutter_count']
            
            normalized_path = os.path.normpath(folder_path)
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) >= 2:
                short_path = os.path.join(path_parts[-2], path_parts[-1])
            else:
                short_path = normalized_path

            # 1) Extract the first column (ToF values) from spectra
            try:
                # Only cast to float32 if needed
                if spectra_data.dtype != np.float32:
                    spectra_data = spectra_data.astype(np.float32)

                tof_values = spectra_data[:, 0]  # Now guaranteed float32 if it wasn't
                self.message.emit("Extracted ToF values from Spectra data.")
            except Exception as e:
                self.message.emit(f"Error extracting ToF values: {e}. Aborting run.")
                self.finished.emit()
                return

            # 2) Calculate intervals and identify segmentation points
            try:
                tof_intervals = np.diff(tof_values)
                segmentation_indices = np.where(tof_intervals > 0.0001)[0] + 1
                segments = np.split(np.arange(len(tof_values)), segmentation_indices)
                n_segments = len(segments)
                self.message.emit(f"Identified {n_segments} segments based on ToF intervals.")
            except Exception as e:
                self.message.emit(f"Error during ToF segmentation: {e}. Aborting run.")
                self.finished.emit()
                return

            # 3) Extract shutter counts for these segments
            try:
                if len(shutter_count_data) < n_segments:
                    self.message.emit(f"Insufficient shutter counts for {n_segments} segments. Aborting run.")
                    self.finished.emit()
                    return

                # Filter only counts > 1000
                filtered_shutter_counts = shutter_count_data[shutter_count_data > 1000]
                if len(filtered_shutter_counts) < n_segments:
                    self.message.emit(
                        f"Only {len(filtered_shutter_counts)} shutter counts > 1000 found, "
                        f"but {n_segments} segments identified. Aborting run."
                    )
                    self.finished.emit()
                    return

                # Take the first n_segments shutter counts from the filtered list
                selected_shutter_counts = filtered_shutter_counts[:n_segments]
                self.message.emit(f"Selected {n_segments} shutter counts for {n_segments} segments.")
            except Exception as e:
                self.message.emit(f"Error processing shutter counts: {e}. Aborting run.")
                self.finished.emit()
                return

            # 3.1) Compute each segment’s mean ToF interval
            segment_intervals = []
            for idx, seg in enumerate(segments):
                if len(seg) <= 1:
                    # No interval or only 1 point
                    mean_interval = 0.0 if len(seg) < 1 else 1e-5
                else:
                    seg_tofs = tof_values[seg]
                    seg_diffs = np.diff(seg_tofs)
                    mean_interval = float(np.mean(seg_diffs))
                segment_intervals.append(mean_interval)

            # Reference interval is that of the first segment
            ref_interval = segment_intervals[0]
            if ref_interval == 0:
                self.message.emit("First segment's interval is 0. Cannot normalize to T1=0.")
                self.finished.emit()
                return

            # Debug messages
            self.message.emit(f"Segment intervals: {segment_intervals}")
            self.message.emit(f"Reference interval (segment 1) = {ref_interval:.7f}")

            # 4) Initialize cumulative intensity arrays for each segment.
            #    We'll base the shape/dtype on the first image in images_dict.
            #    If no images, this could raise KeyError.
            try:
                first_img_key = next(iter(images_dict.keys()))
                first_img_data = images_dict[first_img_key]
                if first_img_data.dtype != np.float32:
                    first_img_data = first_img_data.astype(np.float32)
                shape_512 = first_img_data.shape

                # Quick check the shape is what's expected (512, 512)
                if shape_512 != (512, 512):
                    self.message.emit(f"Image dimensions {shape_512} do not match expected (512, 512). Aborting run.")
                    self.finished.emit()
                    return

                # Create a zero array for each segment
                cumulative_intensities = [
                    np.zeros(shape_512, dtype=np.float32) for _ in segments
                ]
            except StopIteration:
                self.message.emit("No images found in the run. Aborting.")
                self.finished.emit()
                return
            except Exception as e:
                self.message.emit(f"Error initializing cumulative arrays: {e}. Aborting.")
                self.finished.emit()
                return

            # 5) Sort images by ToF order based on numeric suffix
            try:
                def extract_numeric_suffix(suf):
                    # safer approach: filter digits out of the string
                    digits = ''.join(filter(str.isdigit, suf))
                    return int(digits) if digits else -1

                sorted_suffixes = sorted(images_dict.keys(), key=extract_numeric_suffix)
                sorted_images = [images_dict[suf] for suf in sorted_suffixes]
            except Exception as e:
                self.message.emit(f"Error sorting images: {e}. Aborting run.")
                self.finished.emit()
                return

            self.message.emit("--- Starting Overlap Correction ---")

            # 6) Process each image individually
            total_imgs = len(sorted_suffixes)
            for img_idx, suf in enumerate(sorted_suffixes):
                if not self._is_running:
                    self.message.emit("Overlap Correction process has been stopped by the user.")
                    break

                try:
                    image_data = sorted_images[img_idx]
                    # Only cast if needed
                    if image_data.dtype != np.float32:
                        image_data = image_data.astype(np.float32)

                    # Identify the segment this image belongs to
                    # (img_idx is the index in sorted order, not necessarily the real "ToF" index,
                    #  so you might want a different logic if needed. We'll keep your approach.)
                    segment_number = None
                    for seg_num, seg_indices in enumerate(segments):
                        if img_idx in seg_indices:
                            segment_number = seg_num
                            break

                    if segment_number is None:
                        self.message.emit(f"Image {img_idx+1}: No matching segment. Skipping.")
                        continue

                    # If it's the first image in that segment, overwrite
                    seg_idx_within = np.where(segments[segment_number] == img_idx)[0][0]
                    if seg_idx_within == 0:
                        cumulative_intensities[segment_number] = image_data.copy()
                    else:
                        cumulative_intensities[segment_number] += image_data

                    shutter_count = np.float32(selected_shutter_counts[segment_number])
                    if shutter_count == 0:
                        self.message.emit(
                            f"Image {img_idx+1}: Shutter count = 0 for segment {segment_number+1}. Skipping normalisation."
                        )
                        continue

                    # Step 7) Calculate p value
                    #   p = (cumulative_intensity in that segment) / shutter_count
                    p = cumulative_intensities[segment_number] / shutter_count

                    # Step 8) Correct intensities = original intensity / (1 - p)
                    #   and scale by (ref_interval / this_interval)
                    # denom = 1.0 - p
                    denom  = np.where(1.0 - p <= 0, np.float32(1e-10), 1.0 - p)
                    epsilon = 1e-10
                    denom = np.where(denom <= 0, epsilon, denom)  # avoid div-by-zero
                    corrected_intensity = image_data / denom

                    # Scale factor for time intervals
                    this_interval = np.float32(segment_intervals[segment_number])
                    ref_interval  = np.float32(segment_intervals[0])
                    scale_factor  = ref_interval / this_interval if this_interval > 0 else np.float32(1.0)

                    # this_interval = segment_intervals[segment_number]
                    # scale_factor = ref_interval / this_interval if this_interval > 0 else 1.0
                    corrected_intensity *= scale_factor

                    # Check NaN/Inf
                    if np.isnan(corrected_intensity).any() or np.isinf(corrected_intensity).any():
                        self.message.emit(f"Image {img_idx+1}: NaN or Inf after correction. Skipping.")
                        continue

                    # Construct output path
                    try:
                        numeric_suffix = ''.join(filter(str.isdigit, suf))
                        original_filename = f"{self.base_name}_{numeric_suffix}.fits"
                        corrected_filename = f"Corrected_{original_filename}"
                        output_path = os.path.join(self.output_folder, corrected_filename)
                    except Exception as e:
                        self.message.emit(f"Error constructing filename: {e}. Skipping.")
                        continue

                    # Save corrected image
                    fits.writeto(output_path, corrected_intensity, overwrite=True)

                    # Update progress
                    overall_progress = int(((img_idx + 1) / total_imgs) * 100)
                    self.progress_updated.emit(overall_progress)

                except Exception as e:
                    self.message.emit(f"Error processing image '{suf}': {e}. Skipping.")
                    continue

            # Final messages
            if self._is_running:
                self.message.emit("Overlap Correction process completed successfully.")
            else:
                self.message.emit("Overlap Correction process was stopped before completion.")

            # 9) Copy spectra and shuttercount files to output folder
            try:
                spectra_suffix = '_Spectra.txt'
                shuttercount_suffix = '_ShutterCount.txt'

                # Copy spectra files
                spectra_files = [f for f in os.listdir(folder_path) if f.endswith(spectra_suffix)]
                if not spectra_files:
                    self.message.emit(f"No files ending with '{spectra_suffix}' found in \\{short_path}.")
                else:
                    for spectra_file in spectra_files:
                        source_path = os.path.join(folder_path, spectra_file)
                        dest_path = os.path.join(self.output_folder, spectra_file)
                        shutil.copyfile(source_path, dest_path)
                        self.message.emit(f"'{spectra_file}' copied to output folder.")

                # Copy shuttercount files
                shuttercount_files = [f for f in os.listdir(folder_path) if f.endswith(shuttercount_suffix)]
                if not shuttercount_files:
                    self.message.emit(f"No files ending with '{shuttercount_suffix}' found in \\{short_path}.")
                else:
                    for shuttercount_file in shuttercount_files:
                        source_path = os.path.join(folder_path, shuttercount_file)
                        dest_path = os.path.join(self.output_folder, shuttercount_file)
                        shutil.copyfile(source_path, dest_path)
                        self.message.emit(f"'{shuttercount_file}' copied to output folder.")

            except Exception as e:
                self.message.emit(f"Error copying spectra or shuttercount files: {e}")

        except Exception as e:
            # If a top-level error happened, log and skip gracefully
            self.message.emit(f"Error in OverlapCorrectionWorker: {e}")

        # Optional: if memory usage is extremely high, you could call gc.collect() once here
        # gc.collect()

        # Emit finished signal
        self.message.emit("Overlap Correction completed successfully.")
        self.finished.emit()

    def stop(self):
        """
        Stop the Overlap Correction process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Overlap Correction process.")


# import os
import re
# import gc
import psutil
import numpy as np
# from astropy.io import fits
from PyQt5.QtCore import QThread, pyqtSignal

class NormalisationWorker(QThread):
    progress_updated = pyqtSignal(int)   # Emits progress percentage (0-100)
    finished         = pyqtSignal()      # Emits when processing is finished
    message          = pyqtSignal(str)   # Emits messages for user feedback

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
        self.normalisation_image_runs      = normalisation_image_runs
        self.normalisation_open_beam_runs  = normalisation_open_beam_runs
        self.output_folder                 = output_folder
        self.base_name                     = base_name
        self.window_half                   = window_half
        self.adjacent_sum                  = adjacent_sum
        self._is_running                   = True

    @staticmethod
    def _read_shutter_count(folder_path):
        """Read the 2nd column, 1st row from *_ShutterCount.txt."""
        try:
            fname = next(f for f in os.listdir(folder_path)
                         if f.endswith('_ShutterCount.txt'))
        except StopIteration:
            raise FileNotFoundError("no *_ShutterCount.txt found")

        with open(os.path.join(folder_path, fname), 'r') as fh:
            first_line = fh.readline().strip()

        parts = [p for p in re.split(r'[,\s]+', first_line) if p]
        if len(parts) < 2:
            raise ValueError(f"cannot parse shutter count in {fname}")

        return float(parts[1])

    def run(self):
        try:
            # prepare a short display path
            norm_path = os.path.normpath(self.output_folder)
            parts = norm_path.split(os.sep)
            short_path = os.path.join(parts[-2], parts[-1]) if len(parts) >= 2 else norm_path

            # basic validation
            if not self.normalisation_image_runs or not self.normalisation_open_beam_runs:
                self.message.emit("No runs provided. Aborting.")
                return

            if len(self.normalisation_image_runs) != len(self.normalisation_open_beam_runs):
                self.message.emit("Data vs. Open‐beam count mismatch. Aborting.")
                return

            total_images = sum(len(r['images']) for r in self.normalisation_image_runs)
            processed_images = 0

            self.message.emit("<b>--- Starting Normalisation ---</b>")
            window_half = self.window_half
            full_win    = (2*window_half+1)**2
            thresh      = 1e-7
            frame_win   = 2*self.adjacent_sum+1

            self.message.emit(
                f"Using {2*window_half+1}×{2*window_half+1} spatial window "
                f"and {frame_win} frames."
            )

            for run_idx, (data_run, ob_run) in enumerate(
                zip(self.normalisation_image_runs,
                    self.normalisation_open_beam_runs), start=1
            ):
            
                if not self._is_running:
                    self.message.emit("User stopped the process.")
                    break

                # read shutter counts
                try:
                    sc = self._read_shutter_count(data_run['folder_path'])
                    ob = self._read_shutter_count(ob_run['folder_path'])
                    scale = np.float32(ob/sc) if sc>0 else 1.0
                    self.message.emit(
                        f"sample={sc:.0f}, open‐beam={ob:.0f}, scale={scale:.4f}"
                    )
                except Exception as e:
                    self.message.emit(
                        f"shutter‐count error ({e}), scale=1.0"
                    )
                    scale = np.float32(1.0)

                data_imgs = data_run['images']
                ob_imgs   = ob_run['images']
                common    = sorted(set(data_imgs) & set(ob_imgs))
                if not common:
                    self.message.emit(f" no matching suffixes—skipping.")
                    continue

                for i, suffix in enumerate(common):
                    if not self._is_running:
                        break
  

                    try:
                        img  = data_imgs[suffix].astype(np.float32)
                        
                        if self.adjacent_sum == 0:
                            ob0 = ob_imgs[suffix].astype(np.float64).copy()
                            start = 0
                            end = 0

                        else:
                        # build combined open‐beam
                            start = max(0, i-self.adjacent_sum)
                            end   = min(len(common)-1, i+self.adjacent_sum)
                            ob0   = ob_imgs[common[start]].astype(np.float64).copy()
                            for j in range(start+1, end+1):
                                ob0 += ob_imgs[common[j]].astype(np.float64)

                        if img.shape != ob0.shape:
                            self.message.emit(
                                f" {suffix}: shape mismatch—skipping."
                            )
                            continue

                        h, w = img.shape
                        if h < 2*window_half+1 or w < 2*window_half+1:
                            self.message.emit(
                                f" {suffix}: too small for window—skipping."
                            )
                            continue

                        # integral images
                        II = ob0.cumsum(0).cumsum(1)
                        II = np.pad(II, ((1,0),(1,0)), 'constant')
                        II1 = np.pad(np.ones_like(img).cumsum(0).cumsum(1), ((1,0),(1,0)), 'constant')

                        # get sums via broadcasted indices
                        I, J = np.ogrid[:h, :w]
                        i0, i1 = I-window_half, I+window_half+1
                        j0, j1 = J-window_half, J+window_half+1
                        i0, i1 = np.clip(i0,0,h), np.clip(i1,0,h)
                        j0, j1 = np.clip(j0,0,w), np.clip(j1,0,w)

                        part_sum = II[i1, j1] - II[i0, j1] - II[i1, j0] + II[i0, j0]
                        part_cnt = II1[i1,j1] - II1[i0,j1] - II1[i1,j0] + II1[i0,j0]
                        scaled   = np.where(part_cnt>0,
                                            part_sum*(full_win/part_cnt),
                                            thresh).astype(np.float32)

                        normed = ((end-start+1)*full_win*img/scaled)*scale
                        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                        out_fname = f"{self.base_name}_{suffix}.fits"
                        fits.writeto(os.path.join(self.output_folder, out_fname),
                                     normed, overwrite=True)

                        del data_imgs[suffix]
                        processed_images += 1
                        self.progress_updated.emit(int(100*processed_images/total_images))

                    except Exception as e:
                        self.message.emit(
                            f" {suffix}: error ({e})—skipping."
                        )

                        
                    finally:
                        # free big arrays
                        for v in ('img','ob0','II','II1','scaled','normed'):
                            if v in locals():
                                del locals()[v]
                    if suffix == 1500:
                        gc.collect()
                        QThread.sleep(2)
                    if suffix == 2500:
                        gc.collect()
                        QThread.sleep(2)
            
                # per‐run cleanup
                data_imgs.clear()
                gc.collect()
                try:
                    self.copy_related_files(run_idx, data_run)
                except Exception as e:
                    self.message.emit(f"copy_related_files failed: {e}")

                self.message.emit("Run done.")
                if self._is_running:
                    QThread.sleep(5)

            self.message.emit(f"All done—{processed_images} images → {short_path}")

        except Exception as e:
            self.message.emit(f"Fatal error in normalisation: {e}")

        finally:
            # final cleanup & memory report
            gc.collect()
            proc  = psutil.Process(os.getpid())
            memMB = proc.memory_info().rss / (1024.**2)
            self.message.emit(f"<b>Final memory usage:</b> {memMB:.1f} MB")
            self.finished.emit()

    def stop(self):
        """
        Stop the Normalisation process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Normalisation process.")

    def copy_related_files(self, run_idx, data_run):
        """
        Copies related files (e.g., *_Spectra.txt and *_ShutterCount.txt) from the data run's folder
        to the output folder with unique run identifiers.
        """
        try:
            spectra_suffix = '_Spectra.txt'
            shuttercount_suffix = '_ShutterCount.txt'

            def create_unique_filename(run_number, original_filename):
                return f"Run{run_number}_{original_filename}"

            data_folder = data_run['folder_path']
            data_files = os.listdir(data_folder)

            # Spectra
            spectra_files = [f for f in data_files if f.endswith(spectra_suffix)]
            if spectra_files:
                for sf in spectra_files:
                    source_path = os.path.join(data_folder, sf)
                    dest_filename = create_unique_filename(run_idx, sf)
                    dest_path = os.path.join(self.output_folder, dest_filename)
                    shutil.copyfile(source_path, dest_path)
                    self.message.emit(f"Copied '{sf}' to '{dest_filename}'.")
                    # If only one file is expected, you could break here
            else:
                self.message.emit(f"No file ending with '{spectra_suffix}' found in {data_folder}.")

            # ShutterCount
            shuttercount_files = [f for f in data_files if f.endswith(shuttercount_suffix)]
            if shuttercount_files:
                for scf in shuttercount_files:
                    source_path = os.path.join(data_folder, scf)
                    dest_filename = create_unique_filename(run_idx, scf)
                    dest_path = os.path.join(self.output_folder, dest_filename)
                    shutil.copyfile(source_path, dest_path)
                    self.message.emit(f"Copied '{scf}' to '{dest_filename}'.")
                    # If only one file is expected, you could break here
            else:
                self.message.emit(f"No file ending with '{shuttercount_suffix}' found in {data_folder}.")

        except Exception as e:
            self.message.emit(f"Error copying related files: {e}")

class FullProcessWorker(QThread):
    # Signals for messages, progress updates and final completion
    message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    load_progress_updated = pyqtSignal(int) 
    finished = pyqtSignal()

    def __init__(self, sample_folder: str, open_beam_folder: str, output_folder: str,
                 base_name: str, window_half: int, adjacent_sum: int):
        super().__init__()
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True  # Flag to track whether the user requested a stop
        
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

            self.message.emit("=== <b>Full Process Completed Successfully</b> ===")

        except Exception as exc:
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
        self.message.emit("FullProcessWorker: Stop signal received.")

    # ---------------------------------------------------------------
    # Summation Step (conditionally skipped if no subfolders)
    # ---------------------------------------------------------------
    def maybe_do_summation(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
    
        # Check for subfolders
        subfolders = [
            os.path.join(folder, d)
            for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
        
        short_path = self.get_short_path(folder, levels=2)
    
        if not subfolders:
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
            self.message.emit(f"0_sumation_{label}: No valid FITS images in subfolders of '\\{short_path}'. Summation skipped.")
            return folder
    
        # Create a modified base name for image naming (e.g., "summed_sample_data")
        modified_base_name = f"summed_{original_folder_name}"
        worker = SummationWorker(runs, modified_base_name, summation_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
               
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        worker.start(); 
        loop.exec_()     
    
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
            self.message.emit(f"1_clean_{label}: No images found in \\{short_path}. Skipping Clean.")
            return folder
    
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
        worker.start(); 
        loop.exec_()
    
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
            self.message.emit(f"2_correction_{label}: No images found in \\{short_path}. Skipping Overlap Correction.")
            return folder
    
        try:
            all_files = os.listdir(folder)
        except Exception as e:
            self.message.emit(f"Error accessing \\{short_path}: {e}")
            return folder
    
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
            self.message.emit(f"2_correction_{label}: Spectra or ShutterCount missing => Skipping Overlap Correction.")
            return folder
    
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
        worker.start(); 
        loop.exec_()
    
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
            self.message.emit("Normalisation skipped because sample or openbeam has no images.")
            return
    
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
        worker.start(); 
        loop.exec_()
    
        del sample_run, openbeam_run
        import gc
        gc.collect()
        
        normalised_output_short=self.get_short_path(normalised_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>3_normalisation complete</b>, saved at: <b>\\{normalised_output_short}</b>")

  
    def load_run_dict(self, folder: str) -> dict:
        """
        Scans the folder for FITS files with a 5-digit suffix and returns a dictionary:
            { 'folder_path': folder, 'images': {suffix: data} }.
        Also emits loading progress via load_progress_updated.
        """
        # import re
        run = {"folder_path": folder, "images": {}}
        if not os.path.isdir(folder):
            self.message.emit(f"Folder not found: {folder}")
            return run
        
        short_path = self.get_short_path(folder, levels=2)

        try:
            fits_files = [f for f in os.listdir(folder) if f.lower().endswith((".fits", ".fit"))]
            # Filter only files with the proper 5-digit suffix format:
            pattern = re.compile(r'^.+_(\d{5})\.(fits|fit)$', re.IGNORECASE)
            total_files = len(fits_files)
            processed_files = 0

            for f in fits_files:
                processed_files += 1
                match = pattern.match(f)
                if not match:
                    # self.message.emit(f"Skipping file '{f}' because it does not have a 5-digit suffix.")
                    # Emit progress update even if skipped
                    self.load_progress_updated.emit(int((processed_files / total_files) * 100))
                    continue

                # Use the captured 5-digit group as the suffix
                suffix = match.group(1)
                try:
                    path = os.path.join(folder, f)
                    data = fits.getdata(path)
                    run["images"][suffix] = data.astype(np.float32)
                except Exception as e:
                    self.message.emit(f"Error loading file {f} in \\{short_path}: {e}")
                # Update loading progress after processing each file
                self.load_progress_updated.emit(int((processed_files / total_files) * 100))
        except Exception as e:
            self.message.emit(f"Error reading folder \\{short_path}: {e}")
        return run

           
class FilteringWorker(QThread):
    progress_updated = pyqtSignal(int)  # Emits progress percentage
    finished = pyqtSignal()             # Emits when processing is finished
    message = pyqtSignal(str)           # Emits messages for user feedback

    def __init__(self, filtering_image_runs, filtering_mask, output_folder, base_name):
        super().__init__()
        self.filtering_image_runs = filtering_image_runs  # List of data run dictionaries
        self.filtering_mask = filtering_mask              # Single mask image array
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True

    def run(self):
        """
        Main filtering process.
        """
        try:
            # 1) Basic validations
            if not self.filtering_image_runs:
                self.message.emit("No filtering data runs to process.")
                self.finished.emit()
                return

            if self.filtering_mask is None:
                self.message.emit("No mask image provided.")
                self.finished.emit()
                return

            # 2) Convert the mask to float32 or bool (if not already), only once
            if self.filtering_mask.dtype != np.float32 and self.filtering_mask.dtype != bool:
                # If it's an integer or float32, consider converting
                # Example: let's keep it float32 for consistency:
                self.filtering_mask = self.filtering_mask.astype(np.float32)

            mask_shape = self.filtering_mask.shape

            # 3) Counters for progress
            total_runs = len(self.filtering_image_runs)
            total_images = sum(len(run['images']) for run in self.filtering_image_runs)
            processed_images = 0

            self.message.emit("Filtering started...")

            # 4) Process each run
            for run_idx, data_run in enumerate(self.filtering_image_runs, start=1):
                if not self._is_running:
                    self.message.emit("Filtering process has been stopped by the user.")
                    break

                data_folder = data_run.get('folder_path', None)
                data_images = data_run.get('images', {})

                # Copy .txt files for this run (if desired)
                self.copy_related_files(run_idx, data_run)

                # Sort image keys for a consistent order
                suffixes = sorted(data_images.keys())

                # 5) Process each image in the run
                for suffix in suffixes:
                    if not self._is_running:
                        self.message.emit("Filtering process has been stopped by the user.")
                        break

                    try:
                        image_data = data_images[suffix]
                        # Convert to float32 if needed
                        if image_data.dtype != np.float32:
                            image_data = image_data.astype(np.float32)

                        if image_data.shape != mask_shape:
                            self.message.emit(
                                f"Image {suffix}: Mask shape {mask_shape} "
                                f"does not match image shape {image_data.shape}. Skipping."
                            )
                            continue

                        # Apply the mask
                        # If your mask is float32, we can do: (mask != 0)
                        # Or if you kept it as bool, simply: np.where(mask, image_data, 0)
                        # Example assuming mask != 0 means “keep pixel”:
                        filtered_image = np.where(self.filtering_mask != 0, image_data, 0)

                        # Save the filtered image
                        filtered_filename = f"{self.base_name}_{suffix}.fits"
                        filtered_path = os.path.join(self.output_folder, filtered_filename)
                        try:
                            fits.writeto(filtered_path, filtered_image, overwrite=True)
                        except Exception as e:
                            self.message.emit(f"Image {suffix}: Failed to save '{filtered_filename}': {e}. Skipping.")
                            continue

                        # Update progress
                        processed_images += 1
                        overall_progress = int((processed_images / total_images) * 100)
                        self.progress_updated.emit(overall_progress)

                    except Exception as e:
                        self.message.emit(f"Error filtering image {suffix}: {e}")
                        continue

                # Optional: update overall run progress
                run_progress = int((run_idx / total_runs) * 100)
                self.progress_updated.emit(run_progress)
                
            self.output_folder_short = self.get_short_path(self.output_folder, levels=2)

            self.message.emit(
                f"Filtering completed. {processed_images} images filtered and saved to {self.output_folder_short}."
            )

        except Exception as e:
            self.message.emit(f"Error during filtering: {e}")

        finally:
            # Call gc.collect() once at the end if you need to enforce cleanup
            gc.collect()
            self.finished.emit()

    def stop(self):
        """
        Stop the Filtering process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Filtering process.")

    def copy_related_files(self, run_idx, data_run):
        """
        Copies related files (e.g., *_Spectra.txt and *_ShutterCount.txt) from the data run's folder
        to the output folder, renaming them with the run_idx to avoid collisions.
        """
        try:
            folder_path = data_run.get('folder_path', None)
            if not folder_path or not os.path.isdir(folder_path):
                self.message.emit(f"Data run folder not found or invalid: {folder_path}")
                return

            data_files = os.listdir(folder_path)
            spectra_suffix = '_Spectra.txt'
            shuttercount_suffix = '_ShutterCount.txt'

            def create_unique_filename(run_number, original_filename):
                return f"Run{run_number}_{original_filename}"

            # Copy one spectra file if it exists
            spectra_files = [f for f in data_files if f.endswith(spectra_suffix)]
            if spectra_files:
                file = spectra_files[0]
                src = os.path.join(folder_path, file)
                dest_filename = create_unique_filename(run_idx, file)
                dst = os.path.join(self.output_folder, dest_filename)
                shutil.copyfile(src, dst)
                self.message.emit(f"Copied '{file}' to '{dest_filename}'.")
            else:
                self.message.emit(f"No spectra file (*{spectra_suffix}) found in {folder_path}.")

            # Copy one shuttercount file if it exists
            shuttercount_files = [f for f in data_files if f.endswith(shuttercount_suffix)]
            if shuttercount_files:
                file = shuttercount_files[0]
                src = os.path.join(folder_path, file)
                dest_filename = create_unique_filename(run_idx, file)
                dst = os.path.join(self.output_folder, dest_filename)
                shutil.copyfile(src, dst)
                self.message.emit(f"Copied '{file}' to '{dest_filename}'.")
            else:
                self.message.emit(f"No shuttercount file (*{shuttercount_suffix}) found in {folder_path}.")

        except Exception as e:
            self.message.emit(f"Error copying related files: {e}")

    


    
def main():
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    # Optionally set a style sheet if needed
    app.setStyleSheet("""
    QLineEdit {
        max-height: 30px;
    }
    """)

    viewer = FitsViewer()
    viewer.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
