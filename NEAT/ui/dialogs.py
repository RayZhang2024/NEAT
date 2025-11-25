"""Dialog and plotting widgets used by the NEAT UI."""

import os

import numpy as np
from astropy.io import fits
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDesktopWidget,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom


def _scaled_geometry(width_factor: float, height_factor: float) -> tuple[int, int]:
    """Return window dimensions scaled relative to the screen size."""
    app = QApplication.instance()
    if app is not None:
        geometry = QDesktopWidget().screenGeometry()
        return (
            int(geometry.width() * width_factor),
            int(geometry.height() * height_factor),
        )
    # Fallback defaults if no QApplication is active yet.
    default_width = int(1920 * width_factor)
    default_height = int(1080 * height_factor)
    return default_width, default_height

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class FitVisualizationDialog(QDialog):
    """
    Shows the full-pattern fit plus dashed Region-3 edge fits.
    Numerical results are displayed below the plot (not over it).
    """
    def __init__(self, x_pos, y_pos, box_width, box_height,
                 parameters, parent=None):
        super().__init__(parent)
        self.x_pos      = x_pos
        self.y_pos      = y_pos
        self.box_width  = box_width
        self.box_height = box_height
        self.parameters = parameters

        # Window & widgets
        self.setWindowTitle("Fit check")
        self.setGeometry(200, 200, 1200, 1000)

        main_layout = QVBoxLayout(self)

        # Top widget: plot + toolbar
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        self.canvas  = MplCanvas(self, width=5, height=4, dpi=100)
        top_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
        top_layout.addWidget(self.toolbar)

        # Bottom widget: info panel (kept off the plot)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(180)
        bottom_layout.addWidget(self.info_box)

        # Splitter allows adjustable heights between plot and text
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_plot)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

        self.plot_fit()                 # do the drawing

    def plot_fit(self):
        """Draw experimental data, full-pattern model and edge curves."""
        ax = self.canvas.axes
        ax.clear()

        p = self.parameters
        x_exp, y_exp    = p['x_exp'],  p['y_exp']
        model_x, model_y= p['model_x'], p['model_y']

        # Experimental data & global pattern (smaller markers/lines to avoid clutter)
        ax.plot(x_exp, y_exp, 'bo', ms=3, label="Experimental")
        if len(model_x):
            ax.plot(model_x, model_y, 'k-', alpha=0.3, lw=2, label="Fitted pattern")

        # Individual Region-3 edge fits (dashed)
        for ef in p.get('edge_fits', []):
            ax.plot(ef['x'], ef['fit'], '--', lw=2, label=f"Edge {ef['hkl']}")

        # Axes cosmetics
        font_size = 10  # unify plot font with info box
        ax.set_xlabel("Wavelength (A)", fontsize=font_size)
        ax.set_ylabel("Summed Intensity", fontsize=font_size)
        ax.set_title(f"ROI centre ({self.x_pos},{self.y_pos})  -  "
                     f"Box {self.box_width}x{self.box_height}", fontsize=font_size+1)
        ax.tick_params(labelsize=font_size)

        # info box with lattice, d, s, t, eta
        info_lines = ["Lattice parameters:"]
        for name, val in p["lattice_params"].items():
            unc = p["lattice_uncertainties"].get(name, 0.0)
            info_lines.append(f"   {name} = {val:.5f} +/- {unc:.5f}")

        # Pattern-fit (global) parameters
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
            info_lines.append("Pattern fit: s / t / eta  (+/- sigma)")

        def _nan_if_none(v):
            return float("nan") if v is None else v

        for hkl in sorted(bulk["s"].keys()):
            d,du = bulk["d"].get(hkl, float("nan")),  bulk["du"].get(hkl, 0.0)
            s,su = bulk["s"][hkl],                    bulk["su"].get(hkl, 0.0)
            t,tu = bulk["t"][hkl],                    bulk["tu"].get(hkl, 0.0)
            e,eu = bulk["e"][hkl],                    bulk["eu"].get(hkl, 0.0)
            d,du,s,su,t,tu,e,eu = map(_nan_if_none, (d,du,s,su,t,tu,e,eu))
            info_lines.append(
                f"   hkl{hkl}: "
                f"s={s:.6f}+/-{su:.6f}, "
                f"t={t:.6f}+/-{tu:.6f}, "
                f"eta={e:.6f}+/-{eu:.6f}"
            )

        # Region-fit parameters (if available)
        edge_fits = p.get("edge_fits", [])
        if edge_fits:
            info_lines.append("")
            info_lines.append("Edge fit: d / s / t / eta  (+/- sigma)")
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
                    f"d={d:.6f}+/-{du:.6f}, "
                    f"s={s:.6f}+/-{su:.6f}, "
                    f"t={t:.6f}+/-{tu:.6f}, "
                    f"eta={e:.6f}+/-{eu:.6f}"
                )

        # Render the text in the info panel (not over the plot)
        self.info_box.setPlainText("\n".join(info_lines))

        # Legend (deduplicate labels and keep small)
        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for h, l in zip(handles, labels):
            dedup[l] = h
        if dedup:
            ax.legend(dedup.values(), dedup.keys(), loc="best", prop={"size": font_size}, framealpha=0.8)

        self.canvas.draw()

    def save_plot(self):
        """
        Save the current figure to an image file.
        """
        opts = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;All Files (*)",
            options=opts,
        )
        if not file_path:
            return
        try:
            self.canvas.fig.savefig(file_path, bbox_inches="tight")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not save plot:\n{e}")
class AdjustmentsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)                      # keep the parent

        self.setWindowTitle("Adjust Image")
        self.parent = parent

        # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ Build the GUI 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        layout = QVBoxLayout()

        # 鉃?Contrast
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

        # 鉃?Brightness
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

        # 鉃?Minimum intensity
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

        # 鉃?Maximum intensity
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

        # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ Size & position 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        #
        # 1. Shrink-wrap to the minimal size that fits the widgets.
        self.adjustSize()
        # 2. Add a small margin so the controls aren鈥檛 flush with the edges.
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
        width, height = _scaled_geometry(0.5, 0.8)
        self.resize(width, height)
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
        control_layout_top = QHBoxLayout()
        control_layout_bottom = QHBoxLayout()

        # Color bar min and max inputs
        self.color_min_input = QLineEdit()
        self.color_max_input = QLineEdit()
        self.d0_input = QLineEdit()
        self.color_min_input.setPlaceholderText("Color Bar Min")
        self.color_max_input.setPlaceholderText("Color Bar Max")
        self.d0_input.setPlaceholderText("d0")
        self.color_min_input.editingFinished.connect(self.update_plot)
        self.color_max_input.editingFinished.connect(self.update_plot)

        control_layout_top.addWidget(QLabel("Color Bar Min:"))
        control_layout_top.addWidget(self.color_min_input)
        control_layout_top.addWidget(QLabel("Color Bar Max:"))
        control_layout_top.addWidget(self.color_max_input)
        control_layout_top.addWidget(QLabel("d0:"))
        control_layout_top.addWidget(self.d0_input)

        # Add a button to calculate strain
        self.calculate_strain_button = QPushButton("Calculate Strain")
        self.calculate_strain_button.clicked.connect(self.calculate_strain)
        control_layout_top.addWidget(self.calculate_strain_button)

        # Add input fields for x_min, x_max, y_min, y_max
        self.x_min_input = QLineEdit()
        self.x_max_input = QLineEdit()
        self.y_min_input = QLineEdit()
        self.y_max_input = QLineEdit()
        self.x_min_input.setPlaceholderText("x min")
        self.x_max_input.setPlaceholderText("x max")
        self.y_min_input.setPlaceholderText("y min")
        self.y_max_input.setPlaceholderText("y max")

        control_layout_bottom.addWidget(QLabel("x min:"))
        control_layout_bottom.addWidget(self.x_min_input)
        control_layout_bottom.addWidget(QLabel("x max:"))
        control_layout_bottom.addWidget(self.x_max_input)
        control_layout_bottom.addWidget(QLabel("y min:"))
        control_layout_bottom.addWidget(self.y_min_input)
        control_layout_bottom.addWidget(QLabel("y max:"))
        control_layout_bottom.addWidget(self.y_max_input)

        # Add a button to calculate the mean value
        self.calculate_mean_button = QPushButton("Mean, Std Dev")
        self.calculate_mean_button.clicked.connect(self.calculate_mean)
        control_layout_bottom.addWidget(self.calculate_mean_button)

        self.unit_switch_checkbox = QCheckBox("mm")
        self.unit_switch_checkbox.stateChanged.connect(self.toggle_units)
        control_layout_bottom.addWidget(self.unit_switch_checkbox)

        # Add a toggle button to enable/disable point selection
        self.toggle_select_button = QPushButton("Line profile")
        self.toggle_select_button.setCheckable(True)
        self.toggle_select_button.setStyleSheet("background-color: none")
        self.toggle_select_button.clicked.connect(self.toggle_select_mode)
        control_layout_bottom.addWidget(self.toggle_select_button)

        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.apply_filter)
        control_layout_bottom.addWidget(self.filter_button)

        # Add a button to save as FITS
        self.save_fits_button = QPushButton("Save image")
        self.save_fits_button.clicked.connect(self.save_as_fits)
        control_layout_bottom.addWidget(self.save_fits_button)

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
        main_layout.addLayout(control_layout_top)
        main_layout.addLayout(control_layout_bottom)

        # Add Matplotlib's Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
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
        if not file_path:           # user hit 鈥淐ancel鈥?
            return
    
        try:
            # ------------------------------------------------------------------
            # 2. Prepare data (resize 鈫?float32 鈫?flip vertically)
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
        if not cmin_text or not cmax_text:
            mean_val = np.nanmean(self.Z)
            std_val = np.nanstd(self.Z)
            cmin = mean_val - 2 * std_val
            cmax = mean_val + 2 * std_val
            self.color_min_input.setText(f"{cmin:.4f}")
            self.color_max_input.setText(f"{cmax:.4f}")
        else:
            cmin = float(cmin_text)
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
            self.toggle_select_button.setText("Line profile")
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
    
            # Calculate mean and standard deviation
            mean_value = np.nanmean(Z_selected)
            std_value = np.nanstd(Z_selected)
    
            # Update the mean label
            unit = '脜' if self.parameter_name != 'Strain' else '碌蔚'  # 碌蔚 for microstrain
            self.mean_label.setText(f"Mean: {mean_value:.6f} {unit} | Std: {std_value:.6f} {unit}")
    
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
                value_str = f"{z_value:.6f} 脜" if z_value is not None else "Out of range"
            else:
                value_str = f"{z_value:.0f} 碌蔚" if z_value is not None else "Out of range"
    
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
        width, height = _scaled_geometry(0.7, 0.7)
        self.setGeometry(200, 200, width, height)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Store the data for saving
        self.distances = np.sqrt((x_coords - x_coords[0])**2 + (y_coords - y_coords[0])**2)
        self.z_values = z_values


        # Matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
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

class OpenBeamPlotDialog(QDialog):
    def __init__(self, wavelengths, intensities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Beam Intensity vs Wavelength")
        width, height = _scaled_geometry(0.5, 0.5)
        self.setGeometry(200, 200, width, height)
        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=7, height=5, dpi=100)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Plotting
        self.canvas.axes.plot(wavelengths, intensities, 'o-', color='green')
        self.canvas.axes.set_xlabel("Wavelength (脜)")
        self.canvas.axes.set_ylabel("Summed Intensity")
        self.canvas.axes.set_title("Open Beam Intensity vs Wavelength")
        self.canvas.draw()

__all__ = [
    "MplCanvas",
    "FitVisualizationDialog",
    "AdjustmentsDialog",
    "ParameterPlotDialog",
    "LineProfileDialog",
    "OpenBeamPlotDialog",
]


