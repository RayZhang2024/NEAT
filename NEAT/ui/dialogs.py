"""Dialog and plotting widgets used by the NEAT UI."""

import os
from collections import deque

import numpy as np
from astropy.io import fits
from PIL import Image
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDesktopWidget,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
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
from matplotlib.patches import Circle, Rectangle
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


class MaskGeneratorDialog(QDialog):
    """Interactive mask generator for preprocessing filtering."""

    mask_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Generator")
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowSystemMenuHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        width, height = _scaled_geometry(0.5, 0.55)
        self.setGeometry(200, 140, width, height)
        self.setMinimumSize(760, 520)

        self.source_image = None
        self.current_mask = None
        self._threshold_min = None
        self._threshold_max = None
        self._painting = False
        self._brush_outline = None
        self._last_cursor = None
        self._undo_stack = deque(maxlen=50)
        self._overlay_colors = {
            "Green": (0.0, 1.0, 0.0),
            "Red": (1.0, 0.1, 0.1),
            "Blue": (0.2, 0.45, 1.0),
            "Yellow": (1.0, 0.9, 0.1),
            "Magenta": (1.0, 0.2, 1.0),
            "Cyan": (0.1, 1.0, 1.0),
        }

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        main_layout.addWidget(splitter)
        self.mask_splitter = splitter
        self.setStyleSheet(
            "QFrame#mask_left_panel, QFrame#mask_right_panel { border: 1px solid #b9b9b9; }"
        )

        left_panel = QFrame(self)
        self.mask_left_panel = left_panel
        left_panel.setObjectName("mask_left_panel")
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setFrameShadow(QFrame.Plain)
        left_panel.setMinimumWidth(180)
        left_panel.setMaximumWidth(16777215)
        left_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        left_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        right_panel = QFrame(self)
        self.mask_right_panel = right_panel
        right_panel.setObjectName("mask_right_panel")
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setFrameShadow(QFrame.Plain)
        right_panel.setMinimumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        controls_row_0 = QHBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_source_image)
        controls_row_0.addWidget(self.load_image_button)
        controls_row_0.addStretch()
        left_layout.addLayout(controls_row_0)

        controls_row_1 = QHBoxLayout()
        controls_row_1.addWidget(QLabel("Threshold:"))
        controls_row_1.addStretch()
        left_layout.addLayout(controls_row_1)

        controls_row_2 = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(500)
        self.threshold_slider.setMaximumWidth(260)
        self.threshold_slider.valueChanged.connect(self.regenerate_mask_from_threshold)
        controls_row_2.addWidget(self.threshold_slider)
        controls_row_2.addStretch()
        left_layout.addLayout(controls_row_2)

        controls_row_2_value = QHBoxLayout()
        self.threshold_value_label = QLabel("0.0000")
        self.threshold_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.threshold_value_label.setFixedWidth(90)
        controls_row_2_value.addWidget(self.threshold_value_label)
        controls_row_2_value.addStretch()
        left_layout.addLayout(controls_row_2_value)

        controls_row_3 = QHBoxLayout()
        self.clear_to_threshold_button = QPushButton("Reset")
        self.clear_to_threshold_button.clicked.connect(self.regenerate_mask_from_threshold)
        controls_row_3.addWidget(self.clear_to_threshold_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_last_brush_action)
        controls_row_3.addWidget(self.undo_button)
        controls_row_3.addStretch()
        left_layout.addLayout(controls_row_3)

        controls_row_4 = QHBoxLayout()
        self.invert_threshold_checkbox = QCheckBox("Invert")
        self.invert_threshold_checkbox.toggled.connect(self.regenerate_mask_from_threshold)
        controls_row_4.addWidget(self.invert_threshold_checkbox)
        controls_row_4.addStretch()
        left_layout.addLayout(controls_row_4)

        controls_row_5 = QHBoxLayout()
        controls_row_5.addWidget(QLabel("Highlight:"))
        self.overlay_color_combo = QComboBox()
        self.overlay_color_combo.addItems(list(self._overlay_colors.keys()))
        self.overlay_color_combo.setCurrentText("Green")
        self.overlay_color_combo.setMaximumWidth(160)
        self.overlay_color_combo.currentTextChanged.connect(lambda _: self._draw_mask_preview())
        controls_row_5.addWidget(self.overlay_color_combo)
        controls_row_5.addStretch()
        left_layout.addLayout(controls_row_5)

        controls_row_6 = QHBoxLayout()
        controls_row_6.addWidget(QLabel("Brush size:"))
        controls_row_6.addStretch()
        left_layout.addLayout(controls_row_6)

        controls_row_7 = QHBoxLayout()
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 40)
        self.brush_size_slider.setValue(8)
        self.brush_size_slider.setMaximumWidth(260)
        self.brush_size_slider.valueChanged.connect(self._refresh_brush_outline_geometry)
        controls_row_7.addWidget(self.brush_size_slider)
        controls_row_7.addStretch()
        left_layout.addLayout(controls_row_7)

        controls_row_8 = QHBoxLayout()
        controls_row_8.addWidget(QLabel("Shape:"))
        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItems(["Circle", "Square"])
        self.brush_shape_combo.setMaximumWidth(160)
        self.brush_shape_combo.currentTextChanged.connect(self._refresh_brush_outline_geometry)
        controls_row_8.addWidget(self.brush_shape_combo)
        controls_row_8.addStretch()
        left_layout.addLayout(controls_row_8)

        controls_row_9 = QHBoxLayout()
        self.keep_radio = QRadioButton("Brush")
        self.exclude_radio = QRadioButton("Erase")
        self.keep_radio.setMinimumWidth(90)
        self.exclude_radio.setMinimumWidth(90)
        self._brush_mode_updating = False
        self.keep_radio.setAutoExclusive(False)
        self.exclude_radio.setAutoExclusive(False)
        self.brush_mode_group = QButtonGroup(self)
        self.brush_mode_group.setExclusive(False)
        self.brush_mode_group.addButton(self.keep_radio)
        self.brush_mode_group.addButton(self.exclude_radio)
        self.keep_radio.toggled.connect(self._on_keep_mode_toggled)
        self.exclude_radio.toggled.connect(self._on_erase_mode_toggled)
        # Explicitly start with no active paint mode.
        self.keep_radio.setChecked(False)
        self.exclude_radio.setChecked(False)
        controls_row_9.addWidget(self.keep_radio)
        controls_row_9.addWidget(self.exclude_radio)
        controls_row_9.addStretch()
        left_layout.addLayout(controls_row_9)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.03)
        self.canvas.axes.set_title("Load an image to generate a mask", fontsize=10)
        self.canvas.axes.set_axis_off()
        right_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
        right_layout.addWidget(self.toolbar)

        controls_row_10 = QHBoxLayout()
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        controls_row_10.addWidget(self.save_mask_button)
        controls_row_10.addStretch()
        left_layout.addLayout(controls_row_10)

        self.status_label = QLabel(
            "Mask values: 1=keep area, 0=excluded area. Select Brush or Erase to edit. "
            "Close this window to apply the current mask."
        )
        right_layout.addWidget(self.status_label)
        left_layout.addStretch()
        left_target = max(220, self.mask_left_panel.sizeHint().width() + 24)
        max_left = int(width * 0.45)
        left_target = min(left_target, max_left if max_left > 220 else left_target)
        splitter.setSizes([left_target, max(400, width - left_target)])

        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)

    def _to_grayscale(self, arr):
        """Convert input image array to 2D float grayscale."""
        data = np.asarray(arr)
        if data.ndim == 2:
            return data.astype(np.float32)
        if data.ndim == 3:
            if data.shape[-1] >= 3:
                rgb = data[..., :3].astype(np.float32)
                return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
            return np.mean(data.astype(np.float32), axis=-1).astype(np.float32)
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Unsupported image dimensions: {np.asarray(arr).shape}")
        return data.astype(np.float32)

    def load_source_image(self):
        """Load source image from FITS/TIFF and initialize threshold mask."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Image",
            "",
            "Image Files (*.fits *.fit *.fts *.tif *.tiff);;FITS Files (*.fits *.fit *.fts);;TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".fits", ".fit", ".fts"):
                with fits.open(file_path) as hdul:
                    if hdul[0].data is None:
                        raise ValueError("No image data found in primary FITS HDU.")
                    raw = hdul[0].data
            elif ext in (".tif", ".tiff"):
                raw = np.array(Image.open(file_path))
            else:
                raise ValueError("Unsupported file format. Use FITS or TIFF.")

            self.source_image = self._to_grayscale(raw)
            finite = np.isfinite(self.source_image)
            if not np.any(finite):
                raise ValueError("Image contains no finite values.")

            self._threshold_min = float(np.nanmin(self.source_image[finite]))
            self._threshold_max = float(np.nanmax(self.source_image[finite]))
            self.regenerate_mask_from_threshold(preserve_view=False)
            self.status_label.setText(
                f"Loaded image: {os.path.basename(file_path)} | shape={self.source_image.shape}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load Image Failed", f"Could not load image:\n{e}")

    def regenerate_mask_from_threshold(self, *_args, preserve_view=True):
        """Rebuild mask from current threshold controls."""
        if self.source_image is None:
            return
        if self._threshold_min is None or self._threshold_max is None:
            return

        slider_value = self.threshold_slider.value() / 1000.0
        threshold = self._threshold_min + slider_value * (self._threshold_max - self._threshold_min)
        self.threshold_value_label.setText(self._format_threshold_value(threshold))

        finite = np.isfinite(self.source_image)
        if self.invert_threshold_checkbox.isChecked():
            mask = (self.source_image <= threshold) & finite
        else:
            mask = (self.source_image >= threshold) & finite

        self.current_mask = mask.astype(np.uint8)
        # Threshold-based mask rebuilds are intentionally non-undoable.
        self._clear_undo_history()
        self._draw_mask_preview(preserve_view=preserve_view)

    @staticmethod
    def _format_threshold_value(value):
        """Format threshold values compactly for UI readability."""
        abs_value = abs(float(value))
        if abs_value >= 1e4 or (0 < abs_value < 1e-2):
            return f"{value:.3g}"
        return f"{value:.2f}"

    def _draw_mask_preview(self, preserve_view=True):
        """Draw source image with translucent keep-area overlay."""
        ax = self.canvas.axes
        saved_xlim = None
        saved_ylim = None
        if preserve_view and self.source_image is not None and ax.has_data():
            saved_xlim = ax.get_xlim()
            saved_ylim = ax.get_ylim()
        ax.clear()
        self._brush_outline = None

        if self.source_image is None:
            ax.set_title("Load an image to generate a mask", fontsize=10)
            ax.set_position([0.01, 0.05, 0.98, 0.90])
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        finite = np.isfinite(self.source_image)
        if np.any(finite):
            vmin = float(np.nanpercentile(self.source_image[finite], 2))
            vmax = float(np.nanpercentile(self.source_image[finite], 98))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        ax.imshow(self.source_image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if self.current_mask is not None:
            color_name = self.overlay_color_combo.currentText()
            r, g, b = self._overlay_colors.get(color_name, self._overlay_colors["Green"])
            overlay = np.zeros((*self.current_mask.shape, 4), dtype=np.float32)
            overlay[..., 0] = r
            overlay[..., 1] = g
            overlay[..., 2] = b
            overlay[..., 3] = 0.45 * (self.current_mask > 0)
            ax.imshow(overlay, interpolation="nearest")

        if saved_xlim is not None and saved_ylim is not None:
            ax.set_xlim(saved_xlim)
            ax.set_ylim(saved_ylim)
        ax.set_position([0.01, 0.05, 0.98, 0.90])
        ax.set_title("Mask preview (highlight = kept / value 1)", fontsize=10)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _clear_undo_history(self):
        self._undo_stack.clear()
        self._update_undo_button_state()

    def _update_undo_button_state(self):
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(bool(self._undo_stack))

    def _push_undo_snapshot(self):
        if self.current_mask is None:
            return
        self._undo_stack.append(self.current_mask.copy())
        self._update_undo_button_state()

    def undo_last_brush_action(self):
        """Undo the latest brush/erase stroke."""
        if not self._undo_stack:
            return
        self.current_mask = self._undo_stack.pop()
        self._update_undo_button_state()
        self._draw_mask_preview(preserve_view=True)
        if self._last_cursor is not None:
            self._update_brush_outline(*self._last_cursor)

    def _on_keep_mode_toggled(self, checked):
        """Ensure Brush/Erase behave as mutually exclusive toggles with optional none state."""
        if self._brush_mode_updating:
            return
        self._brush_mode_updating = True
        if checked:
            self.exclude_radio.setChecked(False)
        self._brush_mode_updating = False
        self._refresh_brush_outline_style()

    def _on_erase_mode_toggled(self, checked):
        """Ensure Brush/Erase behave as mutually exclusive toggles with optional none state."""
        if self._brush_mode_updating:
            return
        self._brush_mode_updating = True
        if checked:
            self.keep_radio.setChecked(False)
        self._brush_mode_updating = False
        self._refresh_brush_outline_style()

    def _remove_brush_outline(self):
        if self._brush_outline is not None:
            try:
                self._brush_outline.remove()
            except Exception:
                pass
            self._brush_outline = None

    def _brush_outline_color(self):
        if self.keep_radio.isChecked():
            return "#00ff66"
        if self.exclude_radio.isChecked():
            return "#ff4d4d"
        return "#f0f0f0"

    def _refresh_brush_outline_style(self):
        if self._brush_outline is not None:
            self._brush_outline.set_edgecolor(self._brush_outline_color())
            self.canvas.draw_idle()

    def _refresh_brush_outline_geometry(self, *_args):
        if self._last_cursor is not None:
            self._update_brush_outline(*self._last_cursor)

    def _update_brush_outline(self, xdata, ydata):
        if self.source_image is None or xdata is None or ydata is None:
            self._remove_brush_outline()
            self._last_cursor = None
            self.canvas.draw_idle()
            return
        if self._is_navigation_mode_active():
            self._remove_brush_outline()
            self.canvas.draw_idle()
            return

        x = int(round(xdata))
        y = int(round(ydata))
        h, w = self.source_image.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            self._remove_brush_outline()
            self._last_cursor = None
            self.canvas.draw_idle()
            return

        self._last_cursor = (x, y)
        size = max(1, int(self.brush_size_slider.value()))
        color = self._brush_outline_color()
        ax = self.canvas.axes

        shape = self.brush_shape_combo.currentText()
        if shape == "Square":
            x0 = x - size - 0.5
            y0 = y - size - 0.5
            width = 2 * size + 1
            if isinstance(self._brush_outline, Rectangle):
                self._brush_outline.set_xy((x0, y0))
                self._brush_outline.set_width(width)
                self._brush_outline.set_height(width)
                self._brush_outline.set_edgecolor(color)
            else:
                self._remove_brush_outline()
                self._brush_outline = Rectangle(
                    (x0, y0),
                    width,
                    width,
                    fill=False,
                    linewidth=1.2,
                    edgecolor=color,
                    linestyle="-",
                )
                ax.add_patch(self._brush_outline)
        else:
            radius = size
            if isinstance(self._brush_outline, Circle):
                self._brush_outline.center = (x, y)
                self._brush_outline.set_radius(radius)
                self._brush_outline.set_edgecolor(color)
            else:
                self._remove_brush_outline()
                self._brush_outline = Circle(
                    (x, y),
                    radius=radius,
                    fill=False,
                    linewidth=1.2,
                    edgecolor=color,
                    linestyle="-",
                )
                ax.add_patch(self._brush_outline)

        self.canvas.draw_idle()

    def _apply_brush(self, xdata, ydata):
        if self.current_mask is None or xdata is None or ydata is None:
            return

        if self.keep_radio.isChecked():
            brush_value = 1
        elif self.exclude_radio.isChecked():
            brush_value = 0
        else:
            self.status_label.setText("Select Brush or Erase to edit mask.")
            return

        h, w = self.current_mask.shape
        x = int(round(xdata))
        y = int(round(ydata))
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        size = max(1, int(self.brush_size_slider.value()))
        if self.brush_shape_combo.currentText() == "Square":
            half = size
            x_min = max(0, x - half)
            x_max = min(w, x + half + 1)
            y_min = max(0, y - half)
            y_max = min(h, y + half + 1)
            self.current_mask[y_min:y_max, x_min:x_max] = brush_value
        else:
            radius = size
            yy, xx = np.ogrid[:h, :w]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            self.current_mask[circle] = brush_value
        self._draw_mask_preview()
        self._update_brush_outline(x, y)

    def _on_canvas_press(self, event):
        if event.button != 1 or event.inaxes != self.canvas.axes:
            return
        if self._is_navigation_mode_active():
            self._painting = False
            return
        if self.current_mask is None:
            self._painting = False
            return
        if not (self.keep_radio.isChecked() or self.exclude_radio.isChecked()):
            self._painting = False
            self.status_label.setText("Select Brush or Erase to edit mask.")
            return
        # Save one snapshot per stroke for incremental undo.
        self._push_undo_snapshot()
        self._update_brush_outline(event.xdata, event.ydata)
        self._painting = True
        self._apply_brush(event.xdata, event.ydata)

    def _on_canvas_motion(self, event):
        if event.inaxes != self.canvas.axes:
            self._remove_brush_outline()
            self.canvas.draw_idle()
            return
        self._update_brush_outline(event.xdata, event.ydata)
        if not self._painting or self._is_navigation_mode_active():
            return
        self._apply_brush(event.xdata, event.ydata)

    def _on_canvas_release(self, event):
        if event.button == 1:
            self._painting = False

    def _on_canvas_scroll(self, event):
        """Zoom in/out around cursor with mouse wheel."""
        if self.source_image is None or event.inaxes != self.canvas.axes:
            return
        if self._is_navigation_mode_active():
            return
        if event.button not in ("up", "down"):
            return

        ax = self.canvas.axes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2.0
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2.0

        scale_factor = 0.9 if event.button == "up" else 1.1
        new_width = (x_max - x_min) * scale_factor
        new_height = (y_max - y_min) * scale_factor

        if x_max != x_min:
            rel_x = (x_center - x_min) / (x_max - x_min)
        else:
            rel_x = 0.5
        if y_max != y_min:
            rel_y = (y_center - y_min) / (y_max - y_min)
        else:
            rel_y = 0.5

        ax.set_xlim([x_center - new_width * rel_x, x_center + new_width * (1 - rel_x)])
        ax.set_ylim([y_center - new_height * rel_y, y_center + new_height * (1 - rel_y)])
        self.canvas.draw_idle()

    def _is_navigation_mode_active(self):
        """Return True when matplotlib toolbar pan/zoom mode is active."""
        mode = str(getattr(self.toolbar, "mode", "")).strip().lower()
        if mode and mode not in ("none",):
            return True
        active = getattr(self.toolbar, "_active", None)
        return bool(str(active).strip()) if active is not None else False

    def _emit_mask_ready(self):
        """Emit the current mask to preprocessing if available."""
        if self.current_mask is not None:
            self.mask_ready.emit(self.current_mask.astype(np.float32))

    def closeEvent(self, event):
        """Apply mask to filtering when the dialog is closed."""
        self._emit_mask_ready()
        super().closeEvent(event)

    def save_mask(self):
        """Save current mask as FITS or TIFF."""
        if self.current_mask is None:
            QMessageBox.information(self, "No Mask", "Generate or edit a mask first.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "FITS Files (*.fits);;TIFF Files (*.tif *.tiff)",
        )
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if not ext:
                if "FITS" in selected_filter:
                    file_path += ".fits"
                    ext = ".fits"
                else:
                    file_path += ".tif"
                    ext = ".tif"

            if ext in (".fits", ".fit", ".fts"):
                fits.writeto(file_path, self.current_mask.astype(np.uint8), overwrite=True)
            elif ext in (".tif", ".tiff"):
                Image.fromarray((self.current_mask.astype(np.uint8) * 255)).save(file_path)
            else:
                raise ValueError("Unsupported save format. Use FITS or TIFF.")

            QMessageBox.information(self, "Mask Saved", f"Mask saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save mask:\n{e}")

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
        # Keep parameter map fonts independent from global matplotlib rcParams.
        base_font = max(8, self.font().pointSize())
        self._plot_font_size = min(base_font + 1, 12)

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
        self.canvas.mpl_connect("scroll_event", self.zoom)

        # Keep a reference to line profile dialogs to prevent garbage collection
        self.line_profile_dialogs = []

    def _apply_parameter_plot_style(self, x_label, y_label):
        """Apply consistent local font sizing for parameter map canvas."""
        font_size = self._plot_font_size
        self.canvas.axes.set_title(self.parameter_name, fontsize=font_size + 1)
        self.canvas.axes.set_xlabel(x_label, fontsize=font_size)
        self.canvas.axes.set_ylabel(y_label, fontsize=font_size)
        self.canvas.axes.tick_params(labelsize=font_size)
        if self.colorbar is not None:
            self.colorbar.ax.tick_params(labelsize=font_size)
        
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
    
        self._apply_parameter_plot_style(x_label, y_label)
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

            x_label = self.canvas.axes.get_xlabel()
            y_label = self.canvas.axes.get_ylabel()
            self._apply_parameter_plot_style(x_label, y_label)

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
    

            unit_factor = 0.055 if self.unit_in_mm else 1.0
            x_edges = self.calculate_edges(self.X_unique * unit_factor)
            y_edges = self.calculate_edges(self.Y_unique * unit_factor)
            rect_x = x_edges[ixmin]
            rect_y = y_edges[iymin]
            rect_width = x_edges[ixmax] - x_edges[ixmin]
            rect_height = y_edges[iymax] - y_edges[iymin]


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
        """Zoom in/out around cursor using mouse wheel."""
        if event.inaxes != self.canvas.axes:
            return
        if event.button not in ("up", "down"):
            return
        # Avoid conflicts with matplotlib toolbar pan/zoom mode.
        mode = str(getattr(self.toolbar, "mode", "")).strip().lower()
        if mode and mode not in ("none",):
            return
        active = getattr(self.toolbar, "_active", None)
        if active is not None and str(active).strip():
            return
        # Avoid zooming while a drag/select interaction is in progress.
        if self._press_event is not None:
            return

        if self.unit_in_mm:
            x_factor = 0.055
            y_factor = 0.055
        else:
            x_factor = 1.0
            y_factor = 1.0

        x_edges = self.calculate_edges(self.X_unique * x_factor)
        y_edges = self.calculate_edges(self.Y_unique * y_factor)
        x_bounds = (float(x_edges[0]), float(x_edges[-1]))
        y_bounds = (float(y_edges[0]), float(y_edges[-1]))

        ax = self.canvas.axes
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        y_center = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        scale = 0.9 if event.button == "up" else 1.1

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
    "MaskGeneratorDialog",
    "FitVisualizationDialog",
    "AdjustmentsDialog",
    "ParameterPlotDialog",
    "LineProfileDialog",
    "OpenBeamPlotDialog",
]


