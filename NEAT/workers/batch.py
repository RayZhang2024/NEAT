"""Worker threads for batch fitting tasks."""

import datetime
import gc
import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None
    try:
        from PIL import Image
    except ImportError:
        Image = None


def load_image_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".fits", ".fit"):
        return fits.getdata(file_path)
    if imageio is not None:
        arr = imageio.imread(file_path)
        # TIFFs are read with inverted vertical axis compared to FITS; flip to match FITS orientation.
        if ext in (".tiff", ".tif"):
            arr = np.flipud(arr)
        return arr
    if Image is not None:
        with Image.open(file_path) as img:
            arr = np.array(img)
        if ext in (".tiff", ".tif"):
            arr = np.flipud(arr)
        return arr
    raise ImportError("Neither imageio nor Pillow is available to read TIFF files.")
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, least_squares

from ..core import calculate_x_hkl_general, fitting_function_3

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

    @staticmethod
    def _safe_float(widget_or_value, default=""):
        """
        Safely convert a widget or raw value to float, returning default on failure.
        """
        try:
            if widget_or_value is None:
                return default
            value = widget_or_value.text() if hasattr(widget_or_value, "text") else widget_or_value
            return float(value)
        except (TypeError, ValueError):
            return default

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
            ("min_wavelength", self._safe_float(getattr(self.parent, "min_wavelength_input", None))),
            ("max_wavelength", self._safe_float(getattr(self.parent, "max_wavelength_input", None))),
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

    @staticmethod
    def _safe_float(widget_or_value, default=""):
        """
        Safely convert a widget or raw value to float, returning default on failure.
        """
        try:
            if widget_or_value is None:
                return default
            value = widget_or_value.text() if hasattr(widget_or_value, "text") else widget_or_value
            return float(value)
        except (TypeError, ValueError):
            return default

    def run(self):
        box_counter = 0
        params_initialized = False
        failure_reported = 0

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
                        fix_eta=self.fix_eta,
                        max_nfev=300,
                        curve_fit_maxfev=300
                    )
                except Exception as e:
                    error_msg = f"Unexpected error during fitting: {e}"
                    result_dict = None

                # Surface the first few failures so users see why no fits succeed.
                if (error_msg or not result_dict or not result_dict.get("success", False)) and failure_reported < 5:
                    detail = error_msg or result_dict.get("message") if result_dict else ""
                    self.message.emit(
                        f"Fit failed at box centered ({center_col},{center_row}): {detail or 'unknown reason'}"
                    )
                    failure_reported += 1

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
            # Empty payload -> UI treats as error/empty result
            self.finished.emit("")
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
        def d_spacing(h, k, l, structure, lat):
            x_vals = calculate_x_hkl_general(structure, lat, [(h, k, l)])
            if not x_vals or np.isnan(x_vals[0]):
                return float("nan")
            return x_vals[0] / 2.0  # lambda = 2d -> d = lambda/2

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
            ("min_wavelength", self._safe_float(getattr(self.parent, "min_wavelength_input", None))),
            ("max_wavelength", self._safe_float(getattr(self.parent, "max_wavelength_input", None))),
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
        self._stop_requested = False

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
            fits_files = [
                f for f in os.listdir(self.folder_path)
                if f.lower().endswith(('.fits', '.fit', '.tiff', '.tif'))
            ]
            if not fits_files:
                self.message.emit(f"No FITS files found in folder: \\{short_path}")
                return

            # Sort files to ensure suffixes are aligned
            fits_files.sort()

            total_files = len(fits_files)
            run_dict = {}

            for idx, file in enumerate(fits_files):
                if self._stop_requested:
                    self.message.emit("Image loading cancelled by user.")
                    break
                # Extract suffix as per naming convention
                basename = os.path.splitext(file)[0]
                parts = basename.split('_')
                if len(parts) < 2:
                    self.message.emit(f"Filename '{file}' does not contain an underscore-separated suffix. Skipping.")
                    continue
                suffix = parts[-1]

                # Validate suffix: must be a numeric string (1–10 digits)
                if not suffix.isdigit() or not (1 <= len(suffix) <= 10):
                    self.message.emit(f"Suffix '{suffix}' in filename '{file}' is not a valid numeric identifier. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue
                # suffix_int = int(suffix)
                # if suffix_int < 0 or suffix_int > 2924:
                #     self.message.emit(f"Suffix '{suffix}' in filename '{file}' is outside the allowed range _00000 to _02924. Skipping.")
                #     continue

                # Construct the full path
                file_path = os.path.join(self.folder_path, file)
                # Load image data (FITS or TIFF)
                try:
                    image_data = load_image_file(file_path)
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
                self.message.emit(f"No FITS/TIFF images with numeric suffixes were found in folder: {self.folder_path}")
            else:
                self.message.emit(f"Successfully loaded {len(run_dict)} images from \\{short_path}")
                # Emit both folder_path and run_dict
                self.run_loaded.emit(self.folder_path, run_dict)

        except Exception as e:
            self.message.emit(f"Error loading images from \\{short_path}: {e}")
        finally:
            # Explicitly call garbage collector to ensure all file handles are released
            gc.collect()
            self.finished.emit()

    def stop(self):
        self._stop_requested = True

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

            fits_files = [
                f for f in os.listdir(self.folder_path)
                if f.lower().endswith(('.fits', '.fit', '.tiff', '.tif'))
            ]
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
                # Validate suffix: must be a numeric identifier (1–10 digits)
                if not suffix.isdigit() or not (1 <= len(suffix) <= 10):
                    self.message.emit(f"Suffix '{suffix}' in filename '{file}' is not a valid numeric identifier. Skipping.")
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    continue

                file_path = os.path.join(self.folder_path, file)
                try:
                    image_data = load_image_file(file_path)
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

__all__ = [
    "BatchFitEdgesWorker",
    "BatchFitWorker",
    "ImageLoadWorker",
    "OpenBeamLoadWorker",
]
