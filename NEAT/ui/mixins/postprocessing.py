"""Post-processing and metadata functionality."""

import os

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..dialogs import ParameterPlotDialog
from ...workers.batch import load_image_file


class PostProcessingMixin:
    def setup_PostProcessingTab(self):
        layout = QGridLayout()  # Build layout before attaching to the tab

        # Add widgets to specific grid positions
        self.load_csv_button = QPushButton("Load CSV File")
        self.load_csv_button.clicked.connect(self.load_csv_file)
        layout.addWidget(self.load_csv_button, 0, 0)  # Row 0, Column 0

        self.load_image_button = QPushButton("Load Image Result")
        self.load_image_button.clicked.connect(self.load_image_results)
        layout.addWidget(self.load_image_button, 0, 1)  # Row 0, Column 1
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
        self.parameter_layout = QGridLayout()
        self.parameter_widget.setLayout(self.parameter_layout)
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
        self.current_image_metadata = {}

    def setup_about_tab(self):
        layout = QVBoxLayout(self.tab3)

        about_text = """
        <h2>NEAT Neutron Bragg Edge Analysis Toolkit v4.6.0</h2>

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
        UI Font:     Shift + Up / Shift + Down<br>
        Canvas Font: Ctrl + Up / Ctrl + Down</p>

        """

        about_label = QLabel(about_text)
        about_label.setAlignment(Qt.AlignTop)
        about_label.setWordWrap(True)
        about_label.setTextFormat(Qt.RichText)


        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(about_label)

        layout.addWidget(scroll)

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

    def load_image_results(self):
        """
        Load one or more FITS/TIFF image results and display each in its own window.
        """
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Image Result",
            "",
            "Image Files (*.fits *.fit *.tiff *.tif);;All Files (*)",
        )
        if not files:
            return

        for path in files:
            try:
                data = load_image_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", f"Failed to load '{path}': {exc}")
                continue

            # Reduce to 2D if possible
            arr = np.asarray(data)
            if arr.ndim > 2:
                arr = arr[0]
            if arr.ndim != 2:
                QMessageBox.warning(self, "Unsupported Image", f"Image '{path}' is not 2D; skipping.")
                continue

            # Build coordinate grids
            h, w = arr.shape
            X_unique = np.arange(w)
            Y_unique = np.arange(h)
            Z = arr

            # Minimal metadata
            meta = {
                "Source": os.path.basename(path),
                "Shape": f"{h}x{w}",
            }

            dlg = ParameterPlotDialog(
                X_unique,
                Y_unique,
                Z,
                parameter_name=os.path.basename(path),
                metadata=meta,
                csv_filename=path,
                work_directory=getattr(self, "work_directory", None),
                parent=self,
            )
            dlg.show()

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

        # Layout already assigned when tab was created; just refresh scroll widget
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
