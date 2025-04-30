import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QScrollBar,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QCheckBox, QSpinBox,
    QSizePolicy, QMessageBox
)
import tifffile as tiff
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import shutil
import re
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import circle_perimeter
import cv2
import glob
class TimeLapseViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Time Lapse TIFF Viewer")
        self.setGeometry(100, 100, 1000, 600)

        # Main layout: horizontal split
        self.mainLayout = QHBoxLayout()

        # Left panel: image and buttons
        leftWidget = QWidget()
        leftLayout = QVBoxLayout(leftWidget)

        self.imageLabel = QLabel(self)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setScaledContents(True)
        leftLayout.addWidget(self.imageLabel)

        # Create a horizontal layout for the slider and frame number
        slider_layout = QHBoxLayout()
        
        # Add the scroll bar to the layout with stretch factor of 5
        self.scrollBar = QScrollBar(self)
        self.scrollBar.setOrientation(Qt.Horizontal)
        self.scrollBar.valueChanged.connect(self.scrollBarMoved)
        slider_layout.addWidget(self.scrollBar, stretch=5)
        
        # Create a label for the frame number
        self.frameNumberLabel = QLabel("Frame: 1")
        # Set a fixed width for the label (approximately 1/5 of the slider width)
        self.frameNumberLabel.setFixedWidth(100)  # Adjust this value as needed
        slider_layout.addWidget(self.frameNumberLabel, stretch=1)
        
        # Add the slider layout to the main layout
        leftLayout.addLayout(slider_layout)

        self.loadButton = QPushButton("Load Time-Lapse TIFF", self)
        self.loadButton.clicked.connect(self.loadTiff)
        leftLayout.addWidget(self.loadButton)

        self.selectOutputButton = QPushButton("Select Output Folder", self)
        self.selectOutputButton.clicked.connect(self.selectOutputFolder)
        leftLayout.addWidget(self.selectOutputButton)

        self.saveButton = QPushButton("Save Current Time-Stack", self)
        self.saveButton.clicked.connect(self.saveCurrentImage)
        leftLayout.addWidget(self.saveButton)

        self.nextButton = QPushButton("Next Time-Stack", self)
        self.nextButton.clicked.connect(self.nextTimeStack)
        leftLayout.addWidget(self.nextButton)

        # Right panel: RNA Tracks
        rightWidget = QWidget()
        rightWidget.setFixedWidth(150)
        rightLayout = QVBoxLayout(rightWidget)
        rightLayout.addWidget(QLabel("RNA Tracks"))

        # Scroll area for checkboxes
        self.trackCheckBoxArea = QScrollArea()
        self.trackCheckBoxWidget = QWidget()
        self.trackCheckBoxLayout = QVBoxLayout(self.trackCheckBoxWidget)
        self.trackCheckBoxArea.setWidget(self.trackCheckBoxWidget)
        self.trackCheckBoxArea.setWidgetResizable(True)
        rightLayout.addWidget(self.trackCheckBoxArea)

        # Spin boxes for frame range
        self.minFrameSpinBox = QSpinBox()
        self.minFrameSpinBox.setMinimum(1)
        self.minFrameSpinBox.setValue(1)
        self.minFrameSpinBox.valueChanged.connect(self.showImage)
        rightLayout.addWidget(QLabel("Min Frame"))
        rightLayout.addWidget(self.minFrameSpinBox)

        self.maxFrameSpinBox = QSpinBox()
        self.maxFrameSpinBox.setMinimum(1)
        self.maxFrameSpinBox.setValue(1)
        self.maxFrameSpinBox.valueChanged.connect(self.showImage)
        rightLayout.addWidget(QLabel("Max Frame"))
        rightLayout.addWidget(self.maxFrameSpinBox)

        # Combine layouts
        self.mainLayout.addWidget(leftWidget, stretch=4)
        self.mainLayout.addWidget(rightWidget, stretch=1)

        centralWidget = QWidget(self)
        centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(centralWidget)

        # Initialize variables
        self.timeStackFiles = []
        self.images = []
        self.currentFileIndex = 0
        self.currentIndex = 0
        self.outputFolder = ''
        self.trackCheckBoxes = {}
        self.spot_data = {}  # Store spot coordinates for overlay
        self.unannotated_image = None
        self.current_channel = None  # Add this line to store the current channel

    def extract_identifiers(self, filepath):
        """Extract col, row, field, cell, and channel from file name."""
        match = re.search(r'col(\d+)_row(\d+)_field(\d+)_cell(\d+)_Ch(\d+)', os.path.basename(filepath), re.IGNORECASE)
        if match:
            col, row, field, cell, ch = map(int, match.groups())
            return col, row, field, cell, ch
        else:
            raise ValueError(f"Could not extract identifiers from file name: {os.path.basename(filepath)}")

    def COORDINATES_TO_CIRCLE(self, coordinates, ImageForSpots, circ_radius=5):
        """Draw circles at specified coordinates on an image."""
        circles = np.zeros((ImageForSpots.shape), dtype=np.uint8)
        if coordinates.any():
            for center_y, center_x in zip(coordinates[:, 1], coordinates[:, 0]):
                circy, circx = circle_perimeter(int(center_y), int(center_x), circ_radius, shape=ImageForSpots.shape)
                circles[circy, circx] = 255
        return circles

    def loadTiff(self):
        filepaths, _ = QFileDialog.getOpenFileNames(self, "Open Time-Lapse TIFF", "", "TIFF Files (*.tiff *.tif)")
        if filepaths:
            if "annotated_spot_image_patches" not in filepaths[0] or "spot_image_patches" not in filepaths[0]:
                QMessageBox.critical(self, "Error", "Please select time-lapses from unannotated_spot_image_patches or spot_image_patches folders")
                return
            
            self.timeStackFiles = filepaths
            self.currentFileIndex = 0
            self.loadCurrentTimeStack()

    def loadCurrentTimeStack(self):
        if self.currentFileIndex >= len(self.timeStackFiles):
            return

        filepath = self.timeStackFiles[self.currentFileIndex]
        
        # Extract identifiers
        try:
            col, row, fov, cell, ch = self.extract_identifiers(filepath)
            self.current_channel = ch  # Store the channel number
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Load unannotated image
        source_dir = os.path.dirname(filepath)
        parent_dir = Path(source_dir).parent
        unannotated_dir = os.path.join(parent_dir, 'unannotated_spot_image_patches')
        unannotated_filename = f"raw_spot_img_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}.tif"
        unannotated_path = os.path.join(unannotated_dir, unannotated_filename)
        
        try:
            print(f"Loading image from: {unannotated_path}")  # Debugging line
            self.images = tiff.imread(unannotated_path)
            if self.images.size == 0:
                raise ValueError("Empty TIFF stack")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIFF file: {e}")
            return

        self.currentIndex = 0
        num_frames = len(self.images)
        self.scrollBar.setMaximum(num_frames - 1)
        self.minFrameSpinBox.setMaximum(num_frames)
        self.minFrameSpinBox.setValue(1)
        self.maxFrameSpinBox.setMaximum(num_frames)
        self.maxFrameSpinBox.setValue(num_frames)

        # Load spot data from complete_tables
        spot_intensity_dir = os.path.join(parent_dir, 'spot_intensity_tables', 'complete_tables')
        spot_labels = []
        self.spot_data = {}
        pattern = f'spot_intensity_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}_spot(\d+).csv'
        try:
            spot_files = [f for f in os.listdir(spot_intensity_dir) if re.match(pattern, f)]
            for f in spot_files:
                match = re.search(pattern, f)
                if match:
                    spot_label = match.group(1)
                    spot_labels.append(spot_label)
                    spot_file = os.path.join(spot_intensity_dir, f)
                    df = pd.read_csv(spot_file)
                    
                    # Print column names for debugging
                    
                    # Try different possible column name patterns
                    x_col = f'ch{ch}_spot_no_{spot_label}_x'
                    y_col = f'ch{ch}_spot_no_{spot_label}_y'
                    
                    # If the exact pattern doesn't exist, try alternative patterns
                    if x_col not in df.columns:
                        x_col = f'ch{ch}_spot_{spot_label}_x'
                        y_col = f'ch{ch}_spot_{spot_label}_y'
                    
                    if x_col not in df.columns:
                        x_col = f'ch{ch}_spot{spot_label}_x'
                        y_col = f'ch{ch}_spot{spot_label}_y'
                    
                    if x_col not in df.columns:
                        print(f"Warning: Could not find x,y columns for spot {spot_label} in {f}")
                        continue
                    
                    if {'t', x_col, y_col}.issubset(df.columns):
                        self.spot_data[spot_label] = df[['t', x_col, y_col]]
                    else:
                        print(f"Warning: Missing required columns for spot {spot_label} in {f}")
        except Exception as e:
            print(f"Warning: Error processing spot files in {spot_intensity_dir}: {e}")

        if not spot_labels:
            print(f"Warning: No spot intensity tables found for col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}")

        spot_labels = sorted(set(spot_labels))

        # Clear existing checkboxes
        for i in reversed(range(self.trackCheckBoxLayout.count())):
            self.trackCheckBoxLayout.itemAt(i).widget().setParent(None)

        # Create new checkboxes
        self.trackCheckBoxes = {}
        for label in spot_labels:
            checkbox = QCheckBox(f"Spot {label}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.showImage)  # Update image when checkbox changes
            self.trackCheckBoxLayout.addWidget(checkbox)
            self.trackCheckBoxes[label] = checkbox

        self.showImage()

    def selectOutputFolder(self):
        self.outputFolder = QFileDialog.getExistingDirectory(self, "Select Output Folder")

    def saveCurrentImage(self):
        # Check if images and output folder are available
        if self.images.size == 0 or not self.outputFolder:
            QMessageBox.warning(self, "Warning", "No image loaded or output folder not selected")
            return

        # Extract identifiers from the current file
        originalPath = self.timeStackFiles[self.currentFileIndex]
        col, row, fov, cell, ch = self.extract_identifiers(originalPath)

        # Get selected frame range (0-based)
        min_frame = self.minFrameSpinBox.value() - 1
        max_frame = self.maxFrameSpinBox.value() - 1

        # Validate frame range
        if min_frame > max_frame or min_frame < 0 or max_frame >= len(self.images):
            QMessageBox.warning(self, "Warning", "Invalid frame range selected")
            return

        # Get checked spots (tracks being saved)
        checked_spots = [label for label, cb in self.trackCheckBoxes.items() if cb.isChecked()]

        # Define output root and create subfolders
        output_root = Path(self.outputFolder)
        os.makedirs(output_root / 'annotated_spot_image_patches', exist_ok=True)
        os.makedirs(output_root / 'spot_image_patches', exist_ok=True)
        os.makedirs(output_root / 'unannotated_spot_image_patches', exist_ok=True)
        os.makedirs(output_root / 'spot_intensity_tables' / 'complete_tables', exist_ok=True)
        os.makedirs(output_root / 'spot_intensity_tables' / 'integrated_intensity_tables', exist_ok=True)
        os.makedirs(output_root / 'single_track_images', exist_ok=True)
        os.makedirs(output_root / 'single_track_tables', exist_ok=True)

        # Save unannotated stack with original dtype
        unannotated_stack = self.images[min_frame:max_frame + 1]
        unannotated_file = output_root / 'unannotated_spot_image_patches' / f'raw_spot_img_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}.tif'
        tiff.imwrite(unannotated_file, unannotated_stack)

        # Helper function to generate annotated stack
        def generate_annotated_stack(with_text):
            stack = []
            for frame in range(min_frame, max_frame + 1):
                image = self.images[frame].copy()
                # Normalize image to 0-255 for display, convert to RGB
                image = np.nan_to_num(image, nan=0.0)
                if np.min(image) < 0:
                    image -= np.min(image)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                pil_image = Image.fromarray(image).convert('RGB')
                draw = ImageDraw.Draw(pil_image)
                for spot_label in checked_spots:
                    if spot_label in self.spot_data:
                        df = self.spot_data[spot_label]
                        # Dynamically construct the column names based on channel and spot label
                        x_col = f'ch{self.current_channel}_spot_no_{spot_label}_x'
                        y_col = f'ch{self.current_channel}_spot_no_{spot_label}_y'
                        frame_data = df[df['t'] == frame]
                        for _, row in frame_data.iterrows():
                            x = row[x_col]  # Use the actual column name
                            y = row[y_col]  # Use the actual column name
                            # Draw yellow circle (radius 5)
                            draw.ellipse([(y - 5, x - 5), (y + 5, x + 5)], outline=(255, 255, 0))
                            if with_text:
                                # Use a smaller font for spot labels
                                try:
                                    font = ImageFont.truetype("arial.ttf", 8)  # 8-point font size
                                except OSError:
                                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)  # Fallback font
                                except OSError:
                                    font = ImageFont.load_default()  # Default font if others fail
                                    print("Warning: Using default font, size may not be adjustable")
                                draw.text((y + 6, x), spot_label, fill=(255, 255, 0), font=font)
                annotated_image = np.array(pil_image)
                stack.append(annotated_image)
            return np.stack(stack, axis=0)

        # Save annotated stack with circles and numbers
        annotated_stack = generate_annotated_stack(with_text=True)
        annotated_file = output_root / 'annotated_spot_image_patches' / f'annotated_spot_img_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}.tif'
        tiff.imwrite(annotated_file, annotated_stack)

        # Save spot stack with circles only
        spot_stack = generate_annotated_stack(with_text=False)
        spot_file = output_root / 'spot_image_patches' / f'spot_img_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}.tif'
        tiff.imwrite(spot_file, spot_stack)

        # Define source root for additional files
        source_root = Path(self.timeStackFiles[0]).parent.parent
        source_complete_tables = source_root / 'spot_intensity_tables' / 'complete_tables'
        source_integrated_tables = source_root / 'spot_intensity_tables' / 'integrated_intensity_tables'
        output_complete_tables = output_root / 'spot_intensity_tables' / 'complete_tables'
        output_integrated_tables = output_root / 'spot_intensity_tables' / 'integrated_intensity_tables'

        # Save filtered spot intensity tables
        for spot_label in checked_spots:
            # Filter and save CSV files in complete_tables
            csv_file = source_complete_tables / f'spot_intensity_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}_spot{spot_label}.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df_filtered = df[(df['t'] >= min_frame) & (df['t'] <= max_frame)]
                df_filtered = df_filtered.reset_index(drop=True)
                df_filtered.to_csv(output_complete_tables / csv_file.name, index=False)

            # Filter and save .trk files in integrated_intensity_tables
            trk_file = source_integrated_tables / f'integrated_intensity_for_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}_spot{spot_label}.trk'
            if trk_file.exists():
                df = pd.read_csv(trk_file, sep='\t', header=None)
                # Columns: [row, col, integrated_intensity, t, HMM_state], 't' is column 3
                df_filtered = df[(df[3] >= min_frame) & (df[3] <= max_frame)]
                df_filtered = df_filtered.reset_index(drop=True)
                df_filtered.to_csv(output_integrated_tables / trk_file.name, sep='\t', index=False, header=False)

        # Copy files from single_track_images and single_track_tables for checked spots
        source_single_track_images = source_root / 'single_track_images'
        source_single_track_tables = source_root / 'single_track_tables'
        for spot_label in checked_spots:
            # Copy single_track_images
            img_pattern = str(source_single_track_images / f'*_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}_spot{spot_label}.*')
            for file in glob.glob(img_pattern):
                shutil.copy2(file, output_root / 'single_track_images' / os.path.basename(file))

            # Copy single_track_tables
            table_pattern = str(source_single_track_tables / f'*_col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}_spot{spot_label}.*')
            for file in glob.glob(table_pattern):
                shutil.copy2(file, output_root / 'single_track_tables' / os.path.basename(file))

        # Copy unannotated_spot_image_patches for different channels
        source_unannotated = source_root / 'unannotated_spot_image_patches'
        unannotated_pattern = str(source_unannotated / f'raw_spot_img_for_col{col}_row{row}_field{fov}_cell{cell}_Ch*.tif')
        for file in glob.glob(unannotated_pattern):
            # Extract channel number from filename
            match = re.search(r'_Ch(\d+)', file)
            if match and int(match.group(1)) != ch:
                shutil.copy2(file, output_root / 'unannotated_spot_image_patches' / os.path.basename(file))

        # Notify user of success
        QMessageBox.information(self, "Success", f"Saved current image and files for col{col}_row{row}_field{fov}_cell{cell}_Ch{ch}")

    def nextTimeStack(self):
        if self.currentFileIndex < len(self.timeStackFiles) - 1:
            self.currentFileIndex += 1
            self.loadCurrentTimeStack()

    def showImage(self):
        if self.images.size == 0:
            return

        min_frame = self.minFrameSpinBox.value()
        max_frame = self.maxFrameSpinBox.value()
        if not (min_frame - 1 <= self.currentIndex <= max_frame - 1):
            return

        # Load and process the image
        image = self.images[self.currentIndex].copy()
        
        # Handle NaN values (optional, for robustness)
        image = np.nan_to_num(image, nan=0.0)
        
        # Handle negative values by shifting to positive range
        if np.min(image) < 0:
            image = image - np.min(image)
        
        # Normalize to 0-255 range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to QImage and QPixmap
        qImage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImage)

        # Collect all coordinates for selected spot_labels in the current frame
        all_coordinates = []
        spot_positions = {}  # To store positions for text drawing
        for spot_label, df in self.spot_data.items():
            if spot_label in self.trackCheckBoxes and self.trackCheckBoxes[spot_label].isChecked():
                frame_data = df[df['t'] == self.currentIndex]
                for _, row in frame_data.iterrows():
                    try:
                        # Get the actual column names from the DataFrame
                        x_col = [col for col in df.columns if f'ch{self.current_channel}_spot' in col and 'x' in col][0]
                        y_col = [col for col in df.columns if f'ch{self.current_channel}_spot' in col and 'y' in col][0]
                        
                        x = row[x_col]
                        y = row[y_col]
                        all_coordinates.append([y, x])
                        spot_positions[(y, x)] = spot_label
                    except (IndexError, KeyError) as e:
                        print(f"Warning: Could not process spot {spot_label} at frame {self.currentIndex}: {e}")
                        continue

        # If there are coordinates to process
        if all_coordinates:
            all_coordinates = np.array(all_coordinates)
            # Generate circle mask using COORDINATES_TO_CIRCLE
            circle_image = self.COORDINATES_TO_CIRCLE(all_coordinates, image, circ_radius=5)
            
            # Create an ARGB image for the overlay
            height, width = circle_image.shape
            argb_image = np.zeros((height, width), dtype=np.uint32)
            argb_image[circle_image == 255] = 0xFFFFFF00  # Yellow: A=255, R=255, G=255, B=0
            
            # Convert to QImage
            circle_qimage = QImage(argb_image.tobytes(), width, height, QImage.Format_ARGB32)
            
            # Overlay circles and draw text using QPainter
            painter = QPainter(pixmap)
            painter.drawImage(0, 0, circle_qimage)  # Draw the circle overlay
            
            # Set pen and font for text
            painter.setPen(QPen(QColor(255, 255, 0), 1))  # Yellow pen
            painter.setFont(QFont('Arial', 8))
            
            # Draw text for each spot
            for (y, x), label in spot_positions.items():
                painter.drawText(int(y) + 6, int(x), label)
            
            painter.end()

        # Update the frame number label
        self.frameNumberLabel.setText(f"Frame: {self.currentIndex + 1}")
        
        # Set the pixmap to the label
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.scrollBar.setValue(self.currentIndex)

    def scrollBarMoved(self, position):
        self.currentIndex = position
        self.showImage()


def main():
    app = QApplication(sys.argv)
    viewer = TimeLapseViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()