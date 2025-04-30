# Time-Lapse TIFF Viewer

A PyQt5-based application for viewing and analyzing time-lapse TIFF images, specifically designed for inspecting single nucleus time-lapse data with RNA track visualization and saving capabilities.

## Features

- Load and display time-lapse TIFF images from specified directories.
- Scroll through frames using a slider to inspect tracking quality.
- Visualize RNA tracks with customizable overlays (yellow circles and labels).
- Select and save specific tracks and frame ranges to an output directory.
- Support for navigating multiple time-stack files.
- Maintains folder structure for saved files (e.g., annotated/unannotated images, spot intensity tables).

## Prerequisites

- Python 3.8 or higher
- Required Python packages:
  ```
  pip install PyQt5 tifffile numpy pandas Pillow scikit-image opencv-python
  ```

## Installation

Clone this repository:

```bash
git clone https://github.com/CBIIT/ImageViewer.git
cd ImageViewer
```

Install the required dependencies (see [Prerequisites](#prerequisites)).

## Usage

Run the application:

```bash
python ImageViewer.py
```
This launches the Time-Lapse TIFF Viewer GUI.

### Load Time-Lapse TIFF
- Click the **"Load Time-Lapse TIFF"** button.
- Select TIFF files from either the `/cell_tracking/spot_image_patches` or `/cell_tracking/annotated_spot_image_patches` directory.
- The first single nucleus time-lapse will be displayed.

### Select Output Folder
- Click the **"Select Output Folder"** button to specify where to save inspected or modified tracks.
- Output will maintain the same subfolder structure as the source `cell_tracking` folder.

### Inspect the Time-Lapse
- Use the scroll bar below the image to navigate through the time-lapse frames and verify tracking quality.
- RNA tracks are displayed with yellow circles and labels on the image.

### Manage Tracks
- On the right panel, toggle checkboxes to include or exclude specific RNA tracks (e.g., to remove mis-segmented burst tracks).
- Adjust the minimum and maximum frame numbers using the spin boxes on the bottom right to select a specific frame range.

### Save Current Time-Stack
- If the tracking is satisfactory, click **"Save Current Time-Stack"**.
- Saves the current image, selected tracks, and related files (e.g., spot intensity tables, nuclei images) to the output folder.
- If a frame range is specified, only the selected frames are saved.

### Load Next Time-Stack
- Click **"Next Time-Stack"** to load the next single nucleus time-lapse for inspection.
- Repeat the process until all images are inspected.

## File Structure

The application expects and generates files in the following structure:

```
cell_tracking/
├── annotated_spot_image_patches/
├── spot_image_patches/
├── unannotated_spot_image_patches/
├── spot_intensity_tables/
│   ├── complete_tables/
│   └── integrated_intensity_tables/
├── single_track_images/
└── single_track_tables/
```

- **Input:** TIFF files must be loaded from `spot_image_patches` or `annotated_spot_image_patches`.
- **Output:** Saved files are organized into the same subfolder structure in the user-specified output directory.

## Notes

- Ensure the input TIFF files follow the naming convention:
  - `colX_rowY_fieldZ_cellW_ChV.tif` (e.g., `col1_row2_field3_cell4_Ch5.tif`).
- Spot intensity tables should be in CSV or TRK format with appropriate column names:
  - e.g., `chV_spot_no_W_x`, `chV_spot_no_W_y`
- The application handles multiple channels and copies unannotated images for other channels when saving.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed for analyzing single nucleus time-lapse data in a research context.
