# Raw Data Collection

This directory contains raw datasets collected from individual sensors used in
the capstone project. The purpose of this folder is to store **unprocessed or
minimally processed sensor outputs** that can be used for offline analysis,
labeling, and future machine learning experiments.

Data in this folder is organized by sensor modality and by collection session.

---

## mmWave Radar Raw Data

The `mmWave_Radar_Raw_Data/` folder contains raw and parsed data collected from
the mmWave radar sensor.

Typical contents include:
- radar configuration files (`profile.cfg`)
- raw or parsed radar data logs
- point cloud data stored in CSV format
- utility scripts for parsing and visualization

### Data Format
Radar point cloud data is typically stored in a CSV file with fields such as:
- spatial coordinates (x, y, z)
- velocity (if available)
- signal strength or SNR (if available)

These CSV files represent radar observations after parsing the raw UART data
stream and are suitable for analysis in Python or spreadsheet tools.

---

## RGB Camera Raw Data

The `RGB_Camera_Raw_Data/` folder contains raw RGB data collected from the Intel
RealSense camera.

Typical contents include:
- Python scripts for logging RGB frames
- folders containing captured RGB image sequences
- optional CSV files that index frames and timestamps

### Data Format
Raw camera data is stored as:
- RGB image files (`.jpg`), representing raw visual observations
- accompanying CSV logs (when present) that record metadata such as:
  - timestamps
  - frame indices
  - image filenames

The image files represent the raw sensor output, while CSV files serve as an
index to organize, synchronize, and label the data during offline processing.

---

## Notes

- Data in this directory is collected independently of the live fusion system.
- Raw datasets are intended for offline experimentation, labeling, and machine
  learning development.
- Folder names such as `*_Raw_Data_1` indicate individual data collection
  sessions and may be extended with additional sessions over time.
