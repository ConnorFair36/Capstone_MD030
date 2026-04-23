# mmWave Radar Standalone Demo

This folder contains standalone scripts used to interface with the mmWave radar
sensor, log raw radar data, and parse point cloud information for offline
analysis.

The code in this directory was developed to validate radar operation
independently of the camera and sensor fusion system.

---

## Purpose

This demo focuses on:
- configuring the mmWave radar sensor
- logging raw radar UART data to a binary file
- parsing logged radar data into point cloud representations
- visualizing and inspecting radar outputs offline

This workflow enables repeatable analysis without requiring the radar hardware
to be connected at all times.

---

## Folder Contents

| File | Description |
|-----|------------|
| `radar_logger.py` | Connects to the radar UART data port and logs the raw data stream to a `.bin` file |
| `radar_raw.bin` | Example raw radar data capture (binary UART stream) |
| `radar_parse_pointcloud.py` | Parses a logged `.bin` file and extracts radar point cloud data |
| `profile.cfg` | Radar configuration file used to initialize the sensor |

---

## Data Logging Workflow

1. **Radar Configuration**
   - The radar is configured using `profile.cfg`
   - Configuration is sent over the radar configuration (CFG) port

2. **Raw Data Logging**
   - `radar_logger.py` reads the radar DATA port
   - The raw UART byte stream is saved to a `.bin` file
   - This file is not human-readable and represents unprocessed radar output

3. **Offline Parsing**
   - `radar_parse_pointcloud.py` reads the `.bin` file
   - Radar frames and TLVs are reconstructed
   - Point cloud data is extracted for analysis or visualization

---

## Running the Demo

### Step 1: Log Raw Radar Data
```bash
python radar_logger.py
```
This will create a .bin file containing the raw radar UART stream.
### Step 2: 
```bash
python radar_parse_pointcloud.py
```
This script processes the logged .bin file and outputs parsed point cloud data
(e.g., CSV or plots, depending on script configuration).

---
## Data Format

### Parsed radar point cloud data typically includes:
- spatial coordinates (x, y, z) in meters

- radial velocity (if available)

- signal strength or SNR (if available)

---

### Notes and Limitations 

Notes and Limitations

Raw `.bin` files can grow quickly depending on logging duration.

Indoor environments may introduce multipath reflections and noise.

This demo is intended for data collection and analysis, not real-time fusion.