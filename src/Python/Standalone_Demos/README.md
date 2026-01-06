# Standalone Sensor and Algorithm Demos

This folder contains independent demos used to validate individual sensors and
processing pipelines before integration into the full sensor fusion system.

Each demo can be run on its own and was developed to confirm correct hardware
operation, data acquisition, and basic processing.

---

## Demo Overview

### Classification_Demo
Demonstrates real-time classification and tracking behavior using vision-based
processing. This demo was used to validate person detection and basic motion
logic prior to fusion.

**Typical use cases:**
- Verifying vision pipeline functionality
- Testing classification behavior under different conditions

---

### mmWave_Radar_Demo
Demonstrates direct interaction with the mmWave radar sensor, including radar
configuration, data logging, and raw data capture.

**Typical use cases:**
- Verifying radar connectivity and configuration
- Collecting raw radar data for analysis

---

### Point_Cloud_Demo
Demonstrates point cloud parsing, visualization, and basic processing using
radar data.

**Typical use cases:**
- Inspecting radar point cloud output
- Validating parsing and plotting utilities

---

### Realsense_Demo
Demonstrates depth sensing and person detection using the Intel RealSense camera
and MediaPipe-based vision processing.

**Typical use cases:**
- Verifying camera operation and depth measurements
- Testing person detection independently of radar

---

## Running the Demos

Each demo folder contains one or more Python scripts that can be run directly.
From within the desired demo directory:

```bash
python <script_name>.py
