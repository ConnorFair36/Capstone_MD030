# Camera and Radar Fusion System

This folder contains the integrated sensor fusion system that runs the camera
and mmWave radar simultaneously and produces fused detection results.

The fusion system was developed after validating each sensor independently and
serves as the primary demonstration of the capstone project.

---

## Contents

- `fusion_system.py`  
  Main entry point for the fusion system. Initializes both sensors and applies
  fusion logic in a continuous loop.

- `camera_module.py`  
  Handles camera input and person detection using a depth camera and
  MediaPipe-based vision processing.

- `radar_module.py`  
  Interfaces with the mmWave radar and provides motion and range information to
  the fusion system.

---

## Fusion Logic Overview

The fusion system follows a rule-based approach:

- If the camera detects a person, camera-based distance estimation is used.
- Radar data is used to confirm motion and detect movement.
- If the camera does not detect a person but radar motion is present, the system
  reports a possible occluded target.
- If neither sensor reports activity, no person is detected.

This approach improves robustness compared to using a single sensor alone.

---

## Running the Fusion System

From this directory:
```bash
python fusion_system.py
