# Python Source Code

This directory contains all Python-based software developed for the capstone
project. The codebase is organized to reflect the engineering workflow used
throughout development: individual sensors and subsystems were validated
independently before being integrated into a unified sensor fusion system.

Each major component is organized into its own subdirectory, with localized
READMEs providing detailed usage instructions where appropriate.

---

## Directory Overview

### `Standalone_Demos/`
Contains independent demos used to validate individual sensors and algorithms
in isolation.

These demos were used to:
- verify mmWave radar operation and data parsing
- explore radar point cloud behavior
- validate RealSense camera streaming and vision-based detection
- test classification logic prior to fusion

This folder represents the early-stage development and validation work for each
sensor modality.

---

### `Camera_&_Fusion_Demos/`
Contains the integrated sensor fusion system that runs the camera and radar
simultaneously.

This directory includes:
- modular camera and radar interfaces
- the main fusion system logic
- real-time fusion rules combining camera-based detection with radar-based motion

The code here represents the transition from standalone validation to a unified
multi-sensor application.

---

### `Raw_Data_Collection/`
Contains scripts and datasets used for collecting raw or minimally processed
sensor data for offline analysis and machine learning.

This folder includes:
- raw mmWave radar data (binary logs and parsed point clouds)
- raw RGB camera data (image sequences and metadata logs)
- session-based data organization for repeatable experiments

Data collected here is intended for labeling, analysis, and future learning-based
experimentation.

---

### `Machine_Learning/`
This directory is reserved for future machine learning development.

At the current stage, it contains documentation outlining:
- intended ML tasks
- planned data inputs and outputs
- how learning-based methods will interface with the collected datasets

No trained models or finalized ML implementations are included yet.

---

## Notes

- Each subdirectory may contain its own README with component-specific details.
- Python is the primary development language for this project.
- Hardware-dependent scripts require the appropriate sensors to be connected.
