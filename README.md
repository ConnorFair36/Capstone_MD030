# mmWave Radar + Camera Sensor Fusion Capstone


This repository contains the software developed for an engineering capstone
project focused on indoor human detection using **sensor fusion** between an
mmWave radar and a depth camera.

The goal of the project is to improve robustness over single-sensor systems by
combining:
- radar-based motion detection and ranging.
- vision-based person detection and distance estimation.

The system is designed to handle common indoor challenges such as occlusion,
poor lighting, and sensor noise.

---

## System Overview

The project is composed of three main components:

- **mmWave Radar**
    - Detects motion and provides range information
    - Useful for detecting moving or partially occluded targets

- **Camera (Intel RealSense + MediaPipe)**
    - Detects people and estimates distance using depth data
    - Provides reliable identification when line-of-sight is available

- **Fusion System**
    - Runs both sensors simultaneously
    - Uses camera distance when a person is visible
    - Uses radar motion to confirm movement or detect occluded targets

---

## Repository Structure
- `Standalone_Demos/` – Standalone demos for individual sensors and algorithms
- `Camera_&_Fusion_Demos/` – Integrated fusion system combining radar and camera inputs
- `Raw_Data_Collection/` – Raw sensor datasets and logging scripts
- `Machine_Learning/` – Planned ML development (documentation only)

See the README files inside each folder for setup and usage instructions.

## Hardware Used
- mmWave radar sensor
- Intel RealSense depth camera
