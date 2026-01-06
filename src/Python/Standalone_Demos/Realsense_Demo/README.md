# RealSense Standalone Demo

This folder contains standalone demos used to validate the Intel RealSense depth
camera and vision pipeline independently of the radar and fusion system.

These demos were used to confirm correct camera operation, depth measurement,
and person detection performance prior to sensor fusion.

---

## Purpose

This demo focuses on:
- verifying RealSense camera streaming (RGB + depth)
- testing depth-based distance estimation
- validating person detection using MediaPipe-based vision processing
- observing performance under different lighting and indoor conditions

---

## Contents

This folder may include:
- scripts for viewing depth/RGB streams
- scripts for person detection and distance estimation
- small utilities for testing camera settings and performance

Script names may vary depending on development stage.

---

## Hardware Requirements

- Intel RealSense depth camera connected via USB

---

## Running the Demo

From this directory:
```bash
python <script_name>.py
