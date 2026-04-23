# Project Overview

[Back to Index](index.md)

---

## What This Project Is

This is a **senior design capstone project** focused on **real-time sensor fusion for pedestrian detection**. The system combines an RGB-D camera and a millimeter-wave radar to detect, classify, and track objects in front of a vehicle, then issues safety commands (`SAFE`, `CAUTION`, or `STOP`) based on how close and how fast an obstacle is approaching.

The core idea is that **no single sensor is sufficient on its own**. Cameras are excellent at identifying *what* an object is (person vs. chair vs. backpack) but struggle in poor lighting, rain, or fog. Radar is excellent at measuring *how far* and *how fast* something is moving, but cannot tell you what it is. By fusing both, the system is more robust than either sensor alone.

---

## The Problem

Autonomous and semi-autonomous vehicles need to detect pedestrians and obstacles reliably enough to make split-second safety decisions. A missed detection or a false positive can have serious consequences. This project demonstrates a low-cost, real-time prototype that:

- Detects pedestrians and common objects in a forward-facing field of view
- Measures distance from two independent sensors (camera depth + radar range)
- Estimates whether an obstacle is approaching, stationary, or moving away
- Fuses both measurements into a single safety decision with confidence scoring
- Displays everything in a live GUI for monitoring and debugging

---

## Sensor Suite

### Intel RealSense D400-series (RGB + Depth)

- **Color stream:** 1280x720 at 30 fps -- used for YOLO object detection
- **Depth stream:** 848x480 at 30 fps -- structured-light infrared, provides per-pixel distance
- **What it gives us:** Object identity (class label), bounding boxes, 3D position from back-projection, frame-to-frame velocity estimation via tracking

### TI mmWave Radar (IWR-series)

- **FMCW radar:** 77 GHz, configurable chirp profile
- **Output:** Point cloud of detected reflections, each with 3D position (x, y, z) and radial Doppler velocity (v)
- **What it gives us:** Range and velocity of nearby objects, independent of lighting or weather, at rates up to 10+ Hz

---

## What the System Outputs

The brain node fuses camera and radar inputs and publishes one of three vehicle commands:

| Command | Meaning | When it triggers |
|---|---|---|
| **SAFE** | No threat detected | No close obstacles, or low confidence from both sensors |
| **CAUTION** | Potential hazard nearby | Person or approaching object at mid-range (1.0--2.8 m) |
| **STOP** | Immediate threat | Person or approaching object very close (< 1.2 m), confirmed by one or both sensors |

---

## GUI Display

The system includes a 4-window desktop GUI for real-time monitoring:

- **Dashboard** -- system status, current detection, vehicle state, overlay/filter controls
- **Live Stream** -- camera RGB with bounding boxes, projected radar points, velocity text
- **Bird's-Eye View** -- top-down radar point cloud with velocity arrows
- **Depth Heat Map** -- color-coded depth visualization (near = red, far = blue)

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Object detection | Ultralytics YOLO (YOLOv8-nano) |
| Camera SDK | Intel RealSense SDK (`pyrealsense2`) |
| Radar interface | Serial (`pyserial`) with TI TLV protocol |
| Middleware | ROS 2 Humble + rosbridge (WebSocket) |
| GUI framework | PySide6 (Qt 6) with OpenCV rendering |
| Communication | `roslibpy` (Python rosbridge client) + `rclpy` (native ROS 2) |

---

Next: [System Architecture](02_system_architecture.md)
