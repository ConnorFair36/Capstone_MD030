# Capstone MD030 -- Camera + Radar Sensor Fusion

A senior design capstone project that fuses an Intel RealSense RGB-D camera with a TI mmWave radar for real-time pedestrian detection and autonomous vehicle safety. The system detects, classifies, and tracks objects, then issues `SAFE`, `CAUTION`, or `STOP` commands based on distance, velocity, and confidence from both sensors.

---

## Quick Start

```bash
# 1. Start brain node + rosbridge in WSL (two terminals)
# 2. Then in Windows/Anaconda terminals, from the src/ folder:
python camera_to_rosbridge.py   # Terminal 3
python radar_to_rosbridge.py    # Terminal 4
python gui_app.py               # Terminal 5
```

See [Setup and Running](src/documentation/06_setup_and_running.md) for full instructions.

---

## Documentation

Full documentation lives in [`src/documentation/`](src/documentation/index.md):

1. [Project Overview](src/documentation/01_project_overview.md) -- Goals, sensors, and what the system does
2. [System Architecture](src/documentation/02_system_architecture.md) -- 4-process design and data flow
3. [Math and Science](src/documentation/03_math_and_science.md) -- Radar processing, camera depth, 3D projection, YOLO, fusion theory
4. [Code Guide](src/documentation/04_code_guide.md) -- File-by-file breakdown with every function and constant
5. [ROS Topics Reference](src/documentation/05_ros_topics.md) -- All 14 topics with types and message examples
6. [Setup and Running](src/documentation/06_setup_and_running.md) -- Prerequisites, dependencies, startup order

---

## Repository Structure

```
Capstone_MD030/
├── README.md              ← you are here
├── src/                   ← production code
│   ├── brain_node.py
│   ├── camera_module.py
│   ├── camera_to_rosbridge.py
│   ├── gui_app.py
│   ├── radar_module.py
│   ├── radar_to_rosbridge.py
│   ├── calibration.json
│   ├── profile.cfg
│   ├── yolo26n.pt
│   └── documentation/
└── experiments/           ← archived iterations and tools
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Object detection | Ultralytics YOLO (YOLOv8-nano) |
| Camera | Intel RealSense D400 (`pyrealsense2`) |
| Radar | TI mmWave IWR-series (`pyserial`) |
| Middleware | ROS 2 Humble + rosbridge |
| GUI | PySide6 + OpenCV |
