# System Architecture

[Back to Index](index.md) | Previous: [Project Overview](01_project_overview.md) | Next: [Math and Science](03_math_and_science.md)

---

## Four-Process Design

The system runs as four independent processes that communicate exclusively through ROS topics. This decoupled design means each process can be started, stopped, or restarted independently without affecting the others.

```
┌─────────────────────┐     ┌─────────────────────┐
│   Intel RealSense   │     │   TI mmWave Radar    │
│   (USB depth+RGB)   │     │   (UART serial)      │
└────────┬────────────┘     └────────┬─────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  camera_module.py   │     │  radar_module.py     │
│  (YOLO + RealSense) │     │  (TLV parser)        │
└────────┬────────────┘     └────────┬─────────────┘
         │ import                    │ import
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│ camera_to_rosbridge │     │ radar_to_rosbridge   │
│ (roslibpy publisher)│     │ (roslibpy publisher) │
└────────┬────────────┘     └────────┬─────────────┘
         │                           │
         ▼     ROS Topics (via       ▼
         ├──── rosbridge on ─────────┤
         │     localhost:9090        │
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│    brain_node.py    │     │     gui_app.py       │
│  (ROS 2 / rclpy)   │     │  (PySide6 desktop)   │
│  Fusion decisions   │     │  4 display windows   │
│  SAFE/CAUTION/STOP  │     │                      │
└─────────┬───────────┘     └─────────────────────┘
          │                           ▲
          └──── /vehicle_cmd ─────────┘
                /fusion_debug
```

---

## Process Descriptions

### Process 1: Camera Bridge (`camera_to_rosbridge.py` + `camera_module.py`)

**Runs on:** Windows (native Python, Anaconda environment)

1. `camera_module.py` opens the Intel RealSense depth and color streams
2. Each frame is passed through YOLO for object detection and tracking
3. Depth at each detection center is extracted and back-projected to 3D
4. Velocity is estimated from frame-to-frame 3D position changes of tracked objects
5. A depth heatmap is generated for visualization
6. `camera_to_rosbridge.py` publishes all of this to 7 ROS topics at ~10 Hz

**Hold logic:** When no object is detected, the bridge continues publishing the last valid detection for 0.4 seconds to prevent flickering in the GUI and brain node.

### Process 2: Radar Bridge (`radar_to_rosbridge.py` + `radar_module.py`)

**Runs on:** Windows (native Python, same or separate terminal)

1. `radar_module.py` sends the chirp configuration (`profile.cfg`) to the radar via serial
2. Raw bytes are read from the data serial port at 921600 baud
3. The binary stream is parsed for TLV frames containing 3D point clouds
4. Points are filtered through a Region of Interest, sorted by range
5. The nearest point's velocity is smoothed and classified as motion state
6. `radar_to_rosbridge.py` publishes to 5 ROS topics at ~10 Hz

**Hold logic:** 0.6-second hold on last valid detection.

### Process 3: Brain Node (`brain_node.py`)

**Runs on:** WSL / Linux (native ROS 2 with `rclpy`)

1. Subscribes to 8 camera + radar summary topics (not the heavy image/JSON topics)
2. Every 100ms, computes a trust score for each sensor based on confidence and data freshness
3. Applies a decision matrix that considers distance, motion, and combined trust
4. Publishes `/vehicle_cmd` (SAFE, CAUTION, or STOP) and `/fusion_debug` (human-readable reason)

### Process 4: GUI (`gui_app.py`)

**Runs on:** Windows (native Python with PySide6)

1. Subscribes to all 14 ROS topics via `roslibpy`
2. Stores data in a shared `AppState` dataclass
3. Refreshes 4 windows at ~12.5 Hz (80ms timer):
   - **Dashboard** -- text-based status panel with overlay/filter controls
   - **Live Stream** -- camera feed with OpenCV overlays
   - **Bird's-Eye View** -- top-down radar point cloud with velocity arrows
   - **Depth Heat Map** -- color-coded depth visualization

---

## Data Flow

The data flows in one direction from sensors to display, with the brain node as a parallel consumer that feeds its decisions back:

1. **Sensors produce raw data** -- RealSense provides RGB frames + depth maps; the radar provides binary TLV packets
2. **Modules parse and process** -- `camera_module` runs YOLO and computes 3D positions; `radar_module` parses point clouds and classifies motion
3. **Bridges publish to ROS** -- structured data is serialized (JSON for complex data, scalars for summaries, base64 for images) and published via `roslibpy`
4. **Rosbridge distributes** -- the WebSocket server on `localhost:9090` relays messages to all subscribers
5. **Brain node fuses and decides** -- subscribes to summary topics only (no images), computes trust-weighted decisions
6. **GUI displays everything** -- subscribes to all topics including images, renders 4 live windows

---

## Why Rosbridge?

The system uses a mix of **native ROS 2** and **Python roslibpy** clients. This is because:

- The **brain node** runs as a native ROS 2 node inside WSL/Linux (where `rclpy` is available)
- The **camera and radar bridges** run on Windows (where `rclpy` is not easily available) and use `roslibpy` to publish over WebSocket
- The **GUI** also runs on Windows and uses `roslibpy` to subscribe

**Rosbridge** acts as the translator: it exposes ROS 2 topics over a WebSocket interface that any language or platform can use. This lets the Windows-side Python code participate in the ROS graph without needing a full ROS 2 installation.

---

## Communication Protocol

All messages use standard `std_msgs` types:

- `std_msgs/Bool` -- boolean flags (detected / not detected)
- `std_msgs/Float32` -- scalar measurements (distance, confidence)
- `std_msgs/String` -- text (labels, motion state, JSON arrays, base64 images)

Complex data (detection lists, point clouds) is serialized as JSON strings. Images are JPEG-encoded and transmitted as base64 strings. This keeps the ROS message types simple and avoids custom message definitions.

See [ROS Topics Reference](05_ros_topics.md) for the complete topic list.

---

Next: [Math and Science](03_math_and_science.md)
