# Fusion Demo Code Guide

> **Folder:** `fusion_demo_04_21_2026/`
> **Last updated:** April 21, 2026

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [How to Run](#how-to-run)
3. [File Breakdown](#file-breakdown)
   - [camera_module.py](#camera_modulepy)
   - [camera_to_rosbridge.py](#camera_to_rosbridgepy)
   - [radar_module.py](#radar_modulepy)
   - [radar_to_rosbridge.py](#radar_to_rosbridgepy)
   - [brain_node.py](#brain_nodepy)
   - [gui_app.py](#gui_apppy)
4. [ROS Topic Reference](#ros-topic-reference)
5. [Data Structures](#data-structures)
6. [Configuration Files](#configuration-files)
7. [Key Constants](#key-constants)

---

## System Architecture

The system runs as **four independent processes** communicating over ROS topics via **rosbridge** (WebSocket on `localhost:9090`).

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

**Data flow summary:**

1. `camera_to_rosbridge` calls `camera_module.get_camera_data()` in a loop, publishes results to 7 ROS topics
2. `radar_to_rosbridge` calls `radar_module.get_radar_data()` in a loop, publishes results to 5 ROS topics
3. `brain_node` subscribes to camera + radar summary topics, fuses them into a vehicle command (`SAFE` / `CAUTION` / `STOP`), publishes to 2 topics
4. `gui_app` subscribes to all 14 topics and renders 4 windows in real time

---

## How to Run

### Prerequisites

- Python 3.10+
- ROS 2 (for `brain_node.py`)
- rosbridge running on `localhost:9090`
- Intel RealSense camera connected via USB
- TI mmWave radar connected via serial (COM5 config, COM6 data -- adjust in `radar_module.py`)

### Python dependencies

```
opencv-python
numpy
pyrealsense2
ultralytics
pyserial
roslibpy
PySide6
rclpy (comes with ROS 2)
```

### Startup order

Run each in a **separate terminal**, from inside the `fusion_demo_04_21_2026/` folder:

```bash
# Terminal 1 -- Start rosbridge (if not already running)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# Terminal 2 -- Camera bridge
python camera_to_rosbridge.py

# Terminal 3 -- Radar bridge
python radar_to_rosbridge.py

# Terminal 4 -- Brain node (ROS 2)
python brain_node.py

# Terminal 5 -- GUI
python gui_app.py
```

The GUI can start before or after the bridges. It will show "Waiting" until data arrives.

---

## File Breakdown

### camera_module.py

**Lines:** 351 | **Purpose:** Hardware + ML pipeline for the Intel RealSense camera

**What it does:**

1. Opens the RealSense depth (848x480) and color (1280x720) streams
2. Transforms raw depth into the color camera frame using calibration extrinsics
3. Runs YOLO tracking (`model.track`) on each color frame
4. For each detected object: extracts bounding box, computes depth at center (median in a 5x5 window), back-projects to 3D `xyz`, estimates velocity from track history
5. Generates a color-coded depth heatmap (JET colormap, 0.1m--6.0m range)
6. Returns everything in a single dict

**Key functions:**

| Function | What it does |
|---|---|
| `initialize(model_path)` | Starts RealSense pipeline, loads YOLO model |
| `shutdown()` | Stops pipeline, cleans up |
| `get_camera_data()` | Main loop call -- returns frame, detections, depth heatmap, primary detection summary |
| `_safe_depth(depth, cx, cy)` | Median depth in a small window around a pixel, filters out invalid values |
| `_transform_depth_to_img(raw_depth)` | Converts depth frame from depth camera origin to color camera origin using extrinsics |
| `depth_to_3d(depth_map, intrinsics)` | Converts a full depth image into a 3xN array of 3D points |
| `pts3d_to_img(pts, intrinsics)` | Projects 3D points back to image pixel coordinates |
| `transform_pts(matrix, points)` | Applies a 4x4 transformation matrix to a 3xN point array |
| `_img_to_pts3d(x, y, depth)` | Back-projects a single pixel + depth to 3D using color intrinsics |
| `depth_to_heatmap_bgr(depth_img_m)` | Converts depth image to a BGR heatmap with near/far labels |
| `_compute_velocity(track_id, xyz, t)` | Estimates velocity from consecutive 3D positions of the same tracked object |
| `_cleanup_old_tracks(now_t)` | Removes tracks older than 1 second |
| `_primary_detection(detections)` | Picks the best detection: closest distance first, then highest confidence |
| `encode_frame_to_b64(frame)` | JPEG-encodes a BGR frame and returns base64 string |

**Detected object classes:**

| COCO ID | Label | Color (BGR) |
|---|---|---|
| 0 | Person | (0, 255, 0) green |
| 24 | Backpack | (255, 0, 0) blue |
| 56 | Chair | (0, 165, 255) orange |
| 63 | Laptop | (255, 0, 255) magenta |
| 67 | Phone | (0, 255, 255) yellow |

---

### camera_to_rosbridge.py

**Lines:** 123 | **Purpose:** Bridges camera_module output to ROS topics

**What it does:**

1. Calls `camera_module.initialize("yolo26n.pt")`
2. Connects to rosbridge at `localhost:9090`
3. Loops at ~10 Hz: calls `get_camera_data()`, publishes 7 topics
4. Implements **hold logic** (0.4 seconds): when no detection is found, it re-publishes the last valid detection for 0.4s to avoid flickering

**Topics published:** `/camera_detected`, `/camera_label`, `/camera_distance`, `/camera_confidence`, `/camera_detections_json`, `/camera_frame_b64`, `/camera_depth_b64`

---

### radar_module.py

**Lines:** 238 | **Purpose:** TI mmWave radar serial interface and point cloud parser

**What it does:**

1. Sends `profile.cfg` line-by-line to the radar config port (COM5 at 115200 baud)
2. Opens the data port (COM6 at 921600 baud)
3. Reads raw bytes into a buffer, searches for the 8-byte magic word to find frame boundaries
4. Parses the TLV (Type-Length-Value) structure: extracts detected points from TLV type 1
5. Each point is 16 bytes: 4 little-endian floats (`x`, `y`, `z`, `v`)
6. Filters points through a Region of Interest (ROI)
7. Sorts by range, picks the nearest as primary target
8. Smooths velocity over a rolling window of 5 readings
9. Classifies motion as `APPROACHING`, `MOVING_AWAY`, or `STATIONARY`
10. Computes a confidence score based on point count, velocity magnitude, and motion type

**Key functions:**

| Function | What it does |
|---|---|
| `initialize()` | Sends config, opens data serial port |
| `close()` | Closes serial port |
| `get_radar_data()` | Main loop call -- parses buffer, returns points + summary dict or `None` |
| `send_cfg()` | Reads `profile.cfg` and sends each line to the config serial port |
| `_parse_packet(packet)` | Parses a complete TLV packet into x/y/z/v arrays |
| `_find_magic(buffer)` | Finds the magic word in the byte buffer |
| `_point_in_roi(x, y, z, r)` | Returns True if a point is within the ROI bounds |
| `_classify_motion(v)` | Maps smoothed velocity to a motion label |
| `_compute_confidence(...)` | Scores detection confidence (0.0 to 1.0) |
| `_smooth_velocity(v)` | Rolling mean over last 5 velocity readings |

**Binary frame structure:**

```
[8-byte magic word][32-byte header][TLV 1][TLV 2]...

Header (8 x uint32 LE):
  version, total_packet_len, platform, frame_num,
  time_cpu_cycles, num_detected_obj, num_tlvs, subframe_num

TLV (type=1, detected points):
  [4-byte type][4-byte length][N x 16-byte points]
  Each point: x(f32), y(f32), z(f32), v(f32)
```

---

### radar_to_rosbridge.py

**Lines:** 104 | **Purpose:** Bridges radar_module output to ROS topics

**What it does:**

1. Calls `radar_module.initialize()`
2. Connects to rosbridge at `localhost:9090`
3. Loops at ~10 Hz: calls `get_radar_data()`, publishes 5 topics
4. Implements **hold logic** (0.6 seconds): re-publishes last valid detection briefly when no data arrives

**Topics published:** `/radar_detected`, `/radar_distance`, `/radar_motion`, `/radar_confidence`, `/radar_points_json`

---

### brain_node.py

**Lines:** 176 | **Purpose:** ROS 2 fusion/decision node

**What it does:**

1. Runs as a native ROS 2 node (uses `rclpy`, not `roslibpy`)
2. Subscribes to 8 camera + radar summary topics
3. Every 100ms (`tick`), computes a vehicle command:
   - Calculates **camera trust** (clamped confidence, 0 if stale)
   - Calculates **radar trust** (confidence + motion bonus: +0.20 approaching, +0.05 moving away, -0.10 stationary, -0.25 none)
   - Applies decision rules (see below)
4. Publishes the resulting state and a debug string

**Decision logic (priority order):**

| Condition | Result | Reason |
|---|---|---|
| Camera+radar both agree: person close + approaching + high trust | `STOP` | `camera_radar_agree_close_threat` |
| Person detected < 1.2m, camera trust >= 0.35 | `STOP` | `camera_close_person` |
| Person detected < 2.5m, camera trust >= 0.35 | `CAUTION` | `camera_midrange_person` |
| Radar approaching < 1.0m, radar trust >= 0.65 | `STOP` | `radar_close_approaching` |
| Radar approaching < 2.8m | `CAUTION` | `radar_approaching` |
| Radar confident but camera weak | `CAUTION` | `radar_confident_camera_weak` |
| Everything else | `SAFE` | `good_visual_no_threat` |

**Staleness:** Both camera and radar data expire after 1.0 second with no update.

---

### gui_app.py

**Lines:** 770 | **Purpose:** PySide6 desktop GUI with 4 windows

**What it does:**

1. Connects to rosbridge via `roslibpy`, subscribes to all 14 topics
2. Stores everything in a shared `AppState` dataclass
3. Refreshes all windows at ~12.5 Hz (80ms timer)

**The 4 windows:**

#### DashboardWindow

A control panel with 6 grouped sections in a 3x2 grid:

| Section | Contents |
|---|---|
| System Status | Connection indicators for camera, rosbridge, brain node, radar |
| Current Detection | Primary detection: label, distance, confidence, velocity |
| Vehicle State | Large colored badge: green SAFE, yellow CAUTION, red STOP + debug reason |
| Overlay Controls | 14 checkboxes controlling what draws on the live view and BEV |
| Object Filters | Per-class toggles: Person, Backpack, Chair, Laptop, Phone |
| Radar Status | Radar detection, distance, motion, confidence, point count |

#### LiveViewWindow

The camera RGB feed with overlays drawn via OpenCV:

- Bounding boxes with labels, confidence, distance, velocity
- Projected radar points (colored dots with velocity text)
- Radar-to-detection match lines
- Radar summary box (top-left corner)
- Class color legend (top-right corner)
- Vehicle state text (bottom)

#### BirdEyeViewWindow

A top-down orthographic view of radar points:

- 420x420 pixel canvas, dark background with 1m grid
- Radar origin at bottom-center, +y upward (forward)
- View range: -5m to +5m lateral, 0 to 10m forward
- Radar points as colored dots (red = approaching, white = stationary, blue = moving away)
- Velocity arrows decomposed from radial Doppler velocity into the x-y ground plane
- Camera detections (if `xyz` available) inverse-transformed to radar frame, shown as labeled squares

#### DepthHeatMapWindow

The depth camera output rendered as a JET color heatmap (0.1m--6.0m range), streamed via `/camera_depth_b64`.

**Radar point projection to camera image:**

The GUI transforms radar points from radar frame to camera frame using:
- Rotation matrix `R_RADAR_TO_CAM` and translation `T_RADAR_TO_CAM` (extrinsics)
- Then pinhole projection with camera intrinsics (`FX`, `FY`, `CX`, `CY`)

This is only used for the Live Stream overlay. The BEV uses raw radar-frame coordinates directly.

---

## ROS Topic Reference

### Camera topics (published by camera_to_rosbridge)

| Topic | Type | Content |
|---|---|---|
| `/camera_detected` | `std_msgs/Bool` | True if any object detected |
| `/camera_label` | `std_msgs/String` | Primary detection label (e.g. "Person") |
| `/camera_distance` | `std_msgs/Float32` | Primary detection depth in metres (999.0 if none) |
| `/camera_confidence` | `std_msgs/Float32` | Primary detection confidence (0.0--1.0) |
| `/camera_detections_json` | `std_msgs/String` | JSON array of all detection dicts |
| `/camera_frame_b64` | `std_msgs/String` | Base64 JPEG of the color frame |
| `/camera_depth_b64` | `std_msgs/String` | Base64 JPEG of the depth heatmap |

### Radar topics (published by radar_to_rosbridge)

| Topic | Type | Content |
|---|---|---|
| `/radar_detected` | `std_msgs/Bool` | True if valid points in ROI |
| `/radar_distance` | `std_msgs/Float32` | Nearest target range in metres (999.0 if none) |
| `/radar_motion` | `std_msgs/String` | `APPROACHING`, `MOVING_AWAY`, `STATIONARY`, or `NONE` |
| `/radar_confidence` | `std_msgs/Float32` | Confidence score (0.0--1.0) |
| `/radar_points_json` | `std_msgs/String` | JSON array of point dicts |

### Brain topics (published by brain_node)

| Topic | Type | Content |
|---|---|---|
| `/vehicle_cmd` | `std_msgs/String` | `SAFE`, `CAUTION`, or `STOP` |
| `/fusion_debug` | `std_msgs/String` | Human-readable debug string with all sensor values and decision reason |

---

## Data Structures

### Radar point dict

Each element of the `radar_points_json` array:

```json
{
  "x": 0.32,
  "y": 2.15,
  "z": -0.08,
  "v": -0.45,
  "range": 2.18
}
```

| Field | Type | Unit | Description |
|---|---|---|---|
| `x` | float | metres | Lateral position (raw from sensor) |
| `y` | float | metres | Forward/depth position (raw from sensor) |
| `z` | float | metres | Vertical/height position (raw from sensor) |
| `v` | float | m/s | Radial Doppler velocity (negative = approaching) |
| `range` | float | metres | Euclidean distance: sqrt(x^2 + y^2 + z^2) |

### Camera detection dict

Each element of the `camera_detections_json` array:

```json
{
  "object_id": 3,
  "class_id": 0,
  "label": "Person",
  "confidence": 0.87,
  "distance": 2.14,
  "bbox": [412, 118, 620, 580],
  "center": [516, 349],
  "color": [0, 255, 0],
  "xyz": [0.15, -0.42, 2.14],
  "velocity": [0.02, -0.01, -0.34],
  "velocity_mag": 0.34
}
```

| Field | Type | Description |
|---|---|---|
| `object_id` | int or null | YOLO tracker ID (persistent across frames) |
| `class_id` | int | COCO class ID |
| `label` | string | Human-readable class name |
| `confidence` | float | Detection confidence (0.0--1.0) |
| `distance` | float or null | Depth at bbox center in metres |
| `bbox` | [x1, y1, x2, y2] | Bounding box in pixels |
| `center` | [cx, cy] | Bbox center in pixels |
| `color` | [B, G, R] | Drawing color for this class |
| `xyz` | [x, y, z] or null | 3D position in camera frame (metres) |
| `velocity` | [vx, vy, vz] or null | Estimated velocity in camera frame (m/s) |
| `velocity_mag` | float or null | Speed (magnitude of velocity vector) |

### get_camera_data() return dict

```json
{
  "frame": "<numpy BGR image>",
  "depth_frame": "<numpy float32 depth in metres>",
  "depth_heatmap_bgr": "<numpy BGR heatmap image>",
  "detected": true,
  "label": "Person",
  "distance": 2.14,
  "confidence": 0.87,
  "detections": ["<array of detection dicts>"]
}
```

### get_radar_data() return dict

```json
{
  "detected": true,
  "distance": 2.18,
  "motion": "APPROACHING",
  "confidence": 0.72,
  "points": ["<array of point dicts>"]
}
```

---

## Configuration Files

### calibration.json

Camera calibration data. The module tries `calibration.json` first, then falls back to `calibration(3).json`.

Contains:

| Key | Description |
|---|---|
| `color_intrinsics` | Color camera intrinsics: `fx`, `fy`, `ppx`, `ppy`, `width`, `height` |
| `depth_intrinsics` | Depth camera intrinsics: same fields |
| `depth_to_color_extrinsics_4x4` | 4x4 transformation matrix from depth origin to color origin |
| `depth_scale_m_per_unit` | Conversion factor from raw depth units to metres |

### profile.cfg

TI mmWave radar configuration file. Sent line-by-line to the config serial port at startup. Controls:

- Chirp profile (start frequency, slope, idle time, ADC samples)
- Frame configuration (periodicity, number of chirps)
- CFAR and detection thresholds
- Output data format

Lines starting with `%` are comments and are skipped.

### yolo26n.pt

Ultralytics YOLO model weights (~5.5 MB). A YOLOv8-nano variant fine-tuned or renamed for this project. Loaded by `camera_module.initialize()`.

---

## Key Constants

### Radar (radar_module.py)

| Constant | Value | Description |
|---|---|---|
| `CFG_PORT` | `"COM5"` | Serial port for radar configuration |
| `DATA_PORT` | `"COM6"` | Serial port for radar data stream |
| `BAUD_CFG` | 115200 | Config port baud rate |
| `BAUD_DATA` | 921600 | Data port baud rate |
| `ROI_X_ABS` | 2.5 m | Max lateral distance for ROI |
| `ROI_Y_MIN` / `ROI_Y_MAX` | 0.1 / 10.0 m | Forward depth range for ROI |
| `ROI_Z_ABS` | 2.0 m | Max height for ROI |
| `RANGE_MIN` / `RANGE_MAX` | 0.1 / 10.0 m | Valid range bounds |
| `APPROACH_THRESH` | -0.2 m/s | Velocity below this = approaching |
| `AWAY_THRESH` | 0.2 m/s | Velocity above this = moving away |
| `VEL_HISTORY_LEN` | 5 | Rolling window for velocity smoothing |

### Camera (camera_module.py)

| Constant | Value | Description |
|---|---|---|
| `CONF_THRESHOLD` | 0.50 | Minimum YOLO confidence to keep a detection |
| `JPEG_QUALITY` | 80 | JPEG compression quality for base64 encoding |
| `DEPTH_HEATMAP_MIN_M` | 0.1 m | Near clip for depth heatmap |
| `DEPTH_HEATMAP_MAX_M` | 6.0 m | Far clip for depth heatmap |

### Brain (brain_node.py)

| Constant | Value | Description |
|---|---|---|
| `STOP_DIST_M` | 1.2 m | Camera distance threshold for STOP |
| `CAUTION_DIST_M` | 2.5 m | Camera distance threshold for CAUTION |
| `RADAR_STOP_DIST_M` | 1.0 m | Radar distance threshold for STOP |
| `RADAR_CAUTION_DIST_M` | 2.8 m | Radar distance threshold for CAUTION |
| `CAM_CONF_GOOD` | 0.75 | High camera confidence threshold |
| `CAM_CONF_POOR` | 0.40 | Low camera confidence threshold |
| `RADAR_CONF_GOOD` | 0.65 | High radar confidence threshold |
| `CAMERA_STALE_S` | 1.0 s | Camera data expiry |
| `RADAR_STALE_S` | 1.0 s | Radar data expiry |

### GUI (gui_app.py)

| Constant | Value | Description |
|---|---|---|
| `FX`, `FY` | 641.6, 640.8 | Color camera focal lengths (pixels) |
| `CX`, `CY` | 650.4, 405.5 | Color camera principal point (pixels) |
| `IMG_W`, `IMG_H` | 1280, 720 | Expected image dimensions |
| `BEV_W`, `BEV_H` | 420, 420 | Bird's-eye view canvas size (pixels) |
| `BEV_X_RANGE` | 5.0 m | BEV half-width (lateral) |
| `BEV_Y_RANGE` | 10.0 m | BEV forward depth |
| `BEV_ARROW_SCALE` | 18.0 | Velocity arrow visual length multiplier |

### Radar-to-Camera Extrinsics (gui_app.py)

```
R_RADAR_TO_CAM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
T_RADAR_TO_CAM = [0.04571, 0.03764, 0.02418]
```

This swaps radar Y (forward) to camera Z (depth) and radar Z (up) to camera -Y (down), plus a small physical offset.
