# ROS Topics Reference

[Back to Index](index.md) | Previous: [Code Guide](04_code_guide.md) | Next: [Setup and Running](06_setup_and_running.md)

---

The system uses **14 ROS topics** for inter-process communication. All messages use standard `std_msgs` types -- no custom message definitions are needed.

Complex data (detection arrays, point clouds) is serialized as JSON strings. Images are JPEG-encoded and transmitted as base64 strings.

---

## Camera Topics

Published by `camera_to_rosbridge.py` at ~10 Hz.

| Topic | Type | Content |
|---|---|---|
| `/camera_detected` | `std_msgs/Bool` | `true` if any object detected this frame |
| `/camera_label` | `std_msgs/String` | Primary detection class name (e.g. `"Person"`) |
| `/camera_distance` | `std_msgs/Float32` | Primary detection depth in metres (`999.0` if none) |
| `/camera_confidence` | `std_msgs/Float32` | Primary detection confidence (`0.0` to `1.0`) |
| `/camera_detections_json` | `std_msgs/String` | JSON array of all detection dicts (see below) |
| `/camera_frame_b64` | `std_msgs/String` | Base64-encoded JPEG of the RGB frame |
| `/camera_depth_b64` | `std_msgs/String` | Base64-encoded JPEG of the depth heatmap |

**Subscribers:** `gui_app.py` (all 7), `brain_node.py` (first 4 only)

---

## Radar Topics

Published by `radar_to_rosbridge.py` at ~10 Hz.

| Topic | Type | Content |
|---|---|---|
| `/radar_detected` | `std_msgs/Bool` | `true` if valid points found in the ROI |
| `/radar_distance` | `std_msgs/Float32` | Nearest target range in metres (`999.0` if none) |
| `/radar_motion` | `std_msgs/String` | `APPROACHING`, `MOVING_AWAY`, `STATIONARY`, or `NONE` |
| `/radar_confidence` | `std_msgs/Float32` | Confidence score (`0.0` to `1.0`) |
| `/radar_points_json` | `std_msgs/String` | JSON array of point dicts (see below) |

**Subscribers:** `gui_app.py` (all 5), `brain_node.py` (first 4 only)

---

## Brain Topics

Published by `brain_node.py` at 10 Hz (100ms timer).

| Topic | Type | Content |
|---|---|---|
| `/vehicle_cmd` | `std_msgs/String` | `SAFE`, `CAUTION`, or `STOP` |
| `/fusion_debug` | `std_msgs/String` | Human-readable debug string (see example below) |

**Subscribers:** `gui_app.py`

---

## Message Examples

### /camera_detections_json

Each element in the JSON array:

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

### /radar_points_json

Each element in the JSON array:

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

### /fusion_debug

Example string:

```
state=CAUTION reason=camera_midrange_person | cam_trust=0.87 label=Person detected=True dist=1.95 | radar_trust=0.72 present=True radar_dist=2.01 motion=APPROACHING
```

---

## Topic Flow Diagram

```
camera_to_rosbridge ──→ /camera_detected ────→ brain_node
                    ──→ /camera_label ───────→ brain_node
                    ──→ /camera_distance ────→ brain_node
                    ──→ /camera_confidence ──→ brain_node
                    ──→ /camera_detections_json ─→ gui_app
                    ──→ /camera_frame_b64 ──────→ gui_app
                    ──→ /camera_depth_b64 ──────→ gui_app

radar_to_rosbridge ───→ /radar_detected ─────→ brain_node
                    ──→ /radar_distance ─────→ brain_node
                    ──→ /radar_motion ───────→ brain_node
                    ──→ /radar_confidence ───→ brain_node
                    ──→ /radar_points_json ────→ gui_app

brain_node ───────────→ /vehicle_cmd ────────→ gui_app
                    ──→ /fusion_debug ───────→ gui_app

Note: gui_app also subscribes to all camera/radar summary topics
      (those lines omitted for clarity)
```

---

Previous: [Code Guide](04_code_guide.md) | Next: [Setup and Running](06_setup_and_running.md)
