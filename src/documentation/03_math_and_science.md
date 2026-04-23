# Math and Science

[Back to Index](index.md) | Previous: [System Architecture](02_system_architecture.md) | Next: [Code Guide](04_code_guide.md)

---

## Table of Contents

1. [Radar Signal Processing](#1-radar-signal-processing)
2. [Camera Depth Estimation](#2-camera-depth-estimation)
3. [Coordinate Transforms and Projection](#3-coordinate-transforms-and-projection)
4. [YOLO Object Detection](#4-yolo-object-detection)
5. [Sensor Fusion and Decision Logic](#5-sensor-fusion-and-decision-logic)

---

## 1. Radar Signal Processing

### FMCW Chirp Basics

The TI mmWave radar uses **Frequency-Modulated Continuous Wave (FMCW)** technology. Instead of sending a pulse and measuring the echo delay (like traditional radar), it continuously transmits a signal whose frequency increases linearly over time. This linear frequency sweep is called a **chirp**.

```
Frequency
    ▲
    │      /    /    /    /
    │     / TX / TX / TX /     ← Transmitted chirps
    │    /   /   /   /
    │   /  /  /  /
    │  / /  / /
    │ // / //
    │────────────────────────→ Time
```

When this chirp hits an object and returns, the reflected signal is delayed by the round-trip time. The radar mixes the transmitted and received signals, producing a **beat frequency** proportional to the distance:

```
Range = (c * f_beat) / (2 * S)
```

Where:
- `c` = speed of light (3 x 10^8 m/s)
- `f_beat` = beat frequency (difference between TX and RX at any instant)
- `S` = chirp slope (Hz/s, how fast the frequency ramps)

The chirp slope `S` is configured in `profile.cfg` and determines the range resolution. A steeper slope gives finer range resolution.

### Doppler Velocity

To measure velocity, the radar transmits **multiple chirps** in a frame and looks at the **phase change** of the reflected signal from chirp to chirp. A moving object causes a consistent phase shift proportional to its radial velocity:

```
v_radial = (lambda * delta_phi) / (4 * pi * T_chirp)
```

Where:
- `lambda` = wavelength (c / f_center, approximately 3.9 mm at 77 GHz)
- `delta_phi` = phase difference between consecutive chirps
- `T_chirp` = time between chirps

Key properties of radial velocity:
- **Negative** velocity means the object is moving **toward** the radar (approaching)
- **Positive** velocity means the object is moving **away**
- Only the component of velocity along the line from radar to target is measured (radial component)
- Lateral motion (crossing in front of the radar) produces zero Doppler shift

This is why we decompose velocity in the bird's-eye view: the scalar `v` only tells us radial motion, so the direction is always along the line from origin to the point.

### CFAR Detection

Before the radar reports detected points, it applies **Constant False Alarm Rate (CFAR)** processing. This is an adaptive thresholding algorithm that:

1. Computes the FFT of the beat signal to produce a range-Doppler map
2. For each cell in the map, estimates the local noise floor from neighboring cells
3. Declares a detection only if the cell's power exceeds the noise estimate by a configurable margin

CFAR parameters (guard cells, training cells, threshold factor) are set in `profile.cfg`. They control the trade-off between detection sensitivity and false alarm rate.

### TLV Output

After CFAR and angle estimation, the radar's on-chip DSP outputs detected points in a **Type-Length-Value (TLV)** binary format over UART:

```
[8-byte magic word][32-byte header][TLV 1][TLV 2]...

Each TLV:
  [4-byte type][4-byte length][payload]

TLV Type 1 (Detected Points):
  N points, each 16 bytes: x(float32) y(float32) z(float32) v(float32)
```

The x, y, z coordinates are already in Cartesian metres (computed on-chip from range, azimuth, and elevation angles). The `v` field is the radial Doppler velocity in m/s.

The radar coordinate frame:
- **x** = lateral (left/right)
- **y** = forward (depth away from sensor)
- **z** = vertical (up/down)

---

## 2. Camera Depth Estimation

### Intel RealSense Structured Light

The Intel RealSense D400-series camera uses **active infrared (IR) stereo** to measure depth. It projects a known IR dot pattern onto the scene, then two IR cameras observe the pattern. The displacement (disparity) of each dot between the two cameras is inversely proportional to depth:

```
depth = (f * B) / disparity
```

Where:
- `f` = focal length of the IR cameras (in pixels)
- `B` = baseline distance between the two IR cameras (in metres)
- `disparity` = horizontal pixel shift of the same point between left and right IR images

The RealSense SDK handles this computation internally and provides a per-pixel depth map at 848x480 resolution, 30 frames per second.

### Depth-to-Color Alignment

The depth camera and the color camera are **physically offset** from each other -- they sit a few centimetres apart on the sensor board and have different fields of view and resolutions. To overlay depth onto the color image correctly, we need to know the spatial relationship between them.

This relationship is captured in `calibration.json` as a **4x4 extrinsic transformation matrix** (`depth_to_color_extrinsics_4x4`). The alignment process:

1. Convert each depth pixel to a 3D point using the **depth camera's intrinsics**
2. Apply the 4x4 extrinsic matrix to move those 3D points into the **color camera's coordinate frame**
3. Project back to 2D using the **color camera's intrinsics**

This produces a depth map that is pixel-aligned with the color image, so the depth at pixel (u, v) in the aligned depth map corresponds to the same object visible at pixel (u, v) in the RGB image.

In our code, `_transform_depth_to_img()` performs this full pipeline:

```python
raw_depth *= depth_scale_m_per_unit          # convert raw units to metres
pts_3d = depth_to_3d(raw_depth, DEPTH_INTRINSICS)    # depth pixels → 3D points
pts_3d = transform_pts(extrinsics_4x4, pts_3d)       # depth frame → color frame
x_px, y_px, d = pts3d_to_img(pts_3d, COLOR_INTRINSICS)  # 3D → color pixels
```

### Depth Map to 3D Point Cloud

The **back-projection** equation converts a 2D pixel coordinate plus depth into a 3D point in the camera's coordinate frame. Given the camera intrinsic parameters:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

Where:
- `(u, v)` = pixel coordinates
- `(cx, cy)` = principal point (optical center, in pixels)
- `(fx, fy)` = focal lengths (in pixels)
- `Z` = depth (in metres)

This is the inverse of perspective projection. Every pixel with a valid depth can be lifted to a 3D point in the camera frame.

In our code, `_img_to_pts3d(x_pixel, y_pixel, depth_m)` performs this for a single detection center, and `depth_to_3d(depth_map, intrinsics)` does it for the entire depth image at once (producing a 3xN array of 3D points).

---

## 3. Coordinate Transforms and Projection

### The Pinhole Camera Model

The pinhole model describes how a 3D point in the camera frame projects onto the 2D image plane:

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

Where:
- `(X, Y, Z)` = 3D point in the camera coordinate frame
- `Z` = depth (distance along the optical axis)
- `(fx, fy)` = focal lengths in pixels
- `(cx, cy)` = principal point in pixels
- `(u, v)` = resulting pixel coordinates

The division by `Z` is the **perspective divide** -- it's why nearby objects appear larger than distant ones.

Our camera intrinsics:
- `fx = 641.6`, `fy = 640.8` (pixels)
- `cx = 650.4`, `cy = 405.5` (pixels)
- Image size: 1280 x 720

### Extrinsic Transforms: Rotation + Translation

To move a point from one coordinate frame to another, we apply a rigid-body transformation consisting of a 3x3 **rotation matrix** R and a 3x1 **translation vector** T:

```
P_target = R * P_source + T
```

This is used in two places:

**1. Radar to Camera** (for projecting radar points onto the live image):

```
R_RADAR_TO_CAM = [[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]]

T_RADAR_TO_CAM = [0.04571, 0.03764, 0.02418]  (metres)
```

This rotation swaps axes:
- Radar X (lateral) stays as Camera X (lateral)
- Radar Y (forward) becomes Camera Z (depth)
- Radar Z (up) becomes Camera -Y (up in image is -Y)

The translation accounts for the physical offset between the radar and camera mounting positions.

**2. Camera to Radar** (inverse, for plotting camera detections on the bird's-eye view):

```
R_CAM_TO_RADAR = R_RADAR_TO_CAM^T  (transpose of rotation = inverse)
T_CAM_TO_RADAR = -R_CAM_TO_RADAR @ T_RADAR_TO_CAM
```

### Full Radar-to-Pixel Pipeline

To draw a radar point on the camera image:

```
1. Start with radar point:  P_radar = [x, y, z]
2. Transform to camera:     P_cam = R_RADAR_TO_CAM @ P_radar + T_RADAR_TO_CAM
3. Extract camera coords:   Xc, Yc, Zc = P_cam
4. Perspective projection:  u = fx * (Xc / Zc) + cx
                            v = fy * (Yc / Zc) + cy
5. Check bounds:            0 <= u < 1280 and 0 <= v < 720
```

### Bird's-Eye View Projection

The BEV is an **orthographic top-down projection** -- it removes perspective entirely. Looking straight down at the ground plane:

- Plot radar `x` (lateral) on the horizontal axis
- Plot radar `y` (forward) on the vertical axis
- Discard `z` (height) entirely

The mapping from metres to pixels:

```
pixel_x = (x_metres / X_RANGE + 1.0) * 0.5 * canvas_width
pixel_y = (1.0 - y_metres / Y_RANGE) * canvas_height
```

The radar origin sits at the bottom-center of the canvas. Forward (increasing y) goes up on screen.

### Velocity Arrow Decomposition

The radar's `v` value is a **scalar radial velocity** -- it tells us how fast the point is moving toward or away from the radar, but not the direction in 2D. To draw a velocity arrow on the BEV, we project `v` along the radial direction in the ground plane:

```
direction = (x, y) / sqrt(x^2 + y^2)     ← unit vector from radar to point
velocity_vector = v * direction            ← 2D velocity in the ground plane
```

This gives us `(vx, vy)` to draw as an arrow from the point's location. The arrow points toward the radar for approaching objects (v < 0) and away for receding objects (v > 0).

Note: this assumes the velocity is purely radial, which is what the Doppler measurement provides. Lateral motion is invisible to the radar and produces no arrow.

---

## 4. YOLO Object Detection

### How YOLO Works

YOLO (**You Only Look Once**) is a family of convolutional neural network (CNN) models designed for real-time object detection. Unlike two-stage detectors that first propose regions and then classify them, YOLO processes the entire image in a single forward pass.

The key idea:

1. The image is divided into a grid of cells
2. Each cell predicts multiple **bounding boxes** with confidence scores
3. Each cell also predicts **class probabilities** (what object is in the box)
4. Non-maximum suppression (NMS) removes duplicate detections

This single-pass design makes YOLO very fast -- the nano variant (YOLOv8n) can run at 30+ FPS even on modest hardware.

### What YOLO Gives Us

For each detected object, YOLO outputs:
- **Bounding box:** `[x1, y1, x2, y2]` pixel coordinates of the rectangle
- **Class ID:** integer identifying the object type (from the COCO dataset)
- **Confidence score:** 0.0 to 1.0, how certain the model is

We filter to keep only 5 classes relevant to our use case: Person (0), Backpack (24), Chair (56), Laptop (63), Phone (67).

### Tracking with Persistent IDs

We use `model.track()` instead of `model.predict()`. This enables the **BoT-SORT** tracker built into Ultralytics, which:

1. Assigns a unique integer ID to each detected object
2. Maintains that ID across frames as the object moves
3. Handles temporary occlusion and re-identification

This persistent ID is essential for velocity estimation, because we need to match the same object across consecutive frames.

### Velocity from Tracking

Once we have the same object's 3D position in two consecutive frames, velocity estimation is straightforward:

```
velocity = (xyz_current - xyz_previous) / delta_time
speed = ||velocity||  (Euclidean norm)
```

Where:
- `xyz_current` and `xyz_previous` are the 3D positions (from back-projection) at the detection center
- `delta_time` is the wall-clock time between frames (~33ms at 30 FPS)

The velocity is a 3D vector `[vx, vy, vz]` in the camera frame. The magnitude (`velocity_mag`) is the scalar speed.

Tracks older than 1 second are pruned to prevent stale data from corrupting velocity estimates.

---

## 5. Sensor Fusion and Decision Logic

### Why Fuse Two Sensors?

Each sensor has strengths and weaknesses:

| Property | Camera | Radar |
|---|---|---|
| Object identity | Excellent (class labels) | None (just reflections) |
| Distance accuracy | Good (depth camera) | Excellent (time-of-flight) |
| Velocity | Estimated (noisy, from tracking) | Direct measurement (Doppler) |
| Works in darkness | No | Yes |
| Works in rain/fog | Degraded | Mostly unaffected |
| Angular resolution | Very high (pixel-level) | Low (few degrees) |
| False positives | Rare for known classes | Common (any reflective surface) |

By combining both, we compensate for each sensor's weaknesses:
- Camera identifies *what* is there (is it a person or a wall?)
- Radar confirms *how close* and *how fast* (independent of lighting)
- When both agree, confidence is high; when they disagree, the system is cautious

### Trust Scoring

Each sensor gets a **trust score** (0.0 to 1.0) that reflects how reliable its current reading is:

**Camera trust:**

```
trust = clamp(camera_confidence, 0.0, 1.0)    if data is fresh (< 1 second old)
trust = 0.0                                     if data is stale
```

**Radar trust:**

```
base = radar_confidence
if motion == "APPROACHING":   base += 0.20    ← approaching objects are high priority
elif motion == "MOVING_AWAY": base += 0.05
elif motion == "STATIONARY":  base -= 0.10
elif motion == "NONE":        base -= 0.25    ← no motion data reduces trust
trust = clamp(base, 0.0, 1.0)                if data is fresh
trust = 0.0                                   if stale
```

The motion-based bonus reflects the physical reality that an approaching object is the most dangerous and deserves the highest trust multiplier.

### Staleness Handling

Both sensors have a **1-second timeout**. If no new data arrives within 1 second, the sensor's trust drops to 0.0 and its readings are treated as unavailable. This prevents the system from acting on outdated information if a sensor disconnects or freezes.

### Decision Matrix

The brain node evaluates rules in priority order. The first matching rule determines the output:

**Priority 1 -- Both sensors agree (highest confidence):**

```
IF   person detected < 2.5m AND camera_trust >= 0.75
AND  radar approaching < 2.8m AND radar_trust >= 0.65
THEN STOP (reason: camera_radar_agree_close_threat)
```

**Priority 2 -- Camera-only rules:**

```
IF   person detected < 1.2m AND camera_trust >= 0.35
THEN STOP (reason: camera_close_person)

IF   person detected < 2.5m AND camera_trust >= 0.35
THEN CAUTION (reason: camera_midrange_person)
```

**Priority 3 -- Radar-only rules:**

```
IF   radar approaching < 1.0m AND radar_trust >= 0.65
THEN STOP (reason: radar_close_approaching)

IF   radar approaching < 2.8m
THEN CAUTION (reason: radar_approaching)

IF   radar_trust >= 0.65 AND camera_trust < 0.40
THEN CAUTION (reason: radar_confident_camera_weak)
```

**Default:**

```
ELSE SAFE (reason: good_visual_no_threat)
```

### Why This Ordering?

The dual-sensor agreement rule (Priority 1) is checked last in the code but overrides everything -- when both sensors independently confirm a close, approaching threat, it's the highest-confidence situation and always results in STOP.

Camera rules come before radar rules because the camera provides object identity. A person at 1.5m is more actionable than an unknown radar reflection at 1.5m. But radar still has fallback authority: if the camera is weak (low confidence or stale) and the radar is confident about an approaching object, the system still triggers CAUTION.

The `SAFE` default ensures the system never blocks unnecessarily when no threat is confirmed.

---

Next: [Code Guide](04_code_guide.md)
