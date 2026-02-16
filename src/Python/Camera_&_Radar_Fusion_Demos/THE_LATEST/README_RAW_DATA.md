# Raw Data Pipeline - Camera & Radar to Brain Node

## Overview
This updated system sends **raw numpy arrays** from camera and radar nodes to a central brain subscriber node for processing.

## Message Type Comparison

### Option 1: std_msgs/ByteMultiArray ❌ (Not chosen)
- **Pros**: Very efficient for large arrays
- **Cons**: Complex serialization, need separate metadata
- **Use case**: When you need maximum performance with huge arrays

### Option 2: sensor_msgs/PointCloud2 ❌ (Not chosen)
- **Pros**: Industry standard, works with ROS visualization tools
- **Cons**: Complex conversion, overkill for simple use case
- **Use case**: When integrating with existing ROS ecosystem

### Option 3: JSON Strings ✅ (CHOSEN)
- **Pros**: Simple, flexible, works perfectly with rosbridge/roslibpy
- **Cons**: Less efficient than raw bytes (but fine for your data sizes)
- **Use case**: Perfect for rosbridge, easy to debug, good for moderate-sized arrays

### Option 4: Float32MultiArray ❌ (Not chosen)
- **Pros**: Standard ROS type
- **Cons**: Must flatten arrays, send shape separately
- **Use case**: When you want ROS standard types but not sensor_msgs

## What Changed

### 1. Camera Node (`camera_to_rosbridge_raw.py`)
**NEW:**
- Publishes to `/camera_raw` (std_msgs/String)
- Sends full color frame (1280x720x3) as JSON:
  ```json
  {
    "frame": [flat array of pixel values],
    "shape": [720, 1280, 3],
    "dtype": "uint8"
  }
  ```
  
**KEPT:**
- Still publishes `/person_detected` and `/person_distance` for backwards compatibility

### 2. Radar Node (`radar_to_rosbridge_raw.py`)
**NEW:**
- Publishes to `/radar_raw` (std_msgs/String)
- Sends point cloud arrays as JSON:
  ```json
  {
    "x": [x1, x2, ...],
    "y": [y1, y2, ...],
    "z": [z1, z2, ...],
    "v": [v1, v2, ...],
    "count": N
  }
  ```

**REQUIRES:**
- Updated `radar_module_raw.py` which stores last raw points

**KEPT:**
- Still publishes `/radar_detected`, `/radar_distance`, `/radar_motion`

### 3. Radar Module (`radar_module_raw.py`)
**NEW:**
- Stores last parsed point cloud in global variable
- New function: `get_last_raw_points()` returns raw x,y,z,v arrays

### 4. Brain Node (`brain_subscribe.py`)
**NEW:**
- Subscribes to `/camera_raw` and `/radar_raw`
- Reconstructs numpy arrays from JSON:
  - `self.camera_frame` → numpy array (H, W, 3)
  - `self.radar_points` → dict of numpy arrays
- Helper methods:
  - `get_camera_frame()` → returns latest frame or None
  - `get_radar_points()` → returns latest point cloud or None
- Logs raw data availability

**KEPT:**
- All original subscription topics and decision logic
- Backwards compatible with old publishers

## How to Use

### Setup
1. Replace your `radar_module.py` with `radar_module_raw.py`
2. Use `camera_to_rosbridge_raw.py` instead of `camera_to_rosbridge.py`
3. Use `radar_to_rosbridge_raw.py` instead of `radar_to_rosbridge.py`
4. Use `brain_subscribe.py` (converted from your .txt file)

### Running the System
```bash
# Terminal 1: Start rosbridge server (in WSL)
roslaunch rosbridge_server rosbridge_websocket.launch

# Terminal 2: Start camera publisher (Windows)
python camera_to_rosbridge_raw.py

# Terminal 3: Start radar publisher (Windows)
python radar_to_rosbridge_raw.py

# Terminal 4: Start brain subscriber (WSL)
python3 brain_subscribe.py

# Terminal 5 (optional): Start dashboard (Windows)
python stoplight_dashboard.py
```

### Processing Raw Data in Brain Node

The brain node now has access to raw numpy arrays. Here's how to use them:

```python
# In brain_subscribe.py, add to tick() or create new methods:

def process_camera_frame(self):
    """Example: process the raw camera frame"""
    frame = self.get_camera_frame()
    if frame is not None:
        # frame is numpy array (720, 1280, 3) uint8
        # Do whatever processing you need:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ... your processing ...
    
def process_radar_points(self):
    """Example: process the raw radar point cloud"""
    points = self.get_radar_points()
    if points is not None:
        # points is dict with numpy arrays:
        # points["x"], points["y"], points["z"], points["v"]
        
        # Example: filter points by distance
        mask = points["y"] < 3.0  # y is forward distance
        close_points = {
            "x": points["x"][mask],
            "y": points["y"][mask],
            "z": points["z"][mask],
            "v": points["v"][mask]
        }
        # ... your processing ...
```

## Data Flow Diagram

```
┌─────────────────┐
│  RealSense Cam  │
└────────┬────────┘
         │ cv2.waitKey()
         ▼
┌─────────────────────────────────┐
│  camera_to_rosbridge_raw.py     │
│  • Reads color frame            │
│  • Converts to JSON             │
└────────┬────────────────────────┘
         │
         ▼ /camera_raw (JSON string)
         │
         ▼
┌─────────────────────────────────┐
│  brain_subscribe.py             │
│  • Parses JSON                  │
│  • Reconstructs numpy array     │  ◄─── You process here!
│  • self.camera_frame available  │
└─────────────────────────────────┘


┌─────────────────┐
│  mmWave Radar   │
└────────┬────────┘
         │ serial port
         ▼
┌─────────────────────────────────┐
│  radar_module_raw.py            │
│  • Parses radar packets         │
│  • Stores x,y,z,v arrays        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  radar_to_rosbridge_raw.py      │
│  • Gets raw arrays              │
│  • Converts to JSON             │
└────────┬────────────────────────┘
         │
         ▼ /radar_raw (JSON string)
         │
         ▼
┌─────────────────────────────────┐
│  brain_subscribe.py             │
│  • Parses JSON                  │
│  • Reconstructs numpy arrays    │  ◄─── You process here!
│  • self.radar_points available  │
└─────────────────────────────────┘
```

## Performance Notes

**Camera Frame Size:**
- Resolution: 1280 × 720 × 3 = 2,764,800 values
- JSON size: ~8-10 MB per frame
- At 10 Hz: ~80-100 MB/s bandwidth

**Radar Point Cloud:**
- Typical: 5-20 points per frame
- 4 values per point (x, y, z, v)
- JSON size: ~500 bytes - 2 KB per frame
- At 10 Hz: ~5-20 KB/s bandwidth

**Total:** ~80-100 MB/s (dominated by camera)

This is totally fine for:
- ✅ localhost/WSL communication
- ✅ 10 Hz publish rate
- ✅ Your processing needs

If you need to optimize later:
- Use lower camera resolution
- Reduce publish rate
- Switch to ByteMultiArray encoding
- Use image compression (JPEG)

## Troubleshooting

**"AttributeError: module 'radar_module' has no attribute 'get_last_raw_points'"**
→ Make sure you're using `radar_module_raw.py` instead of the old `radar_module.py`

**"No raw data received in brain node"**
→ Check that rosbridge is running and publishers are connected

**"JSON decode error"**
→ Check that numpy array conversion to list is working (should be automatic)

**"Stale data warnings"**
→ Normal if publishers haven't started yet; check publish rate and timeouts
