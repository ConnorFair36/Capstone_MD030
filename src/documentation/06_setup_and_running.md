# Setup and Running

[Back to Index](index.md) | Previous: [ROS Topics Reference](05_ros_topics.md)

---

## Prerequisites

### Hardware

- **Intel RealSense D400-series camera** connected via USB 3.0
- **TI mmWave radar** (IWR-series) connected via two USB-to-serial adapters:
  - Config port (default: COM5, 115200 baud)
  - Data port (default: COM6, 921600 baud)
- A Windows PC with WSL 2 installed (for ROS 2)

### Software

- **Python 3.10+** (Windows, Anaconda recommended)
- **ROS 2 Humble** (inside WSL 2)
- **rosbridge_server** ROS 2 package (inside WSL 2)

### Python Dependencies

Install in your Python environment (e.g. `conda activate radar`):

```
opencv-python
numpy
pyrealsense2
ultralytics
pyserial
roslibpy
PySide6
```

`rclpy` comes with the ROS 2 installation inside WSL and does not need to be pip-installed.

---

## Serial Port Configuration

The radar module defaults to `COM5` (config) and `COM6` (data). If your serial adapters enumerate differently, update these constants at the top of `radar_module.py`:

```python
CFG_PORT = "COM5"     # ← change to your config port
DATA_PORT = "COM6"    # ← change to your data port
```

---

## Startup Order

Run each process in a **separate terminal**. The order matters for the first two (ROS infrastructure), but the remaining processes can start in any order.

### Terminal 1 -- ROS 2 Brain Node (WSL)

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run brain_pkg brain_node
```

### Terminal 2 -- Rosbridge Server (WSL)

```bash
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

This opens a WebSocket server on `localhost:9090` that bridges ROS 2 topics to `roslibpy` clients.

### Terminal 3 -- Camera Bridge (Windows / Anaconda)

```bash
conda activate radar
cd src/
python camera_to_rosbridge.py
```

### Terminal 4 -- Radar Bridge (Windows / Anaconda)

```bash
conda activate radar
cd src/
python radar_to_rosbridge.py
```

### Terminal 5 -- GUI (Windows / Anaconda)

```bash
conda activate radar
cd src/
python gui_app.py
```

The GUI will show "Waiting" for each data source until that source's bridge is running and publishing.

---

## Verifying the System

Once all 5 terminals are running:

1. The **Dashboard** window should show "Connected" for Camera Stream, ROS Bridge, and Radar Stream
2. The **Live Stream** window should display the camera feed with bounding boxes appearing when objects are detected
3. The **Bird's-Eye View** should show radar points as colored dots
4. The **Depth Heat Map** should show a color-coded depth visualization
5. The **Vehicle State** badge should cycle between SAFE, CAUTION, and STOP as you move objects in front of the sensors

---

## Stopping the System

Press `Ctrl+C` in any terminal to stop that process. The other processes will continue running (they'll show stale/disconnected indicators in the GUI).

To stop everything, press `Ctrl+C` in each terminal, or click "Quit" in the Dashboard window (this closes the GUI only).

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| GUI shows "Waiting" for everything | Rosbridge not running | Start Terminal 2 first |
| Camera bridge crashes on start | RealSense not connected or wrong calibration file | Check USB connection; ensure `calibration.json` exists in `src/` |
| Radar bridge crashes on start | Wrong COM port or radar not powered | Check `CFG_PORT`/`DATA_PORT` in `radar_module.py`; verify serial adapters in Device Manager |
| Brain node not showing in Dashboard | Rosbridge not bridging ROS 2 topics | Ensure both WSL terminals are running and rosbridge started successfully |
| YOLO is slow | Running on CPU | Check that CUDA/GPU is available; the nano model should still run at ~30 FPS on CPU |
| Depth heatmap is all black | Depth camera stream not aligned | Verify `calibration.json` matches your specific RealSense unit |

---

## Running Individual Modules Standalone

Each sensor module has a `if __name__ == "__main__"` block for standalone testing:

```bash
# Test camera module (prints detection summaries)
python camera_module.py

# Test radar module (prints point cloud data)
python radar_module.py
```

These run without ROS or rosbridge and are useful for verifying hardware connectivity.

---

Previous: [ROS Topics Reference](05_ros_topics.md)
