# Environment Setup Guide

This guide explains how to start all required components for the system.

---

## Terminal 1 – WSL (Run Brain Node)

Open a WSL terminal and run:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run brain_pkg brain_node
```

Keep this terminal running.

---

## Terminal 2 – WSL (Run ROS Bridge Server)

Open a second WSL terminal and run:

```bash
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Keep this terminal running.  
This acts as the ROS bridge server.

---

##  Anaconda Terminal – Run Python Script

Open an Anaconda terminal and run:

```bash
conda activate radar
python your_script_name.py
```

If running multiple Python files:
- Open a new Anaconda terminal for each script
- Activate the environment (`conda activate radar`)
- Run the script separately

---

## Recommended Startup Order

1. Start Brain Node  
2. Start ROS Bridge Server  
3. Run Python script(s)

---

## Stopping the System

To stop any running process, press:

```
Ctrl + C
```

in the corresponding terminal.
