# mmWave Radar Standalone Demo

This folder contains standalone code used to interface with the mmWave radar
sensor and validate radar operation independently of the camera and fusion
system.

These demos were used to confirm correct radar configuration, data streaming,
and basic motion/range detection before integration into the full sensor fusion
pipeline.

---

## Contents

- `radar_module.py`  
  Python module responsible for:
  - opening radar serial ports
  - loading the radar configuration file
  - parsing incoming radar data
  - providing motion and distance information

- `profile.cfg`  
  Radar configuration file that defines chirp parameters, frame rate, and
  detection behavior.

- Additional scripts (if present)  
  Used for logging, debugging, or visualizing radar output during development.

---

## Hardware Requirements

- mmWave radar sensor connected via USB
- Two serial interfaces exposed by the radar:
  - **CFG port** (configuration)
  - **DATA port** (high-speed data stream)

---

## Configuration

The radar requires a configuration file (`profile.cfg`) to be loaded at startup.
This file must be located in the **same directory** as `radar_module.py`.

Serial ports and baud rates are defined in `radar_module.py`:
```python
CFG_PORT  = "COM4"
DATA_PORT = "COM5"
BAUD_CFG  = 115200
BAUD_DATA = 921600
