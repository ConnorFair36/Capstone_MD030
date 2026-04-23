# Experiments (Archived Code)

This folder contains previous iterations, prototypes, and supporting tools developed during the project. The **current production code** lives in `../src/`.

## Contents

| Folder | Description |
|---|---|
| `Fusion_Demo_1/` | Earliest monolithic fusion prototype (no ROS, no GUI) |
| `Fusion_Demo_2/` | Data logging scripts + ~500 captured RGB/depth frame pairs |
| `Fusion_Demo_3/` | More data logging experiments + ~590 captured frame pairs |
| `THE_LATEST/` | Intermediate raw-data variant of the pipeline (`_raw` modules) |
| `Yo/` | Scratch scripts for radar-to-camera projection testing |
| `ros_prac/` | ROS learning and practice scripts (rosbridge experiments, test publishers) |
| `fusion_demo_04_14_2026/` | Previous week's fusion demo (superseded by the current `src/`) |
| `Machine_Learning/` | CenterFusion-style training stack (DLA backbone, nuScenes radar fusion) |
| `Raw_Data_Collection/` | Standalone sensor logging tools for RealSense and mmWave radar |
| `Environment_Setup.md` | Original environment setup notes |
| `old_readmes/` | README files from the previous directory structure |

## Notes

- `Machine_Learning/` and `Raw_Data_Collection/` are independent from the live fusion system and have no code dependencies on `src/`.
- `Fusion_Demo_2/` and `Fusion_Demo_3/` contain large image datasets. Consider removing them if repo size is a concern.
