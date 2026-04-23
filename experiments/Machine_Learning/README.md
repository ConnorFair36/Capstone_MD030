# Machine Learning (Planned)

This directory is reserved for future machine learning development using the
datasets collected in this project.

At the current stage, this folder serves as a placeholder to document the
intended learning pipeline and data interfaces. No trained models or finalized
ML implementations are included yet.

---

## Intended Purpose

Machine learning methods may be explored to:
- classify objects or motion patterns using mmWave radar data
- perform person detection or scene understanding using camera data
- investigate multi-modal learning using synchronized radar and camera inputs

---

## Planned Inputs

Potential ML inputs include:
- radar point cloud data (CSV format)
- raw RGB camera frames
- camera metadata logs (timestamps, bounding boxes, distance estimates)
- synchronized radar–camera observations

These datasets are collected and stored in the `Raw_Data_Collection/` directory.

---

## Planned Outputs

Potential ML outputs may include:
- object or motion classification labels
- detection confidence scores
- learned feature representations for sensor fusion

---

## Notes

- ML development will be pursued after sufficient datasets are collected and
  labeled.
- The exact models, frameworks, and training strategies are intentionally left
  open at this stage.
- This folder exists to clearly separate data collection, sensor processing,
  and learning-based experimentation.
