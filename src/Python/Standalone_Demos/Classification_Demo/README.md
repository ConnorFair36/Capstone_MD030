# Classification Demo

This folder contains standalone demos used to test real-time classification
behavior within the vision pipeline prior to full sensor fusion.

The classification demo was used to validate detection stability, update rate,
and basic decision logic (e.g., person present / not present) under different
indoor conditions.

---

## Purpose

This demo focuses on:
- running the vision pipeline in real time
- evaluating detection consistency across frames
- validating basic classification logic and thresholds
- testing the system’s responsiveness to movement and scene changes

This folder served as an experimental space for iterating on classification
logic before integrating radar information.

---

## Contents

This folder may include:
- scripts for live classification
- scripts for debugging classification outputs (e.g., confidence or labels)
- variants of the same demo used during iterative development

Script names may vary depending on development stage.

---

## Hardware Requirements

Depending on the script, this demo may require:
- Intel RealSense depth camera (preferred), or
- a standard webcam (for RGB-only testing)

---

## Running the Demo

From this directory:
```bash
python <script_name>.py
