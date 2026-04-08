# camera_module.py
#
# Custom YOLO + RealSense camera module for ROS bridge use.
#
# Detects:
#   - person
#   - chair
#   - laptop
#   - phone / cell phone
#
# Returns:
# {
#   "person": bool,
#   "chair": bool,
#   "laptop": bool,
#   "phone": bool,
#   "distance": float or None,          # primary target distance
#   "confidence": float,                # primary target confidence
#   "label": str or None,               # primary target label
#   "detections": list,                 # all accepted detections
#   "low_light": bool,
#   "bad_weather": bool
# }
#
# Public functions:
#   initialize()
#   get_camera_data()
#   shutdown()

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

pipeline = None
align = None
model = None

LOW_LIGHT_THRESHOLD = 55.0
CONF_THRESHOLD = 0.50
BAD_WEATHER_DEFAULT = False


def initialize(model_path="best.pt"):
    """Load custom YOLO model and start RealSense streams."""
    global pipeline, align, model

    print(f"Initializing YOLO model from: {model_path}")
    model = YOLO(model_path)

    print("Model class names:", model.names)

    print("Initializing RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("Camera ready.")


def shutdown():
    """Stop camera pipeline and close OpenCV windows."""
    global pipeline, align, model

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass

    pipeline = None
    align = None
    model = None


def _compute_low_light(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < LOW_LIGHT_THRESHOLD, mean_brightness


def _safe_depth(depth_frame, cx, cy, window=2):
    """
    Get a more stable depth by sampling a small window around the center point
    and taking the median of valid depths.
    """
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    samples = []
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            px = max(0, min(cx + dx, w - 1))
            py = max(0, min(cy + dy, h - 1))
            d = depth_frame.get_distance(px, py)
            if 0.1 <= d <= 20.0:
                samples.append(d)

    if not samples:
        return None

    return float(np.median(samples))


def _normalize_label(label):
    return str(label).strip().lower()


def _get_primary_detection(detections):
    """
    Pick one primary detection for summary output.
    Priority:
      1) valid depth
      2) closer distance
      3) higher confidence
    """
    if not detections:
        return None

    best = None
    best_metric = None

    for det in detections:
        dist = det["distance"]
        conf = det["confidence"]

        if dist is None:
            metric = (1, -conf)
        else:
            metric = (0, dist, -conf)

        if best is None or metric < best_metric:
            best = det
            best_metric = metric

    return best


def get_camera_data():
    """
    Read one frame and return camera data for the ROS bridge.

    Returns:
      None   -> frame not ready
      "QUIT" -> user pressed q
      dict   -> camera data
    """
    global pipeline, align, model

    if pipeline is None or align is None or model is None:
        raise RuntimeError("Camera module not initialized. Call initialize() first.")

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    low_light, brightness = _compute_low_light(color_img)

    results = model(color_img, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            raw_label = model.names.get(cls_id, str(cls_id))

            # Ignore classes outside our target set
            if target_label is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(color_img.shape[1] - 1, x2)
            y2 = min(color_img.shape[0] - 1, y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            dist = _safe_depth(depth_frame, cx, cy, window=2)

            detections.append({
                "label": target_label,
                "raw_label": str(raw_label),
                "confidence": conf,
                "distance": dist,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
            })

    person_found = any(d["label"] ==
