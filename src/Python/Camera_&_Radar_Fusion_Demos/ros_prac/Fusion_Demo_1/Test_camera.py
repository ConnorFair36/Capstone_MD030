# camera_module.py
#
# RealSense + YOLO camera module using a custom trained model (best.pt).
#
# The model itself decides what objects to classify.
# This module:
#   - loads best.pt
#   - runs inference on RealSense color frames
#   - draws bounding boxes and labels
#   - estimates depth at the box center
#   - returns all detections for ROS use
#
# Public functions:
#   initialize(model_path="best.pt")
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
    """Load YOLO model and start RealSense streams."""
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

    # Run YOLO on current frame
    results = model(color_img, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            label = str(model.names.get(cls_id, cls_id))

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(color_img.shape[1] - 1, x2)
            y2 = min(color_img.shape[0] - 1, y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            dist = _safe_depth(depth_frame, cx, cy, window=2)

            detections.append({
                "label": label,
                "confidence": conf,
                "distance": dist,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
            })

    primary = _get_primary_detection(detections)

    primary_distance = None
    primary_conf = 0.0
    primary_label = None

    # Draw all detections
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cx, cy = det["cx"], det["cy"]
        label = det["label"]
        conf = det["confidence"]
        dist = det["distance"]

        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(color_img, (cx, cy), 4, (0, 0, 255), -1)

        dist_text = "N/A" if dist is None else f"{dist:.2f} m"
        label_text = f"{label} {conf:.2f} | {dist_text}"

        cv2.putText(
            color_img,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    if primary is not None:
        primary_distance = primary["distance"]
        primary_conf = float(primary["confidence"])
        primary_label = primary["label"]

        cv2.putText(
            color_img,
            f"Primary: {primary_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            color_img,
            "Primary: NONE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        color_img,
        f"Low light: {low_light}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if low_light else (255, 255, 255),
        2
    )

    cv2.putText(
        color_img,
        f"Brightness: {brightness:.1f}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Temporary local display for testing
    cv2.imshow("RealSense + best.pt Detection", color_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        return "QUIT"

    return {
        "detected": bool(len(detections) > 0),
        "label": primary_label,
        "distance": float(primary_distance) if primary_distance is not None else None,
        "confidence": float(primary_conf),
        "detections": [
            {
                "label": d["label"],
                "confidence": float(d["confidence"]),
                "distance": float(d["distance"]) if d["distance"] is not None else None,
                "bbox": [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])],
                "center": [int(d["cx"]), int(d["cy"])],
            }
            for d in detections
        ],
        "low_light": bool(low_light),
        "bad_weather": bool(BAD_WEATHER_DEFAULT),
    }


if __name__ == "__main__":
    initialize("best.pt")
    try:
        print("Press 'q' to quit")
        while True:
            data = get_camera_data()
            if data == "QUIT":
                break
            if data is not None:
                print(data)
    finally:
        shutdown()
