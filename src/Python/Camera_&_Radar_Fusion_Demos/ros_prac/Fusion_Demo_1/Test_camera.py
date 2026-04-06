# camera_module.py
#
# YOLO + RealSense camera module for ROS bridge use.
#
# Returns:
# {
#   "person": bool,
#   "distance": float or None,
#   "confidence": float,
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


def initialize():
    """Load YOLO model and start RealSense streams."""
    global pipeline, align, model

    print("Initializing YOLO model...")
    model = YOLO("yolov8n.pt")

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


def get_camera_data():
    """
    Read one frame and return camera data for the ROS bridge.

    Returns:
      None -> frame not ready
      "QUIT" -> user pressed q
      dict -> camera data
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

    # Run YOLO
    results = model(color_img, verbose=False)

    best_person = None
    best_metric = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class 0 = person
            if cls != 0 or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(color_img.shape[1] - 1, x2)
            y2 = min(color_img.shape[0] - 1, y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            dist = _safe_depth(depth_frame, cx, cy, window=2)

            # Skip invalid depth for primary selection if you want tighter behavior
            # but still allow fallback if needed.
            if dist is None:
                metric = (1, -conf)   # worse than valid depth
            else:
                metric = (0, dist)    # prefer valid/closer person

            if best_person is None or metric < best_metric:
                best_metric = metric
                best_person = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "distance": dist,
                    "confidence": conf,
                }

    person_found = best_person is not None
    person_distance = None
    person_conf = 0.0

    if person_found:
        x1 = best_person["x1"]
        y1 = best_person["y1"]
        x2 = best_person["x2"]
        y2 = best_person["y2"]
        cx = best_person["cx"]
        cy = best_person["cy"]
        person_distance = best_person["distance"]
        person_conf = float(best_person["confidence"])

        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)

        dist_text = "N/A" if person_distance is None else f"{person_distance:.2f} m"
        label = f"Person {person_conf:.2f} | {dist_text}"
        cv2.putText(
            color_img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            color_img,
            "Person: DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            color_img,
            "Person: NOT DETECTED",
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
        0.7,
        (0, 255, 255) if low_light else (255, 255, 255),
        2
    )

    cv2.putText(
        color_img,
        f"Brightness: {brightness:.1f}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("RealSense + YOLO Person Detection", color_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "QUIT"

    return {
        "person": bool(person_found),
        "distance": float(person_distance) if person_distance is not None else None,
        "confidence": float(person_conf),
        "low_light": bool(low_light),
        "bad_weather": bool(BAD_WEATHER_DEFAULT),
    }


if __name__ == "__main__":
    initialize()
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
