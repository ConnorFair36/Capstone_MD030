# camera_module.py
#
# Uses RealSense + MediaPipe Tasks PoseLandmarker
#
# Returns:
# {
#   "person": bool,
#   "distance": float or None,
#   "confidence": float,   # 0.0 to 1.0
#   "low_light": bool,
#   "bad_weather": bool    # placeholder heuristic / manual flag for now
# }
#
# Usage:
#   initialize()
#   data = get_camera_data()
#   shutdown()

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pipeline = None
align = None
pose_landmarker = None

# -----------------------------
# TUNABLES
# -----------------------------
LOW_LIGHT_THRESH = 55.0     # average grayscale brightness threshold
MIN_VALID_DEPTH = 0.1       # meters
MAX_VALID_DEPTH = 10.0      # meters

# Manual placeholder for demo/testing.
# If you later have an environment node or weather input, replace this.
MANUAL_BAD_WEATHER = False


def initialize():
    """Start RealSense camera stream and pose landmarker."""
    global pipeline, align, pose_landmarker

    print("Initializing RealSense camera...")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    print("Camera ready.")


def shutdown():
    """Stop camera and close display window."""
    global pipeline

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


def _clamp(val, low, high):
    return max(low, min(val, high))


def _compute_low_light(color_img):
    """
    Simple brightness-based low-light check.
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    low_light = mean_brightness < LOW_LIGHT_THRESH
    return low_light, mean_brightness


def _depth_at_pixel(depth_frame, x, y, window=3):
    """
    Get a more stable depth by taking the median over a small window
    around the landmark pixel.
    """
    xs = []
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            px = _clamp(x + dx, 0, w - 1)
            py = _clamp(y + dy, 0, h - 1)
            d = depth_frame.get_distance(px, py)
            if MIN_VALID_DEPTH <= d <= MAX_VALID_DEPTH:
                xs.append(d)

    if len(xs) == 0:
        return None

    return float(np.median(xs))


def _get_landmark_pixel(landmark, img_w, img_h):
    x = int(landmark.x * img_w)
    y = int(landmark.y * img_h)
    x = _clamp(x, 0, img_w - 1)
    y = _clamp(y, 0, img_h - 1)
    return x, y


def _extract_person_distance_and_confidence(result, depth_frame, color_w, color_h):
    """
    Use several upper-body landmarks for a more stable depth estimate.
    Also compute a simple confidence from landmark visibility.
    """
    if not result.pose_landmarks:
        return False, None, 0.0, []

    pose = result.pose_landmarks[0]

    # MediaPipe pose landmark indices commonly used:
    # 0 nose, 11 left shoulder, 12 right shoulder, 23 left hip, 24 right hip
    candidate_indices = [0, 11, 12, 23, 24]

    depth_samples = []
    vis_samples = []
    used_points = []

    for idx in candidate_indices:
        if idx >= len(pose):
            continue

        lm = pose[idx]
        px, py = _get_landmark_pixel(lm, color_w, color_h)
        d = _depth_at_pixel(depth_frame, px, py, window=2)

        # visibility may not always be present depending on runtime result,
        # so default safely
        visibility = getattr(lm, "visibility", 0.5)

        if d is not None:
            depth_samples.append(d)
            vis_samples.append(float(visibility))
            used_points.append((idx, px, py, d, float(visibility)))

    if len(depth_samples) == 0:
        return False, None, 0.0, []

    # median depth is more robust than one landmark
    person_distance = float(np.median(depth_samples))

    # simple confidence:
    # more valid points + higher visibility -> higher confidence
    mean_vis = float(np.mean(vis_samples)) if vis_samples else 0.0
    point_factor = min(len(depth_samples) / 5.0, 1.0)
    confidence = 0.6 * mean_vis + 0.4 * point_factor
    confidence = float(max(0.0, min(confidence, 1.0)))

    return True, person_distance, confidence, used_points


def get_camera_data():
    """
    Reads camera frame and returns:
    {
      "person": bool,
      "distance": float or None,
      "confidence": float,
      "low_light": bool,
      "bad_weather": bool
    }

    Also displays the video window with overlay.
    Press 'q' in the image window to return "QUIT".
    """
    global pipeline, align, pose_landmarker

    if pipeline is None or align is None or pose_landmarker is None:
        raise RuntimeError("Camera not initialized. Call initialize() first.")

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    h, w, _ = color_img.shape

    low_light, brightness = _compute_low_light(color_img)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=color_img
    )

    result = pose_landmarker.detect(mp_image)

    person_present, person_distance, confidence, used_points = _extract_person_distance_and_confidence(
        result, depth_frame, w, h
    )

    bad_weather = MANUAL_BAD_WEATHER

    # -----------------------------
    # DRAW OVERLAY
    # -----------------------------
    if person_present:
        for idx, px, py, d, vis in used_points:
            cv2.circle(color_img, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                color_img,
                f"{idx}:{d:.2f}m",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1
            )

        cv2.putText(
            color_img,
            f"Person: DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            color_img,
            f"Distance: {person_distance:.2f} m",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            color_img,
            f"Confidence: {confidence:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            color_img,
            "Person: NOT DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    cv2.putText(
        color_img,
        f"Brightness: {brightness:.1f}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        color_img,
        f"Low light: {low_light}",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if low_light else (255, 255, 255),
        2
    )

    cv2.putText(
        color_img,
        f"Bad weather: {bad_weather}",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255) if bad_weather else (255, 255, 255),
        2
    )

    cv2.imshow("RealSense Camera", color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "QUIT"

    return {
        "person": bool(person_present),
        "distance": float(person_distance) if person_distance is not None else None,
        "confidence": float(confidence),
        "low_light": bool(low_light),
        "bad_weather": bool(bad_weather)
    }
