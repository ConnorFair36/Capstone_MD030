# camera_module.py
#
# Uses MediaPipe Tasks PoseLandmarker (NOT mp.solutions)
# Returns:
#   { "person": bool, "distance": float or None }

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pipeline = None
align = None
pose_landmarker = None


def initialize():
    """Start RealSense camera stream and pose landmarker."""
    global pipeline, align, pose_landmarker

    print("Initializing RealSense camera...")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # MediaPipe PoseLandmarker
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


def get_camera_data():
    """
    Reads camera frame and returns:
      { "person": bool, "distance": float or None }
    Also displays the video window with distance overlay.
    """

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    h, w, _ = color_img.shape

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=color_img
    )

    result = pose_landmarker.detect(mp_image)

    person_present = False
    person_distance = None

    if result.pose_landmarks:
        person_present = True

        # Nose landmark = index 0
        nose = result.pose_landmarks[0][0]

        cx = int(nose.x * w)
        cy = int(nose.y * h)

        # Clamp to image bounds
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        person_distance = depth_frame.get_distance(cx, cy)

        # Draw marker + distance
        cv2.circle(color_img, (cx, cy), 6, (0, 0, 255), -1)

        text = f"Distance: {person_distance:.2f} m"
        cv2.putText(
            color_img,
            text,
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

    cv2.imshow("RealSense Camera", color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "QUIT"

    return {
        "person": person_present,
        "distance": person_distance
    }
