# camera_module.py
#
# Improved version for capstone:
# - Detects person using MediaPipe PoseLandmarker
# - Measures distance using RealSense depth
# - Adds smoothing and temporal filtering
# - Shows visual cues
# - Returns: { "person": bool, "distance": float or None, "confidence": float }

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# Global variables
pipeline = None
align = None
pose_landmarker = None
distance_buffer = deque(maxlen=5)
person_counter = 0
CONFIDENCE_THRESHOLD = 3  # frames to confirm person detected


def initialize(model_path="pose_landmarker_lite.task"):
    """Start RealSense camera stream and PoseLandmarker"""
    global pipeline, align, pose_landmarker

    print("Initializing RealSense camera...")

    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Load PoseLandmarker model
    base_options = python.BaseOptions(model_asset_path=model_path)
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
      { "person": bool, "distance": float or None, "confidence": float }
    Also displays the video window with bounding box and distance overlay.
    """
    global distance_buffer, person_counter

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
        # Nose = landmark 0
        nose = result.pose_landmarks[0][0]
        cx = int(nose.x * w)
        cy = int(nose.y * h)

        # Clamp
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        # Read depth
        raw_distance = depth_frame.get_distance(cx, cy)
        if raw_distance <= 0 or raw_distance > 10:
            raw_distance = None

        # Smooth distance buffer
        if raw_distance is not None:
            distance_buffer.append(raw_distance)
            person_distance = sum(distance_buffer) / len(distance_buffer)
        else:
            person_distance = None

        # Temporal confirmation
        person_counter += 1
        if person_counter >= CONFIDENCE_THRESHOLD:
            person_present = True
        else:
            person_present = False

        # Draw on image
        cv2.circle(color_img, (cx, cy), 6, (0, 0, 255), -1)
        if person_distance is not None:
            text = f"Distance: {person_distance:.2f} m"
            cv2.putText(color_img, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(color_img, "Person: DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # Reset counter if no person
        person_counter = 0
        distance_buffer.clear()
        cv2.putText(color_img, "Person: NOT DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Confidence = 0.0 to 1.0 based on counter
    confidence = min(1.0, person_counter / CONFIDENCE_THRESHOLD)

    # Show window
    cv2.imshow("RealSense Camera", color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "QUIT"

    return {
        "person": person_present,
        "distance": person_distance,
        "confidence": confidence
    }
