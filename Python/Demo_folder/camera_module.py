# camera_module.py
#
# Provides a function:
#   get_camera_data(color_img)
# Returns:
#   { "person": bool, "distance": float or None }
#
# Also shows the camera window with bounding box and distance display.
# Call initialize() once, then repeatedly call get_camera_data().

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = None
pipeline = None
align = None

def initialize():
    """Start RealSense camera stream."""
    global pipeline, align, pose
    print("Initializing RealSense camera...")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    pose = mp_pose.Pose(static_image_mode=False)

    print("Camera ready.")

def get_camera_data():
    """
    Reads camera frame and returns:
      { "person": bool, "distance": float or None }
    Also displays the video window with bounding box and distance overlay.
    """

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())

    # Run pose detection
    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    person_present = False
    person_distance = None

    if results.pose_landmarks:
        person_present = True

        # Use nose landmark as center reference
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        h, w, _ = color_img.shape
        cx, cy = int(nose.x * w), int(nose.y * h)

        # depth at the nose pixel
        person_distance = depth_frame.get_distance(cx, cy)

        # draw crosshair and distance
        cv2.circle(color_img, (cx, cy), 6, (0, 0, 255), -1)

        text = f"Distance: {person_distance:.2f} m"
        cv2.putText(color_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # if no person detected
        cv2.putText(color_img, "Person: NOT DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show window
    cv2.imshow("RealSense Camera", color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "QUIT"

    return {
        "person": person_present,
        "distance": person_distance
    }
