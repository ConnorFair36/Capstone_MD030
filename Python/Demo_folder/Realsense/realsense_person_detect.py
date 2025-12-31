import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    pose = mp_pose.Pose(static_image_mode=False)

    print("Press 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            # Run pose detection on RGB image
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            person_dist = None

            if results.pose_landmarks:
                # Get coordinates for the nose landmark (center of body)
                # You can change to chest, hips, etc later
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                h, w, _ = color_img.shape
                cx, cy = int(nose.x * w), int(nose.y * h)

                # Draw landmark
                cv2.circle(color_img, (cx, cy), 6, (0, 0, 255), -1)

                # Get depth at this pixel
                person_dist = depth_frame.get_distance(cx, cy)

                # Draw bounding box around upper body (simple estimate)
                box_w = 150
                box_h = 220
                x1, y1 = cx - box_w//2, cy - box_h//2
                x2, y2 = cx + box_w//2, cy + box_h//2
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Distance text
                cv2.putText(
                    color_img,
                    f"Distance: {person_dist:.2f} m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

            if person_dist is None:
                cv2.putText(color_img, "Person: NOT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("RealSense + Person Detection", color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
