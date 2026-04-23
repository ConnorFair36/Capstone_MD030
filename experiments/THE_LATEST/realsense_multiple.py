import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")   # small model, good starting point

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Align depth to color frame
    align = rs.align(rs.stream.color)

    print("Press 'q' to quit")

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert frame to numpy array
            color_img = np.asanyarray(color_frame.get_data())

            # Run YOLO on the color image
            results = model(color_img, verbose=False)

            person_found = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])      # class ID
                    conf = float(box.conf[0])  # confidence score

                    # COCO class 0 = person
                    if cls == 0 and conf > 0.5:
                        person_found = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Clamp coordinates so they stay inside the image
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(color_img.shape[1] - 1, x2)
                        y2 = min(color_img.shape[0] - 1, y2)

                        # Center point of bounding box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Get distance at center point
                        dist = depth_frame.get_distance(cx, cy)

                        # Draw bounding box
                        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw center point
                        cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)

                        # Put label
                        label = f"Person {conf:.2f} | {dist:.2f} m"
                        cv2.putText(
                            color_img,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

            if not person_found:
                cv2.putText(
                    color_img,
                    "Person: NOT DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("RealSense + YOLO Person Detection", color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()