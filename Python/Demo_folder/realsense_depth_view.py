import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    # --- Configure RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams (D455 supports 848x480, 640x480, etc.)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get depth scale (meters per unit)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} meters per unit")

    print("Press 'q' to quit")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # Align depth to color frame
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Colorize depth for display (nice false color)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Get center pixel distance
            h, w = depth_image.shape
            cx, cy = w // 2, h // 2
            depth_value = depth_frame.get_distance(cx, cy)  # in meters

            # Draw a small circle at the center
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), 2)
            cv2.circle(depth_colormap, (cx, cy), 5, (255, 255, 255), 2)

            # Put text showing distance
            text = f"Center distance: {depth_value:.2f} m"
            cv2.putText(color_image, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(depth_colormap, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show both windows
            cv2.imshow('RealSense Color', color_image)
            cv2.imshow('RealSense Depth', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
