import time
import csv
import os
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs


# ====== SETTINGS ======
FPS = 30
DURATION_SEC = 20          # set None to run until 'q'
OUT_RGB_DIR = "rgb"
OUT_DEPTH_DIR = "depth"
CSV_NAME = "camera_frames.csv"
DEPTH_SCALE_TXT = "depth_scale.txt"
# ======================


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def main():
    base_dir = Path(__file__).resolve().parent
    rgb_dir = base_dir / OUT_RGB_DIR
    depth_dir = base_dir / OUT_DEPTH_DIR
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / CSV_NAME
    csv_exists = csv_path.exists()

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable BOTH streams
    config.enable_stream(rs.stream.color, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # Get depth scale (to convert raw uint16 to meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit
    (base_dir / DEPTH_SCALE_TXT).write_text(f"{depth_scale}\n")

    print("RealSense started (RGB + Depth).")
    print(f"Depth scale saved to: {base_dir / DEPTH_SCALE_TXT}")
    print("Press 'q' to stop early.")

    frame_id = 0
    start_time = time.time()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "host_timestamp_ms",
                "frame_id",
                "rgb_filename",
                "depth_filename",
                "depth_scale_m_per_unit"
            ])

        try:
            while True:
                frameset = pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                frame_id += 1
                t_ms = now_ms()

                color_img = np.asanyarray(color_frame.get_data())        # BGR8
                depth_img = np.asanyarray(depth_frame.get_data())        # uint16 (raw depth units)

                rgb_name = f"frame_{frame_id:06d}.jpg"
                depth_name = f"frame_{frame_id:06d}.png"                # 16-bit PNG

                rgb_path = rgb_dir / rgb_name
                depth_path = depth_dir / depth_name

                cv2.imwrite(str(rgb_path), color_img)
                cv2.imwrite(str(depth_path), depth_img)                 # writes 16-bit PNG

                writer.writerow([t_ms, frame_id, f"{OUT_RGB_DIR}/{rgb_name}", f"{OUT_DEPTH_DIR}/{depth_name}", depth_scale])
                f.flush()

                # Display RGB for quick sanity check
                cv2.imshow("RGB Capture", color_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

                if DURATION_SEC is not None and (time.time() - start_time) >= DURATION_SEC:
                    print("Reached capture duration.")
                    break

        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl+C).")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Capture complete.")


if __name__ == "__main__":
    main()
