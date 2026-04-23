import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import os
import time

# =========================
# User-configurable params
# =========================
FPS = 10
DURATION_SEC = 60            # set None to run until 'q'
OUTPUT_DIR = "camera_rgb"
CSV_FILE = "camera_log.csv"

# =========================
# Setup output directories
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Initialize RealSense
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, rs.format.bgr8, FPS)

pipeline.start(config)

print("RealSense started.")
print("Logging RGB frames...")
print("Press 'q' to stop early.")

# =========================
# CSV setup
# =========================
csv_path = CSV_FILE
csv_exists = os.path.exists(csv_path)

csv_file = open(csv_path, "a", newline="")
csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow([
        "timestamp_ms",
        "frame_id",
        "rgb_filename"
    ])

# =========================
# Capture loop
# =========================
frame_id = 0
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        timestamp_ms = int(time.time() * 1000)
        frame_id += 1

        filename = f"frame_{frame_id:06d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(filepath, color_image)

        csv_writer.writerow([
            timestamp_ms,
            frame_id,
            filename
        ])
        csv_file.flush()

        cv2.imshow("RGB Capture", color_image)

        # Stop conditions
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

        if DURATION_SEC is not None:
            if time.time() - start_time >= DURATION_SEC:
                print("Reached capture duration.")
                break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    pipeline.stop()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Capture complete.")
