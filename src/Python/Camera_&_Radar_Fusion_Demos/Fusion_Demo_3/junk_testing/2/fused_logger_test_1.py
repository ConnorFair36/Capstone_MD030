import struct
import time
import csv
import threading
import queue
from pathlib import Path

import numpy as np
import cv2
import serial
import pyrealsense2 as rs


# =========================
# SETTINGS (edit as needed)
# =========================
# Camera
FPS = 30
DURATION_SEC = 20  # None = until 'q'
OUT_RGB_DIR = "rgb"
OUT_DEPTH_DIR = "depth"
CAMERA_CSV_NAME = "camera_frames.csv"
SYNC_CSV_NAME = "sync_frames.csv"
DEPTH_SCALE_TXT = "depth_scale.txt"
CALIB_JSON = "calibration.json"

# Radar (Windows example uses COM ports; on Pi/Linux use /dev/ttyACM0, /dev/ttyUSB0, etc.)
RADAR_ENABLE = True
CFG_PORT = "COM3"          # e.g. "COM3" or "/dev/ttyACM0"
DATA_PORT = "COM4"         # e.g. "COM4" or "/dev/ttyACM1" or "/dev/ttyUSB0"
CFG_FILE = "profile.cfg"
BAUD_CFG = 115200
BAUD_DATA = 921600
RADAR_RAW_BIN = "radar_raw.bin"
RADAR_FRAMES_CSV = "radar_frames.csv"

# Radar packet parsing
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 8 * 4  # 8 uint32 after magic
# =========================


def now_ms() -> int:
    return time.time_ns() // 1_000_000


# --------------------------
# RealSense intrinsics/extrinsics helpers
# --------------------------
def intrinsics_to_dict(intr: rs.intrinsics):
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def extrinsics_4x4(extr: rs.extrinsics) -> np.ndarray:
    R = np.array(extr.rotation, dtype=float).reshape(3, 3)  # row-major 3x3
    t = np.array(extr.translation, dtype=float).reshape(3, 1)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


# --------------------------
# Radar config + packet parsing (adapted from your radar_logger_with_timestamp.py)
# --------------------------
def send_cfg(cfg_ser: serial.Serial, cfg_file_path: Path) -> None:
    """Send .cfg lines to the radar, like the TI visualizer does."""
    with open(cfg_file_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("%")]

    for line in lines:
        cfg_ser.write((line + "\n").encode("utf-8"))
        time.sleep(0.05)


def find_magic(buf: bytearray) -> int:
    return buf.find(MAGIC_WORD)


def try_extract_packet(buf: bytearray):
    """
    If a full packet exists at the start of buf, return (frame_num, total_packet_len).
    Otherwise return (None, None).
    """
    if len(buf) < len(MAGIC_WORD) + HEADER_LEN:
        return None, None

    header_start = len(MAGIC_WORD)
    header = buf[header_start:header_start + HEADER_LEN]

    try:
        _, total_packet_len, _, frame_num, _, _, _, _ = struct.unpack("<8I", header)
    except struct.error:
        return None, None

    if total_packet_len <= 0:
        return None, None

    if len(buf) < total_packet_len:
        return None, None

    return frame_num, total_packet_len


class RadarLoggerThread:
    """
    Background radar logger:
      - Sends config once
      - Logs raw bytes continuously to radar_raw.bin
      - Detects full packets and writes (host_timestamp_ms, frame_num, pkt_len, bin_offset_end) to radar_frames.csv
      - Also pushes frame events to a queue so the camera loop can "sync" (count radar frames between camera frames)
    """
    def __init__(self, base_dir: Path, frame_event_q: queue.Queue):
        self.base_dir = base_dir
        self.frame_event_q = frame_event_q
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.cfg_path = (base_dir / CFG_FILE).resolve()
        self.out_bin_path = (base_dir / RADAR_RAW_BIN).resolve()
        self.out_csv_path = (base_dir / RADAR_FRAMES_CSV).resolve()

    def start(self):
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"CFG file not found: {self.cfg_path}")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        print(f"[RADAR] Opening CFG port {CFG_PORT} @ {BAUD_CFG}")
        cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)

        print(f"[RADAR] Opening DATA port {DATA_PORT} @ {BAUD_DATA}")
        data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)

        time.sleep(0.5)

        print(f"[RADAR] Sending config file: {self.cfg_path.name}")
        send_cfg(cfg_ser, self.cfg_path)
        print("[RADAR] Config sent. Radar should now be running.")
        cfg_ser.close()

        csv_exists = self.out_csv_path.exists()
        buf = bytearray()
        bin_offset = 0

        with open(self.out_csv_path, "a", newline="") as fcsv, open(self.out_bin_path, "wb") as fbin:
            writer = csv.writer(fcsv)
            if not csv_exists:
                writer.writerow(["host_timestamp_ms", "radar_frame_num", "packet_len_bytes", "bin_offset_end"])

            try:
                while not self._stop_evt.is_set():
                    chunk = data_ser.read(4096)
                    if chunk:
                        fbin.write(chunk)
                        bin_offset += len(chunk)
                        buf.extend(chunk)

                    # parse as many packets as are available
                    while True:
                        magic_idx = find_magic(buf)
                        if magic_idx < 0:
                            # keep tail to avoid losing partial magic word
                            if len(buf) > len(MAGIC_WORD):
                                buf = buf[-len(MAGIC_WORD):]
                            break

                        if magic_idx > 0:
                            buf = buf[magic_idx:]

                        frame_num, pkt_len = try_extract_packet(buf)
                        if frame_num is None:
                            break

                        t_ms = now_ms()
                        writer.writerow([t_ms, frame_num, pkt_len, bin_offset])
                        fcsv.flush()

                        # Push event to queue for sync
                        # (t_ms, frame_num)
                        try:
                            self.frame_event_q.put_nowait((t_ms, frame_num))
                        except queue.Full:
                            # If queue is full, drop sync events (radar_csv still has truth)
                            pass

                        # Remove packet from buffer
                        del buf[:pkt_len]

            except Exception as e:
                print(f"[RADAR] Error: {e}")

            finally:
                data_ser.close()
                print("[RADAR] Ports closed. Radar thread exiting.")


def main():
    base_dir = Path(__file__).resolve().parent

    rgb_dir = base_dir / OUT_RGB_DIR
    depth_dir = base_dir / OUT_DEPTH_DIR
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Start RealSense
    # --------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # Your original script used enable_stream(stream, format, FPS) :contentReference[oaicite:2]{index=2}
    # Here we keep that style (RealSense picks default resolution); you can also specify width/height if you want.
    config.enable_stream(rs.stream.color, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # Depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    (base_dir / DEPTH_SCALE_TXT).write_text(f"{depth_scale}\n")

    # Intrinsics + extrinsics collection (WHAT YOUR GRAD STUDENT WANTS)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    d2c = depth_stream.get_extrinsics_to(color_stream)
    T_d2c = extrinsics_4x4(d2c)

    calib = {
        "color_intrinsics": intrinsics_to_dict(color_intr),
        "depth_intrinsics": intrinsics_to_dict(depth_intr),
        "depth_to_color_extrinsics_4x4": T_d2c.tolist(),
        "depth_scale_m_per_unit": depth_scale,
        "notes": "Depth PNG stores uint16 depth units. Multiply by depth_scale_m_per_unit to get meters."
    }
    (base_dir / CALIB_JSON).write_text(__import__("json").dumps(calib, indent=2))

    print("[CAM] RealSense started (RGB + Depth).")
    print(f"[CAM] Saved: {DEPTH_SCALE_TXT} and {CALIB_JSON}")
    print("[CAM] Press 'q' to stop early.")

    # --------------------------
    # Start Radar thread (optional)
    # --------------------------
    radar_event_q: queue.Queue = queue.Queue(maxsize=10000)
    radar_thread = None

    if RADAR_ENABLE:
        radar_thread = RadarLoggerThread(base_dir=base_dir, frame_event_q=radar_event_q)
        radar_thread.start()
        print(f"[RADAR] Logging raw bytes to: {RADAR_RAW_BIN}")
        print(f"[RADAR] Logging frame timestamps to: {RADAR_FRAMES_CSV}")

    # --------------------------
    # CSV setup
    # --------------------------
    cam_csv_path = base_dir / CAMERA_CSV_NAME
    cam_csv_exists = cam_csv_path.exists()

    sync_csv_path = base_dir / SYNC_CSV_NAME
    sync_csv_exists = sync_csv_path.exists()

    frame_id = 0
    start_time = time.time()

    # Track radar events since last camera frame (for a simple sync artifact)
    last_cam_t_ms = None

    with open(cam_csv_path, "a", newline="") as fcam, open(sync_csv_path, "a", newline="") as fsync:
        cam_writer = csv.writer(fcam)
        if not cam_csv_exists:
            cam_writer.writerow([
                "host_timestamp_ms",
                "frame_id",
                "rgb_filename",
                "depth_filename",
                "depth_scale_m_per_unit"
            ])

        sync_writer = csv.writer(fsync)
        if not sync_csv_exists:
            sync_writer.writerow([
                "camera_frame_id",
                "camera_host_timestamp_ms",
                "radar_frames_since_prev_camera",
                "last_radar_frame_num_seen",
                "last_radar_host_timestamp_ms"
            ])

        try:
            while True:
                frameset = pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                frame_id += 1
                cam_t_ms = now_ms()

                color_img = np.asanyarray(color_frame.get_data())  # BGR8
                depth_img = np.asanyarray(depth_frame.get_data())  # uint16 depth units

                rgb_name = f"frame_{frame_id:06d}.jpg"
                depth_name = f"frame_{frame_id:06d}.png"

                rgb_path = rgb_dir / rgb_name
                depth_path = depth_dir / depth_name

                cv2.imwrite(str(rgb_path), color_img)
                cv2.imwrite(str(depth_path), depth_img)

                cam_writer.writerow([cam_t_ms, frame_id, f"{OUT_RGB_DIR}/{rgb_name}", f"{OUT_DEPTH_DIR}/{depth_name}", depth_scale])
                fcam.flush()

                # ---- Sync logging: drain radar events up to "now" ----
                radar_count = 0
                last_radar_frame = None
                last_radar_t_ms = None

                if RADAR_ENABLE:
                    # Drain everything currently available. This does NOT perfectly align frames,
                    # but it gives you a simple "what arrived between camera frames" record.
                    while True:
                        try:
                            rt_ms, rframe = radar_event_q.get_nowait()
                        except queue.Empty:
                            break
                        radar_count += 1
                        last_radar_frame = rframe
                        last_radar_t_ms = rt_ms

                sync_writer.writerow([
                    frame_id,
                    cam_t_ms,
                    radar_count,
                    last_radar_frame if last_radar_frame is not None else "",
                    last_radar_t_ms if last_radar_t_ms is not None else ""
                ])
                fsync.flush()

                # Show RGB
                cv2.imshow("RGB Capture", color_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[CAM] Stopped by user.")
                    break

                if DURATION_SEC is not None and (time.time() - start_time) >= DURATION_SEC:
                    print("[CAM] Reached capture duration.")
                    break

                last_cam_t_ms = cam_t_ms

        except KeyboardInterrupt:
            print("[SYS] Interrupted by user (Ctrl+C).")

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("[CAM] Capture complete.")

            if radar_thread is not None:
                radar_thread.stop()
                print("[RADAR] Radar stopped.")


if __name__ == "__main__":
    main()
