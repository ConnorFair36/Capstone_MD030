import struct
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==== EDIT THESE IF NEEDED ====
CFG_PORT = "COM4"
DATA_PORT = "COM5"
BAUD_CFG = 115200
BAUD_DATA = 921600
CFG_FILE = "profile.cfg"
# ===============================

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DETECTED_POINTS_TLV_TYPE = 1

TRAIL_LEN = 60

# Doppler velocity thresholds
VEL_TOWARD_THRESH = -0.2   # m/s (negative = approaching)
VEL_AWAY_THRESH = +0.2     # m/s


def send_cfg(cfg_ser, cfg_file):
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("%")]
    for line in lines:
        cfg_ser.write((line + "\n").encode("utf-8"))
        time.sleep(0.05)


def find_magic(buf):
    return buf.find(MAGIC_WORD)


def parse_packet(packet_bytes):
    if len(packet_bytes) < len(MAGIC_WORD) + 8 * 4:
        return None, []

    header_start = len(MAGIC_WORD)
    header = packet_bytes[header_start:header_start + 8 * 4]
    (_, total_packet_len,
     _, frame_num,
     _, num_detected_obj,
     num_tlvs,
     _) = struct.unpack("<8I", header)

    points = []  # -> (x, y, z, v)

    tlv_idx = header_start + 8 * 4
    packet_end = total_packet_len

    for _ in range(num_tlvs):
        if tlv_idx + 8 > packet_end:
            break
        tlv_type, tlv_len = struct.unpack("<2I", packet_bytes[tlv_idx:tlv_idx + 8])
        tlv_value_start = tlv_idx + 8
        tlv_value_end = tlv_value_start + tlv_len
        if tlv_value_end > packet_end:
            break

        if tlv_type == DETECTED_POINTS_TLV_TYPE and num_detected_obj > 0:
            pts_bytes = packet_bytes[tlv_value_start:tlv_value_end]
            point_size = 16
            for obj_idx in range(num_detected_obj):
                start = obj_idx * point_size
                end = start + point_size
                x, y, z, v = struct.unpack("<4f", pts_bytes[start:end])
                points.append((x, y, z, v))

        tlv_idx = tlv_value_end

    return frame_num, points


def live_view():
    print(f"Opening CFG port {CFG_PORT} @ {BAUD_CFG}")
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)

    print(f"Opening DATA port {DATA_PORT} @ {BAUD_DATA}")
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)

    time.sleep(0.5)
    print(f"Sending config file: {CFG_FILE}")
    send_cfg(cfg_ser, CFG_FILE)
    cfg_ser.close()
    print("Config sent.\n")

    buf = bytearray()

    plt.ion()
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    centroid_dot, = ax.plot([], [], 'ro', markersize=8)
    trail_line, = ax.plot([], [], 'r-', linewidth=1)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 10)

    trail = deque(maxlen=TRAIL_LEN)

    try:
        while True:
            chunk = data_ser.read(4096)
            if chunk:
                buf.extend(chunk)

            while True:
                magic_idx = find_magic(buf)
                if magic_idx < 0:
                    if len(buf) > len(MAGIC_WORD):
                        buf = buf[-len(MAGIC_WORD):]
                    break

                if magic_idx > 0:
                    buf = buf[magic_idx:]

                if len(buf) < len(MAGIC_WORD) + 8 * 4:
                    break

                header_start = len(MAGIC_WORD)
                header = buf[header_start:header_start + 8 * 4]
                (_, total_packet_len,
                 _, frame_num,
                 _, num_detected_obj,
                 num_tlvs,
                 _) = struct.unpack("<8I", header)

                if len(buf) < total_packet_len:
                    break

                packet = buf[:total_packet_len]
                buf = buf[total_packet_len:]
                frame_num, points = parse_packet(packet)

                zone = "NO_TARGET"
                motion = "NO_TARGET"
                dist = float("nan")
                mean_v = float("nan")

                if points:
                    pts = np.array(points)  # shape (N, 4)
                    xs = pts[:, 0]
                    ys = pts[:, 1]
                    vs = pts[:, 3]

                    cx = float(np.mean(xs))
                    cy = float(np.mean(ys))
                    mean_v = float(np.mean(vs))  # *** true Doppler velocity ***

                    dist = cy

                    if cy < 2.0:
                        zone = "NEAR"
                    elif cy < 4.0:
                        zone = "MID"
                    elif cy < 6.0:
                        zone = "FAR"
                    else:
                        zone = "OUT_OF_RANGE"

                    # --------- MOTION FROM TRUE DOPPLER VELOCITY ----------
                    if mean_v < VEL_TOWARD_THRESH:
                        motion = "APPROACHING"
                    elif mean_v > VEL_AWAY_THRESH:
                        motion = "MOVING AWAY"
                    else:
                        motion = "STATIONARY"

                    print(
                        f"Frame {frame_num}: {zone} | {motion} | dist={dist:.2f} m | mean_v={mean_v:.3f}"
                    )

                    trail.append((cx, cy))
                    scatter.set_offsets(np.c_[xs, ys])
                    centroid_dot.set_data([cx], [cy])
                    trail_line.set_data([p[0] for p in trail],
                                        [p[1] for p in trail])

                ax.set_title(f"Frame {frame_num} | {zone} | {motion} | dist={dist:.2f} m")
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        data_ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    live_view()
