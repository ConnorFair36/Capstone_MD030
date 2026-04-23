# radar_module.py
#
# Updated radar module for ROS fusion with brain_node.
#
# Returns:
# {
#   "detected": bool,
#   "distance": float | None,
#   "motion":   "APPROACHING" / "MOVING_AWAY" / "STATIONARY" / "NONE",
#   "confidence": float   # 0.0 to 1.0
# }
#
# Usage:
#   initialize()
#   data = get_radar_data()
#   close()

import serial
import struct
import time
import math
import numpy as np
from collections import deque

CFG_PORT = "COM3"
DATA_PORT = "COM4"
BAUD_CFG = 115200
BAUD_DATA = 921600
CFG_FILE = "profile.cfg"

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DETECTED_POINTS_TLV_TYPE = 1

# Motion thresholds
APPROACH_THRESH = -0.2
AWAY_THRESH = 0.2

# ROI filtering
ROI_X_ABS = 2.5
ROI_Y_MIN = 0.1
ROI_Y_MAX = 10.0
ROI_Z_ABS = 2.0
RANGE_MIN = 0.1
RANGE_MAX = 10.0

# Velocity smoothing
VEL_HISTORY_LEN = 5

data_ser = None
buf = bytearray()
vel_hist = deque(maxlen=VEL_HISTORY_LEN)


def send_cfg():
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)
    with open(CFG_FILE, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("%")]
    for line in lines:
        cfg_ser.write((line + "\n").encode("utf-8"))
        time.sleep(0.05)
    cfg_ser.close()


def initialize():
    """Open radar ports and start streaming."""
    global data_ser, buf
    print("Initializing radar...")
    send_cfg()
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)
    buf = bytearray()
    print("Radar ready.")


def close():
    global data_ser
    if data_ser is not None:
        try:
            data_ser.close()
        except Exception:
            pass
        data_ser = None


def _find_magic(buffer):
    return buffer.find(MAGIC_WORD)


def _point_in_roi(x, y, z, r):
    if abs(x) > ROI_X_ABS:
        return False
    if y < ROI_Y_MIN or y > ROI_Y_MAX:
        return False
    if abs(z) > ROI_Z_ABS:
        return False
    if r < RANGE_MIN or r > RANGE_MAX:
        return False
    return True


def _classify_motion(v):
    if v < APPROACH_THRESH:
        return "APPROACHING"
    elif v > AWAY_THRESH:
        return "MOVING_AWAY"
    else:
        return "STATIONARY"


def _compute_confidence(num_valid_points, motion, raw_velocity):
    """
    Simple demo-friendly confidence model.
    You can tune this later.
    """
    conf = 0.0

    # More valid points = more confidence
    conf += min(num_valid_points / 8.0, 0.5)

    # Stronger nonzero motion = more confidence
    speed_mag = abs(raw_velocity)
    conf += min(speed_mag / 1.5, 0.3)

    # Motion label bonus
    if motion == "APPROACHING":
        conf += 0.2
    elif motion == "MOVING_AWAY":
        conf += 0.1
    elif motion == "STATIONARY":
        conf += 0.05

    return max(0.0, min(conf, 1.0))


def _smooth_velocity(v):
    vel_hist.append(v)
    return float(np.mean(vel_hist))


def _parse_packet(packet):
    """
    Parse one packet and return all x, y, z, v points.
    Assumes detected points TLV stores repeated float32 x,y,z,v tuples.
    """
    header_start = len(MAGIC_WORD)
    header = packet[header_start:header_start + 32]
    (_, total_len, _, frame_num, _, num_obj, num_tlvs, _) = struct.unpack("<8I", header)

    tlv_idx = header_start + 32
    packet_end = total_len

    xs, ys, zs, vs = [], [], [], []

    for _ in range(num_tlvs):
        if tlv_idx + 8 > packet_end:
            break

        tlv_type, tlv_len = struct.unpack("<2I", packet[tlv_idx:tlv_idx + 8])
        tlv_val_start = tlv_idx + 8
        tlv_val_end = tlv_idx + tlv_len

        if tlv_val_end > len(packet):
            break

        if tlv_type == DETECTED_POINTS_TLV_TYPE and num_obj > 0:
            blob = packet[tlv_val_start:tlv_val_end]

            for i in range(num_obj):
                start = i * 16
                end = start + 16
                if end > len(blob):
                    break

                x, y, z, v = struct.unpack("<4f", blob[start:end])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                vs.append(v)

        tlv_idx = tlv_val_end

    return xs, ys, zs, vs


def get_radar_data():
    """
    Reads incoming radar bytes, parses a frame if available, and returns:
    {
      "detected": bool,
      "distance": float | None,
      "motion": str,
      "confidence": float
    }

    Returns None if no complete frame is available yet.
    """
    global buf

    if data_ser is None:
        raise RuntimeError("Radar not initialized. Call initialize() first.")

    chunk = data_ser.read(4096)
    if chunk:
        buf.extend(chunk)

    while True:
        magic_idx = _find_magic(buf)

        if magic_idx < 0:
            if len(buf) > len(MAGIC_WORD):
                buf = buf[-len(MAGIC_WORD):]
            break

        if magic_idx > 0:
            buf = buf[magic_idx:]

        if len(buf) < len(MAGIC_WORD) + 32:
            break

        header = buf[len(MAGIC_WORD):len(MAGIC_WORD) + 32]
        (_, total_len, *_rest) = struct.unpack("<8I", header)

        if total_len <= 0:
            buf = buf[1:]
            continue

        if len(buf) < total_len:
            break

        packet = buf[:total_len]
        buf = buf[total_len:]

        xs, ys, zs, vs = _parse_packet(packet)

        if len(xs) == 0:
            return {
                "detected": False,
                "distance": None,
                "motion": "NONE",
                "confidence": 0.0
            }

        # Build valid points in ROI
        valid_points = []
        for x, y, z, v in zip(xs, ys, zs, vs):
            r = math.sqrt(x * x + y * y + z * z)
            if _point_in_roi(x, y, z, r):
                valid_points.append((r, x, y, z, v))

        if len(valid_points) == 0:
            return {
                "detected": False,
                "distance": None,
                "motion": "NONE",
                "confidence": 0.0
            }

        # Sort by nearest target
        valid_points.sort(key=lambda p: p[0])

        # Use nearest valid target as primary
        primary_r, primary_x, primary_y, primary_z, primary_v = valid_points[0]

        # Smooth primary velocity
        v_smooth = _smooth_velocity(primary_v)
        motion = _classify_motion(v_smooth)

        confidence = _compute_confidence(
            num_valid_points=len(valid_points),
            motion=motion,
            raw_velocity=v_smooth
        )

        return {
            "detected": True,
            "distance": float(primary_r),
            "motion": motion,
            "confidence": float(confidence)
        }

    return None


if __name__ == "__main__":
    try:
        initialize()
        while True:
            data = get_radar_data()
            if data is not None:
                print(data)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping radar...")
    finally:
        close()
