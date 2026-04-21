import math
import struct
import time
from collections import deque

import numpy as np
import serial

CFG_PORT = "COM5"
DATA_PORT = "COM6"
BAUD_CFG = 115200
BAUD_DATA = 921600
CFG_FILE = "profile.cfg"

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DETECTED_POINTS_TLV_TYPE = 1

APPROACH_THRESH = -0.2
AWAY_THRESH = 0.2

ROI_X_ABS = 2.5
ROI_Y_MIN = 0.1
ROI_Y_MAX = 10.0
ROI_Z_ABS = 2.0
RANGE_MIN = 0.1
RANGE_MAX = 10.0

VEL_HISTORY_LEN = 5

data_ser = None
buf = bytearray()
vel_hist = deque(maxlen=VEL_HISTORY_LEN)


def send_cfg() -> None:
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("%")]
    for line in lines:
        cfg_ser.write((line + "\n").encode("utf-8"))
        time.sleep(0.05)
    cfg_ser.close()


def initialize() -> None:
    global data_ser, buf
    print("Initializing radar...")
    send_cfg()
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)
    buf = bytearray()
    print("Radar ready.")


def close() -> None:
    global data_ser
    if data_ser is not None:
        try:
            data_ser.close()
        except Exception:
            pass
        data_ser = None


def _find_magic(buffer: bytearray) -> int:
    return buffer.find(MAGIC_WORD)


def _point_in_roi(x: float, y: float, z: float, r: float) -> bool:
    if abs(x) > ROI_X_ABS:
        return False
    if y < ROI_Y_MIN or y > ROI_Y_MAX:
        return False
    if abs(z) > ROI_Z_ABS:
        return False
    if r < RANGE_MIN or r > RANGE_MAX:
        return False
    return True


def _classify_motion(v: float) -> str:
    if v < APPROACH_THRESH:
        return "APPROACHING"
    if v > AWAY_THRESH:
        return "MOVING_AWAY"
    return "STATIONARY"


def _compute_confidence(num_valid_points: int, motion: str, raw_velocity: float) -> float:
    conf = 0.0
    conf += min(num_valid_points / 8.0, 0.5)
    conf += min(abs(raw_velocity) / 1.5, 0.3)
    if motion == "APPROACHING":
        conf += 0.2
    elif motion == "MOVING_AWAY":
        conf += 0.1
    elif motion == "STATIONARY":
        conf += 0.05
    return max(0.0, min(conf, 1.0))


def _smooth_velocity(v: float) -> float:
    vel_hist.append(v)
    return float(np.mean(vel_hist))


def _parse_packet(packet: bytes):
    header_start = len(MAGIC_WORD)
    header = packet[header_start:header_start + 32]
    (_, total_len, _, _frame_num, _, num_obj, num_tlvs, _) = struct.unpack("<8I", header)

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

        packet = bytes(buf[:total_len])
        buf = buf[total_len:]

        xs, ys, zs, vs = _parse_packet(packet)
        if len(xs) == 0:
            return {
                "detected": False,
                "distance": None,
                "motion": "NONE",
                "confidence": 0.0,
                "points": [],
            }

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
                "confidence": 0.0,
                "points": [],
            }

        valid_points.sort(key=lambda p: p[0])
        point_dicts = [
            {"x": float(x), "y": float(y), "z": float(z), "v": float(v), "range": float(r)}
            for r, x, y, z, v in valid_points
        ]

        primary_r, _primary_x, _primary_y, _primary_z, primary_v = valid_points[0]
        v_smooth = _smooth_velocity(primary_v)
        motion = _classify_motion(v_smooth)
        confidence = _compute_confidence(len(valid_points), motion, v_smooth)

        return {
            "detected": True,
            "distance": float(primary_r),
            "motion": motion,
            "confidence": float(confidence),
            "points": point_dicts,
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
        pass
    finally:
        close()
