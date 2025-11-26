# radar_module.py
#
# Provides a simple function `get_radar_data()` that returns:
# {
#   "distance": float in meters or None,
#   "motion":   "APPROACHING" / "MOVING_AWAY" / "STATIONARY" / "NONE"
# }
#
# Usage: call initialize() once, then call get_radar_data() repeatedly.

import serial
import struct
import time
import numpy as np

CFG_PORT = "COM4"
DATA_PORT = "COM5"
BAUD_CFG = 115200
BAUD_DATA = 921600
CFG_FILE = "profile.cfg"

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DETECTED_POINTS_TLV_TYPE = 1

# Motion thresholds (tuned later)
APPROACH_THRESH = -0.2   # mean_v < -0.2 m/s
AWAY_THRESH      = +0.2  # mean_v > +0.2 m/s

data_ser = None
buf = bytearray()

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
    global data_ser
    print("Initializing radar...")
    send_cfg()
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)
    print("Radar ready.")

def _find_magic(buf):
    return buf.find(MAGIC_WORD)

def _parse_packet(packet):
    header_start = len(MAGIC_WORD)
    header = packet[header_start:header_start + 32]
    (_, total_len, _, frame_num, _, num_obj, num_tlvs, _) = struct.unpack("<8I", header)

    tlv_idx = header_start + 32
    packet_end = total_len

    xs, ys, zs, vs = [], [], [], []

    for _ in range(num_tlvs):
        if tlv_idx + 8 > packet_end:
            break
        tlv_type, tlv_len = struct.unpack("<2I", packet[tlv_idx:tlv_idx+8])
        tlv_val_start = tlv_idx + 8
        tlv_val_end = tlv_val_start + tlv_len

        if tlv_type == DETECTED_POINTS_TLV_TYPE and num_obj > 0:
            blob = packet[tlv_val_start:tlv_val_end]
            for i in range(num_obj):
                start = i * 16
                x, y, z, v = struct.unpack("<4f", blob[start:start+16])
                xs.append(x); ys.append(y); zs.append(z); vs.append(v)

        tlv_idx = tlv_val_end

    return xs, ys, zs, vs

def get_radar_data():
    """
    Reads incoming radar bytes, parses a frame if available, and returns:
      { "distance": float | None, "motion": str }
    """
    global buf

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

        if len(buf) < total_len:
            break

        packet = buf[:total_len]
        buf = buf[total_len:]

        xs, ys, zs, vs = _parse_packet(packet)

        if len(ys) == 0:
            return {"distance": None, "motion": "NONE"}

        mean_y = float(np.mean(ys))  # forward distance
        mean_v = float(np.mean(vs))  # Doppler velocity

        # classify motion
        if mean_v < APPROACH_THRESH:
            motion = "APPROACHING"
        elif mean_v > AWAY_THRESH:
            motion = "MOVING_AWAY"
        else:
            motion = "STATIONARY"

        return {
            "distance": mean_y,
            "motion": motion
        }

    return None
