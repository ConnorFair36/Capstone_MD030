# radar_module.py (debug version)
import serial
import struct
import time
import numpy as np

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

# Clustering parameter
CLUSTER_DIST_THRESH = 0.5  # meters

data_ser = None
buf = bytearray()


# ---------------- CONFIG ----------------
def send_cfg():
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)
    with open(CFG_FILE, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("%")]
    for line in lines:
        cfg_ser.write((line + "\n").encode("utf-8"))
        time.sleep(0.05)
    cfg_ser.close()


def initialize():
    global data_ser
    print("Initializing radar...")
    send_cfg()
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)
    print("Radar ready.")


# ---------------- PARSING ----------------
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
        tlv_type, tlv_len = struct.unpack("<2I", packet[tlv_idx:tlv_idx + 8])
        tlv_val_start = tlv_idx + 8
        tlv_val_end = tlv_val_start + tlv_len

        if tlv_type == DETECTED_POINTS_TLV_TYPE and num_obj > 0:
            blob = packet[tlv_val_start:tlv_val_end]
            for i in range(num_obj):
                start = i * 16
                x, y, z, v = struct.unpack("<4f", blob[start:start + 16])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                vs.append(v)

        tlv_idx = tlv_val_end

    return xs, ys, zs, vs


# ---------------- CLUSTERING ----------------
def cluster_points(xs, ys, vs, dist_thresh=CLUSTER_DIST_THRESH):
    clusters = []
    for x, y, v in zip(xs, ys, vs):
        placed = False
        for cluster in clusters:
            cx = np.mean([p[0] for p in cluster])
            cy = np.mean([p[1] for p in cluster])
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < dist_thresh:
                cluster.append((x, y, v))
                placed = True
                break
        if not placed:
            clusters.append([(x, y, v)])
    return clusters


# ---------------- MAIN FUNCTION ----------------
def get_radar_data():
    """
    Returns:
    {
        "distance": float | None,
        "motion": "APPROACHING" / "MOVING_AWAY" / "STATIONARY" / "NONE"
    }
    """
    global buf

    chunk = data_ser.read(4096)
    print(f"[DEBUG] Read chunk length: {len(chunk)}")  # debug
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

        # ---- DEBUG RAW POINTS ----
        print(f"[DEBUG] Raw points detected: {len(xs)}")
        if xs:
            print(f"[DEBUG] x: {xs[:5]}, y: {ys[:5]}, v: {vs[:5]}")

        #  No detections
        if len(ys) == 0:
            print("[DEBUG] No points detected in this packet")
            return {"distance": None, "motion": "NONE"}

        # ---------------- FILTER ----------------
        filtered_xs = []
        filtered_ys = []
        filtered_vs = []

        for x, y, v in zip(xs, ys, vs):
            if 0.05 < y < 15 and abs(v) < 5.0:
                filtered_xs.append(x)
                filtered_ys.append(y)
                filtered_vs.append(v)

        print(f"[DEBUG] Filtered points: {len(filtered_ys)}")
        if len(filtered_ys) == 0:
            print("[DEBUG] No points survived filtering")
            return {"distance": None, "motion": "NONE"}

        # ---------------- CLUSTER ----------------
        clusters = cluster_points(filtered_xs, filtered_ys, filtered_vs)
        print(f"[DEBUG] Number of clusters: {len(clusters)}")

        # pick largest cluster
        best_cluster = max(clusters, key=len)
        cluster_ys = [p[1] for p in best_cluster]
        cluster_vs = [p[2] for p in best_cluster]
        mean_y = float(np.median(cluster_ys))
        mean_v = float(np.median(cluster_vs))

        print(f"[DEBUG] Best cluster center: y={mean_y:.2f}, v={mean_v:.2f}")

        # ---------------- MOTION ----------------
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