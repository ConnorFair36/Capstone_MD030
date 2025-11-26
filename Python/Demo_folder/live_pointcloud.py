import struct
import serial
import time
import matplotlib.pyplot as plt

# ==== EDIT THESE IF NEEDED ====
CFG_PORT = "COM4"          # your config port (same as radar_logger.py)
DATA_PORT = "COM5"         # your data port
BAUD_CFG = 115200
BAUD_DATA = 921600
CFG_FILE = "profile.cfg"   # your working cfg with guiMonitor -1 1 1 0 0 0 0
# ===============================

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DETECTED_POINTS_TLV_TYPE = 1  # confirmed from your data


def send_cfg(cfg_ser, cfg_file):
    """Send cfg file to radar over the config UART."""
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("%")]

    for line in lines:
        cmd = (line + "\n").encode("utf-8")
        cfg_ser.write(cmd)
        time.sleep(0.05)   # small delay so radar can keep up


def find_magic(buf):
    """Find index of magic word in buffer."""
    return buf.find(MAGIC_WORD)


def parse_packet(packet_bytes):
    """
    Parse one full packet (starting at magic, correct length).
    Returns (frame_num, list_of_points), where points = [(x, y), ...]
    """
    if len(packet_bytes) < len(MAGIC_WORD) + 8 * 4:
        return None, []

    # Header starts right after magic
    header_start = len(MAGIC_WORD)
    header = packet_bytes[header_start:header_start + 8 * 4]
    (version,
     total_packet_len,
     platform,
     frame_num,
     time_cpu_cycles,
     num_detected_obj,
     num_tlvs,
     subframe_num) = struct.unpack("<8I", header)

    points_xy = []

    # TLVs come after header
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
            point_size = 16  # x, y, z, v as 4 floats
            expected_len = num_detected_obj * point_size

            if len(pts_bytes) >= expected_len:
                for obj_idx in range(num_detected_obj):
                    start = obj_idx * point_size
                    end = start + point_size
                    x, y, z, v = struct.unpack("<4f", pts_bytes[start:end])
                    points_xy.append((x, y))

        tlv_idx = tlv_value_end

    return frame_num, points_xy


def live_view():
    print(f"Opening CFG port {CFG_PORT} @ {BAUD_CFG}")
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)

    print(f"Opening DATA port {DATA_PORT} @ {BAUD_DATA}")
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)

    time.sleep(0.5)

    print(f"Sending config file: {CFG_FILE}")
    send_cfg(cfg_ser, CFG_FILE)
    print("Config sent. Radar should now be running.\n")

    # We don't need cfg port anymore
    cfg_ser.close()

    buf = bytearray()

    plt.ion()
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m - distance from radar)")
    ax.set_title("Live Radar Point Cloud (Top-Down View)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    try:
        while True:
            # Read some new bytes from the data port
            chunk = data_ser.read(4096)
            if chunk:
                buf.extend(chunk)

            # Try to extract as many full packets as we can
            while True:
                magic_idx = find_magic(buf)
                if magic_idx < 0:
                    # Keep last few bytes in case they are start of magic
                    if len(buf) > len(MAGIC_WORD):
                        buf = buf[-len(MAGIC_WORD):]
                    break

                # Discard any bytes before magic
                if magic_idx > 0:
                    buf = buf[magic_idx:]

                # Now buf starts with magic
                if len(buf) < len(MAGIC_WORD) + 8 * 4:
                    # Not enough for header yet
                    break

                # Peek header to know packet length
                header_start = len(MAGIC_WORD)
                header = buf[header_start:header_start + 8 * 4]
                (_, total_packet_len,
                 _, _, _, _,
                 _, _) = struct.unpack("<8I", header)

                if len(buf) < total_packet_len:
                    # Wait for more data
                    break

                # We have a full packet
                packet = buf[:total_packet_len]
                buf = buf[total_packet_len:]

                frame_num, points_xy = parse_packet(packet)

                # Update plot with this frame's points
                if points_xy:
                    xs, ys = zip(*points_xy)
                else:
                    xs, ys = [], []

                ax.cla()
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m - distance from radar)")
                ax.set_title(f"Live Radar Point Cloud (Frame {frame_num})")
                ax.grid(True)
                ax.set_aspect("equal", adjustable="box")
                #Lock the axis limits so the view doesn't jump around
                ax.set_xlim(-4, 4)    # left/right span in meters
                ax.set_ylim(0, 10)    # distance from radar in meters
                ax.scatter(xs, ys, s=10)
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        data_ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    live_view()
