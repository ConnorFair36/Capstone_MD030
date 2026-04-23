import struct
import csv
from pathlib import Path

# Input raw binary file captured from the radar
RAW_FILE = "radar_raw.bin"

# Output CSV file with parsed point cloud
OUT_CSV = "pointcloud.csv"

# Magic word used by TI mmWave demo (8 bytes)
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# TLV type used for detected points in your SDK/config
DETECTED_POINTS_TLV_TYPE = 1  # confirmed from your data


def find_magic(data: bytearray, start_idx: int = 0) -> int:
    """Find the next magic word starting at or after start_idx."""
    return data.find(MAGIC_WORD, start_idx)


def parse_file():
    raw_path = Path(RAW_FILE)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print(f"[ERROR] {RAW_FILE} not found or empty.")
        return

    data = bytearray(raw_path.read_bytes())
    print(f"[INFO] Loaded {len(data)} bytes from {RAW_FILE}")

    out_path = Path(OUT_CSV)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "obj_index", "x_m", "y_m", "z_m", "vel_mps"])

        idx = 0
        frame_count = 0
        parsed_points = 0

        while True:
            magic_idx = find_magic(data, idx)
            if magic_idx < 0:
                break  # no more packets

            # Magic (8 bytes) + header (8 * 4 bytes)
            header_start = magic_idx + len(MAGIC_WORD)
            if header_start + 8 * 4 > len(data):
                break  # not enough bytes for a full header

            # Header: 8 uint32, little-endian
            header = data[header_start:header_start + 8 * 4]
            (version,
             total_packet_len,
             platform,
             frame_num,
             time_cpu_cycles,
             num_detected_obj,
             num_tlvs,
             subframe_num) = struct.unpack("<8I", header)

            packet_start = magic_idx
            packet_end = packet_start + total_packet_len
            if packet_end > len(data):
                # Incomplete packet at end of file
                break

            # TLVs come immediately after the 8-int header
            tlv_idx = header_start + 8 * 4

            # Loop over TLVs
            for _ in range(num_tlvs):
                # Need at least 8 bytes for TLV header
                if tlv_idx + 8 > packet_end:
                    break

                tlv_type, tlv_len = struct.unpack("<2I", data[tlv_idx:tlv_idx + 8])
                tlv_value_start = tlv_idx + 8
                tlv_value_end = tlv_value_start + tlv_len  # tlv_len = payload length only

                # Sanity check
                if tlv_value_end > packet_end or tlv_len < 0:
                    break  # malformed TLV, bail out of this packet

                # Detected Points TLV
                if tlv_type == DETECTED_POINTS_TLV_TYPE and num_detected_obj > 0:
                    pts_bytes = data[tlv_value_start:tlv_value_end]
                    point_size = 16  # 4 floats: x, y, z, velocity

                    expected_len = num_detected_obj * point_size
                    if len(pts_bytes) >= expected_len:
                        for obj_idx in range(num_detected_obj):
                            start = obj_idx * point_size
                            end = start + point_size
                            x, y, z, v = struct.unpack("<4f", pts_bytes[start:end])
                            writer.writerow([frame_num, obj_idx, x, y, z, v])
                            parsed_points += 1

                # Move to next TLV (header + payload)
                tlv_idx = tlv_value_end

            frame_count += 1
            idx = packet_end  # move to next packet

    print(f"[INFO] Parsed {frame_count} frames, {parsed_points} points total.")
    print(f"[INFO] Output saved to {out_path.resolve()}")


if __name__ == "__main__":
    parse_file()
