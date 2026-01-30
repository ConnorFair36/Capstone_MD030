import struct
import serial
import time
import csv
from pathlib import Path

# ======= EDIT THESE =======
CFG_PORT = "COM3"          # CONFIG port
DATA_PORT = "COM4"         # DATA port
CFG_FILE = "profile.cfg"   # radar config file (same as TI visualizer)
BAUD_CFG = 115200
BAUD_DATA = 921600
LOG_SECONDS = 10           # capture duration (seconds); set None to run until Ctrl+C
# ==========================

OUT_BIN = "radar_raw.bin"
OUT_FRAMES_CSV = "radar_frames.csv"

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 8 * 4  # 8 uint32 after magic


def now_ms() -> int:
    return time.time_ns() // 1_000_000


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
    If a full packet exists at the start of buf, return (packet_bytes, frame_num, total_packet_len).
    Otherwise return (None, None, None).
    """
    if len(buf) < len(MAGIC_WORD) + HEADER_LEN:
        return None, None, None

    # Header starts right after magic
    header_start = len(MAGIC_WORD)
    header = buf[header_start:header_start + HEADER_LEN]

    # Unpack: version, totalPacketLen, platform, frameNum, timeCpuCycles, numDetectedObj, numTLVs, subFrameNum
    try:
        _, total_packet_len, _, frame_num, _, _, _, _ = struct.unpack("<8I", header)
    except struct.error:
        return None, None, None

    if total_packet_len <= 0:
        return None, None, None

    if len(buf) < total_packet_len:
        return None, None, None

    packet = bytes(buf[:total_packet_len])
    return packet, frame_num, total_packet_len


def main():
    base_dir = Path(__file__).resolve().parent
    cfg_path = (base_dir / CFG_FILE).resolve()
    out_bin_path = (base_dir / OUT_BIN).resolve()
    out_csv_path = (base_dir / OUT_FRAMES_CSV).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"CFG file not found: {cfg_path}")

    print(f"Opening CFG port {CFG_PORT} @ {BAUD_CFG}")
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)

    print(f"Opening DATA port {DATA_PORT} @ {BAUD_DATA}")
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.05)

    time.sleep(0.5)

    print(f"Sending config file: {cfg_path.name}")
    send_cfg(cfg_ser, cfg_path)
    print("Config sent. Radar should now be running.\n")

    # config port not needed after sending cfg
    cfg_ser.close()

    print(f"Logging RAW bytes to: {out_bin_path}")
    print(f"Logging per-frame timestamps to: {out_csv_path}")

    csv_exists = out_csv_path.exists()
    with open(out_csv_path, "a", newline="") as fcsv, open(out_bin_path, "wb") as fbin:
        writer = csv.writer(fcsv)
        if not csv_exists:
            writer.writerow(["host_timestamp_ms", "radar_frame_num", "packet_len_bytes", "bin_offset_end"])

        buf = bytearray()
        start_time = time.time()
        bin_offset = 0

        try:
            while True:
                if LOG_SECONDS is not None and (time.time() - start_time) >= LOG_SECONDS:
                    break

                chunk = data_ser.read(4096)
                if chunk:
                    fbin.write(chunk)
                    bin_offset += len(chunk)
                    buf.extend(chunk)

                # parse as many packets as available in buffer
                while True:
                    magic_idx = find_magic(buf)
                    if magic_idx < 0:
                        # keep last few bytes to avoid losing a partial magic word
                        if len(buf) > len(MAGIC_WORD):
                            buf = buf[-len(MAGIC_WORD):]
                        break

                    if magic_idx > 0:
                        buf = buf[magic_idx:]

                    packet, frame_num, pkt_len = try_extract_packet(buf)
                    if packet is None:
                        break

                    # At this moment, we've identified a full packet in the stream.
                    # Record a host timestamp for this frame number.
                    t_ms = now_ms()
                    writer.writerow([t_ms, frame_num, pkt_len, bin_offset])
                    fcsv.flush()

                    # remove packet from buffer
                    del buf[:pkt_len]

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C).")

    data_ser.close()
    print("Done logging. Ports closed.")


if __name__ == "__main__":
    main()
