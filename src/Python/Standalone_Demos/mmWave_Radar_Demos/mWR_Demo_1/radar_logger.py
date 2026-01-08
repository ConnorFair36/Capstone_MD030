import serial
import time
from pathlib import Path

# ======= EDIT THESE THREE THINGS =======
CFG_PORT = "COM4"      # <- replace with your CONFIG port
DATA_PORT = "COM5"     # <- replace with your DATA port
CFG_FILE = "profile.cfg"  # <- the same .cfg you use in the TI visualizer
# =======================================

BAUD_CFG = 115200
BAUD_DATA = 921600
LOG_FILE = "radar_raw.bin"   # where we save raw bytes
LOG_SECONDS = 10             # how long to capture for


def send_cfg(cfg_port, cfg_file):
    """Send .cfg lines to the radar, like the TI visualizer does."""
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("%")]

    for line in lines:
        cmd = (line + "\n").encode("utf-8")
        cfg_port.write(cmd)
        time.sleep(0.05)   # small delay so radar can keep up


def main():
    # 1) Open config port
    print(f"Opening CFG port {CFG_PORT} @ {BAUD_CFG}")
    cfg_ser = serial.Serial(CFG_PORT, BAUD_CFG, timeout=1)

    # 2) Open data port
    print(f"Opening DATA port {DATA_PORT} @ {BAUD_DATA}")
    data_ser = serial.Serial(DATA_PORT, BAUD_DATA, timeout=1)

    time.sleep(0.5)

    # 3) Send the cfg to start the radar
    print(f"Sending config file: {CFG_FILE}")
    send_cfg(cfg_ser, CFG_FILE)
    print("Config sent. Radar should now be running.\n")

    # 4) Start logging bytes from data port
    log_path = Path(LOG_FILE)
    print(f"Logging data for {LOG_SECONDS} seconds to {log_path.resolve()}")

    start = time.time()
    with open(log_path, "wb") as f:
        while time.time() - start < LOG_SECONDS:
            chunk = data_ser.read(4096)  # read whatever is available
            if chunk:
                f.write(chunk)

    print("Done logging.")

    cfg_ser.close()
    data_ser.close()
    print("Ports closed.")


if __name__ == "__main__":
    main()
