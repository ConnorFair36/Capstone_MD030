import time
import roslibpy
import numpy as np
import cv2

# Map state -> color (BGR)
STATE_COLOR = {
    "SAFE": (0, 255, 0),       # green
    "CAUTION": (0, 255, 255),  # yellow
    "STOP": (0, 0, 255),       # red
}

def main():
    # If localhost doesn't work, replace with your WSL IP
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    state = {"value": "SAFE", "last_update": time.time()}

    def on_cmd(msg):
        s = str(msg.get("data", "SAFE")).strip().upper()
        if s not in STATE_COLOR:
            s = "SAFE"
        state["value"] = s
        state["last_update"] = time.time()

    sub = roslibpy.Topic(client, '/vehicle_cmd', 'std_msgs/String')
    sub.subscribe(on_cmd)

    W, H = 1280, 700
    window_name = "STOPLIGHT (vehicle_cmd) - press q to quit"

    try:
        while True:
            img = np.zeros((H, W, 3), dtype=np.uint8)

            # If brain goes silent, fall back to SAFE after 2 seconds
            if time.time() - state["last_update"] > 2.0:
                state["value"] = "SAFE"

            color = STATE_COLOR[state["value"]]
            img[:] = color

            # Big text
            cv2.putText(
                img,
                state["value"],
                (40, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 0),
                8,
                cv2.LINE_AA
            )

            # Small label
            cv2.putText(
                img,
                "/vehicle_cmd",
                (40, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sub.unsubscribe()
        client.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()