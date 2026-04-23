import json
import time
import cv2
import numpy as np
import roslibpy

# ----------------------------
# Shared state updated by ROS callbacks
# ----------------------------
state = {
    "detected": False,
    "label": "",
    "distance": 999.0,
    "confidence": 0.0,
    "low_light": False,
    "brightness": 0.0,
    "bad_weather": False,
    "detections": [],
}


# ----------------------------
# ROS callback helpers
# ----------------------------
def on_detected(msg):
    state["detected"] = bool(msg.get("data", False))


def on_label(msg):
    state["label"] = str(msg.get("data", ""))


def on_distance(msg):
    try:
        state["distance"] = float(msg.get("data", 999.0))
    except Exception:
        state["distance"] = 999.0


def on_confidence(msg):
    try:
        state["confidence"] = float(msg.get("data", 0.0))
    except Exception:
        state["confidence"] = 0.0


def on_low_light(msg):
    state["low_light"] = bool(msg.get("data", False))


def on_brightness(msg):
    try:
        state["brightness"] = float(msg.get("data", 0.0))
    except Exception:
        state["brightness"] = 0.0


def on_bad_weather(msg):
    state["bad_weather"] = bool(msg.get("data", False))


def on_detections(msg):
    raw = msg.get("data", "[]")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            state["detections"] = parsed
        else:
            state["detections"] = []
    except Exception:
        state["detections"] = []


# ----------------------------
# Drawing
# ----------------------------
def draw_ui(canvas):
    canvas[:] = 20

    h, w = canvas.shape[:2]

    title = "ROS Brain Node - Camera Output"
    cv2.putText(
        canvas, title, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )

    detected_text = f"Detected: {state['detected']}"
    label_text = f"Primary Label: {state['label'] if state['label'] else 'NONE'}"

    dist_val = state["distance"]
    dist_text = "N/A" if dist_val >= 999.0 else f"{dist_val:.2f} m"

    conf_text = f"Confidence: {state['confidence']:.2f}"
    distance_text = f"Distance: {dist_text}"
    low_light_text = f"Low Light: {state['low_light']}"
    brightness_text = f"Brightness: {state['brightness']:.1f}"
    bad_weather_text = f"Bad Weather: {state['bad_weather']}"

    y = 90
    line_gap = 35

    summary_lines = [
        detected_text,
        label_text,
        conf_text,
        distance_text,
        low_light_text,
        brightness_text,
        bad_weather_text,
        f"Detections Count: {len(state['detections'])}",
    ]

    for line in summary_lines:
        cv2.putText(
            canvas, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2
        )
        y += line_gap

    # Divider
    cv2.line(canvas, (20, 390), (w - 20, 390), (100, 100, 100), 2)

    cv2.putText(
        canvas, "Detections:", (20, 430),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
    )

    # List each detection
    y = 470
    max_rows = 8

    for i, det in enumerate(state["detections"][:max_rows]):
        label = str(det.get("label", "unknown"))
        conf = float(det.get("confidence", 0.0))

        d = det.get("distance", None)
        d_text = "N/A" if d is None else f"{float(d):.2f} m"

        bbox = det.get("bbox", [])
        center = det.get("center", [])

        line1 = f"{i+1}. {label} | conf={conf:.2f} | dist={d_text}"
        line2 = f"    bbox={bbox} | center={center}"

        cv2.putText(
            canvas, line1, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2
        )
        y += 28

        cv2.putText(
            canvas, line2, (40, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1
        )
        y += 38

    if len(state["detections"]) == 0:
        cv2.putText(
            canvas, "No detections received.",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
        )


def main():
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    if not client.is_connected:
        raise RuntimeError("Could not connect to rosbridge at localhost:9090")

    # Subscribers
    detected_sub = roslibpy.Topic(client, "/camera_detected", "std_msgs/Bool")
    label_sub = roslibpy.Topic(client, "/camera_label", "std_msgs/String")
    distance_sub = roslibpy.Topic(client, "/camera_distance", "std_msgs/Float32")
    confidence_sub = roslibpy.Topic(client, "/camera_confidence", "std_msgs/Float32")
    low_light_sub = roslibpy.Topic(client, "/camera_low_light", "std_msgs/Bool")
    brightness_sub = roslibpy.Topic(client, "/camera_brightness", "std_msgs/Float32")
    bad_weather_sub = roslibpy.Topic(client, "/camera_bad_weather", "std_msgs/Bool")
    detections_sub = roslibpy.Topic(client, "/camera_detections_json", "std_msgs/String")

    detected_sub.subscribe(on_detected)
    label_sub.subscribe(on_label)
    distance_sub.subscribe(on_distance)
    confidence_sub.subscribe(on_confidence)
    low_light_sub.subscribe(on_low_light)
    brightness_sub.subscribe(on_brightness)
    bad_weather_sub.subscribe(on_bad_weather)
    detections_sub.subscribe(on_detections)

    try:
        while True:
            canvas = np.zeros((720, 1000, 3), dtype=np.uint8)
            draw_ui(canvas)

            cv2.imshow("ROS Brain Node Output", canvas)

            # Press q to quit the brain display
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Stopping brain node...")

    finally:
        try:
            detected_sub.unsubscribe()
            label_sub.unsubscribe()
            distance_sub.unsubscribe()
            confidence_sub.unsubscribe()
            low_light_sub.unsubscribe()
            brightness_sub.unsubscribe()
            bad_weather_sub.unsubscribe()
            detections_sub.unsubscribe()
        except Exception:
            pass

        client.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
