import time
import os
import json
import importlib.util
import roslibpy

# Force-load camera_module.py from the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_MODULE_PATH = os.path.join(SCRIPT_DIR, "camera_module.py")

spec = importlib.util.spec_from_file_location("camera_module", CAMERA_MODULE_PATH)
camera_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_module)

print("Loaded camera_module from:", CAMERA_MODULE_PATH)
print("Has initialize:", hasattr(camera_module, "initialize"))
print("Has get_camera_data:", hasattr(camera_module, "get_camera_data"))
print("Has shutdown:", hasattr(camera_module, "shutdown"))


def main():
    # Load the camera module using best.pt
    camera_module.initialize("best.pt")

    # Connect to rosbridge websocket
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    # ROS publishers
    detected_pub = roslibpy.Topic(client, '/camera_detected', 'std_msgs/Bool')
    label_pub = roslibpy.Topic(client, '/camera_label', 'std_msgs/String')
    dist_pub = roslibpy.Topic(client, '/camera_distance', 'std_msgs/Float32')
    conf_pub = roslibpy.Topic(client, '/camera_confidence', 'std_msgs/Float32')
    low_light_pub = roslibpy.Topic(client, '/camera_low_light', 'std_msgs/Bool')
    bad_weather_pub = roslibpy.Topic(client, '/camera_bad_weather', 'std_msgs/Bool')
    detections_pub = roslibpy.Topic(client, '/camera_detections_json', 'std_msgs/String')

    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ
    last_pub = 0.0

    # Hold last valid detection briefly to reduce flicker
    HOLD_SECS = 0.4
    last_valid_t = 0.0
    last_valid_detected = False
    last_valid_label = ""
    last_valid_distance = 999.0
    last_valid_conf = 0.0
    last_valid_low_light = False
    last_valid_bad_weather = False
    last_valid_detections_json = "[]"

    try:
        while True:
            data = camera_module.get_camera_data()

            if data is None:
                time.sleep(0.001)
                continue

            if data == "QUIT":
                break

            now = time.time()
            if now - last_pub < pub_period:
                time.sleep(0.001)
                continue
            last_pub = now

            detected = bool(data.get("detected", False))
            label = data.get("label", None)
            distance = data.get("distance", None)
            confidence = float(data.get("confidence", 0.0))
            detections = data.get("detections", [])
            low_light = bool(data.get("low_light", False))
            bad_weather = bool(data.get("bad_weather", False))

            label_val = "" if label is None else str(label)
            distance_val = 999.0 if distance is None else float(distance)
            detections_json = json.dumps(detections)

            # Save last valid detection
            if detected:
                last_valid_t = now
                last_valid_detected = True
                last_valid_label = label_val
                last_valid_distance = distance_val
                last_valid_conf = confidence
                last_valid_low_light = low_light
                last_valid_bad_weather = bad_weather
                last_valid_detections_json = detections_json

            # Hold the last valid detection briefly
            if (not detected) and ((now - last_valid_t) < HOLD_SECS):
                detected = last_valid_detected
                label_val = last_valid_label
                distance_val = last_valid_distance
                confidence = last_valid_conf
                low_light = last_valid_low_light
                bad_weather = last_valid_bad_weather
                detections_json = last_valid_detections_json

            # Reset after timeout
            if (not detected) and ((now - last_valid_t) >= HOLD_SECS):
                label_val = ""
                distance_val = 999.0
                confidence = 0.0
                detections_json = "[]"

            # Publish to ROS
            detected_pub.publish(roslibpy.Message({'data': bool(detected)}))
            label_pub.publish(roslibpy.Message({'data': label_val}))
            dist_pub.publish(roslibpy.Message({'data': float(distance_val)}))
            conf_pub.publish(roslibpy.Message({'data': float(confidence)}))
            low_light_pub.publish(roslibpy.Message({'data': bool(low_light)}))
            bad_weather_pub.publish(roslibpy.Message({'data': bool(bad_weather)}))
            detections_pub.publish(roslibpy.Message({'data': detections_json}))

    finally:
        try:
            detected_pub.unadvertise()
            label_pub.unadvertise()
            dist_pub.unadvertise()
            conf_pub.unadvertise()
            low_light_pub.unadvertise()
            bad_weather_pub.unadvertise()
            detections_pub.unadvertise()
        except Exception:
            pass

        client.terminate()
        camera_module.shutdown()


if __name__ == "__main__":
    main()
