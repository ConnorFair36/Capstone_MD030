import json
import time

import roslibpy

import camera_module


def main() -> None:
    camera_module.initialize("yolo26n.pt")

    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()
    if not client.is_connected:
        raise RuntimeError("Could not connect to rosbridge at localhost:9090")

    detected_pub = roslibpy.Topic(client, "/camera_detected", "std_msgs/Bool")
    label_pub = roslibpy.Topic(client, "/camera_label", "std_msgs/String")
    dist_pub = roslibpy.Topic(client, "/camera_distance", "std_msgs/Float32")
    conf_pub = roslibpy.Topic(client, "/camera_confidence", "std_msgs/Float32")
    detections_pub = roslibpy.Topic(client, "/camera_detections_json", "std_msgs/String")
    frame_pub = roslibpy.Topic(client, "/camera_frame_b64", "std_msgs/String")

    pub_hz = 10.0
    pub_period = 1.0 / pub_hz
    last_pub = 0.0

    hold_secs = 0.4
    last_valid_t = 0.0
    last_valid = {
        "detected": False,
        "label": "",
        "distance": 999.0,
        "confidence": 0.0,
        "detections_json": "[]",
        "frame_b64": "",
    }

    try:
        while True:
            data = camera_module.get_camera_data()
            if data is None:
                time.sleep(0.001)
                continue

            now = time.time()
            if now - last_pub < pub_period:
                time.sleep(0.001)
                continue
            last_pub = now

            detected = bool(data.get("detected", False))
            label = data.get("label") or ""
            distance = data.get("distance")
            confidence = float(data.get("confidence", 0.0))
            detections = data.get("detections", [])
            frame = data.get("frame")

            detections_json = json.dumps(detections)
            frame_b64 = camera_module.encode_frame_to_b64(frame) if frame is not None else ""
            distance_val = 999.0 if distance is None else float(distance)

            if detected:
                last_valid_t = now
                last_valid = {
                    "detected": True,
                    "label": label,
                    "distance": distance_val,
                    "confidence": confidence,
                    "detections_json": detections_json,
                    "frame_b64": frame_b64,
                }

            if (not detected) and ((now - last_valid_t) < hold_secs):
                detected = last_valid["detected"]
                label = last_valid["label"]
                distance_val = last_valid["distance"]
                confidence = last_valid["confidence"]
                detections_json = last_valid["detections_json"]
                if frame_b64 == "":
                    frame_b64 = last_valid["frame_b64"]

            if (not detected) and ((now - last_valid_t) >= hold_secs):
                label = ""
                distance_val = 999.0
                confidence = 0.0
                detections_json = "[]"

            detected_pub.publish(roslibpy.Message({"data": bool(detected)}))
            label_pub.publish(roslibpy.Message({"data": label}))
            dist_pub.publish(roslibpy.Message({"data": float(distance_val)}))
            conf_pub.publish(roslibpy.Message({"data": float(confidence)}))
            detections_pub.publish(roslibpy.Message({"data": detections_json}))
            if frame_b64:
                frame_pub.publish(roslibpy.Message({"data": frame_b64}))

    finally:
        try:
            detected_pub.unadvertise()
            label_pub.unadvertise()
            dist_pub.unadvertise()
            conf_pub.unadvertise()
            detections_pub.unadvertise()
            frame_pub.unadvertise()
        except Exception:
            pass
        client.terminate()
        camera_module.shutdown()


if __name__ == "__main__":
    main()
