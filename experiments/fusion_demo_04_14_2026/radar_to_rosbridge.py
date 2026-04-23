import json
import time

import roslibpy

import radar_module


def main() -> None:
    radar_module.initialize()

    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    radar_detected_pub = roslibpy.Topic(client, "/radar_detected", "std_msgs/Bool")
    radar_dist_pub = roslibpy.Topic(client, "/radar_distance", "std_msgs/Float32")
    radar_motion_pub = roslibpy.Topic(client, "/radar_motion", "std_msgs/String")
    radar_conf_pub = roslibpy.Topic(client, "/radar_confidence", "std_msgs/Float32")
    radar_points_pub = roslibpy.Topic(client, "/radar_points_json", "std_msgs/String")

    pub_hz = 10.0
    pub_period = 1.0 / pub_hz
    last_pub = 0.0

    hold_secs = 0.6
    last_valid_t = 0.0
    last_valid = {
        "detected": False,
        "distance": 999.0,
        "motion": "NONE",
        "confidence": 0.0,
        "points_json": "[]",
    }

    try:
        while True:
            data = radar_module.get_radar_data()
            now = time.time()

            if now - last_pub < pub_period:
                time.sleep(0.001)
                continue
            last_pub = now

            detected = False
            dist_val = 999.0
            motion = "NONE"
            conf = 0.0
            points_json = "[]"

            if data is not None:
                detected = bool(data.get("detected", False))
                dist = data.get("distance")
                if dist is not None:
                    dist_val = float(dist)
                motion = str(data.get("motion", "NONE"))
                conf = float(data.get("confidence", 0.0))
                points_json = json.dumps(data.get("points", []))

                if detected:
                    last_valid_t = now
                    last_valid = {
                        "detected": detected,
                        "distance": dist_val,
                        "motion": motion,
                        "confidence": conf,
                        "points_json": points_json,
                    }

            if (not detected) and ((now - last_valid_t) < hold_secs):
                detected = last_valid["detected"]
                dist_val = last_valid["distance"]
                motion = last_valid["motion"]
                conf = last_valid["confidence"]
                points_json = last_valid["points_json"]

            if (not detected) and ((now - last_valid_t) >= hold_secs):
                dist_val = 999.0
                motion = "NONE"
                conf = 0.0
                points_json = "[]"

            radar_detected_pub.publish(roslibpy.Message({"data": bool(detected)}))
            radar_dist_pub.publish(roslibpy.Message({"data": float(dist_val)}))
            radar_motion_pub.publish(roslibpy.Message({"data": motion}))
            radar_conf_pub.publish(roslibpy.Message({"data": float(conf)}))
            radar_points_pub.publish(roslibpy.Message({"data": points_json}))

    finally:
        try:
            radar_detected_pub.unadvertise()
            radar_dist_pub.unadvertise()
            radar_motion_pub.unadvertise()
            radar_conf_pub.unadvertise()
            radar_points_pub.unadvertise()
        except Exception:
            pass
        client.terminate()
        radar_module.close()


if __name__ == "__main__":
    main()
