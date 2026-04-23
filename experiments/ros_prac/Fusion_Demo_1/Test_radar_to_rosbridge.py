import time
import roslibpy
import radar_module  # uses COM ports + profile.cfg


def main():
    radar_module.initialize()

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    radar_detected_pub = roslibpy.Topic(client, '/radar_detected', 'std_msgs/Bool')
    radar_dist_pub     = roslibpy.Topic(client, '/radar_distance', 'std_msgs/Float32')
    radar_motion_pub   = roslibpy.Topic(client, '/radar_motion', 'std_msgs/String')
    radar_conf_pub     = roslibpy.Topic(client, '/radar_confidence', 'std_msgs/Float32')

    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ
    last_pub = 0.0

    # persistence to reduce flicker
    HOLD_SECS = 0.6
    last_valid_t = 0.0
    last_valid_dist = 999.0
    last_valid_motion = "NONE"
    last_valid_conf = 0.0
    last_valid_detected = False

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

            if data is not None:
                dist = data.get("distance", None)
                motion = str(data.get("motion", "NONE"))
                conf = float(data.get("confidence", 0.0))

                # Prefer explicit detected field if present
                if "detected" in data:
                    detected = bool(data["detected"])
                else:
                    detected = (dist is not None)

                if dist is not None:
                    dist_val = float(dist)

                if detected:
                    last_valid_t = now
                    last_valid_detected = True
                    last_valid_dist = dist_val
                    last_valid_motion = motion
                    last_valid_conf = conf

            # brief hold to reduce blinking
            if (not detected) and ((now - last_valid_t) < HOLD_SECS):
                detected = last_valid_detected
                dist_val = last_valid_dist
                motion = last_valid_motion
                conf = last_valid_conf

            # if hold expired, reset to safe empty values
            if (not detected) and ((now - last_valid_t) >= HOLD_SECS):
                dist_val = 999.0
                motion = "NONE"
                conf = 0.0

            radar_detected_pub.publish(roslibpy.Message({'data': bool(detected)}))
            radar_dist_pub.publish(roslibpy.Message({'data': float(dist_val)}))
            radar_motion_pub.publish(roslibpy.Message({'data': str(motion)}))
            radar_conf_pub.publish(roslibpy.Message({'data': float(conf)}))

    finally:
        radar_detected_pub.unadvertise()
        radar_dist_pub.unadvertise()
        radar_motion_pub.unadvertise()
        radar_conf_pub.unadvertise()
        client.terminate()
        radar_module.close()


if __name__ == "__main__":
    main()
