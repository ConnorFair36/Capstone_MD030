import time
import roslibpy
import radar_module  # uses COM ports + profile.cfg

def main():
    radar_module.initialize()

    # Same pattern as camera_to_rosbridge.py
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    radar_detected_pub = roslibpy.Topic(client, '/radar_detected', 'std_msgs/Bool')
    radar_dist_pub     = roslibpy.Topic(client, '/radar_distance', 'std_msgs/Float32')
    radar_motion_pub   = roslibpy.Topic(client, '/radar_motion', 'std_msgs/String')

    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ
    last_pub = 0.0

    # simple persistence so it doesn't flicker
    last_seen_t = 0.0
    HOLD_SECS = 0.6

    try:
        while True:
            data = radar_module.get_radar_data()

            now = time.time()
            if now - last_pub < pub_period:
                continue
            last_pub = now

            detected = False
            dist_val = 999.0
            motion = "NONE"

            if data is not None:
                motion = str(data.get("motion", "NONE"))
                dist = data.get("distance", None)

                if dist is not None:
                    dist_val = float(dist)

                # treat radar as "detected" only if it isn't NONE and has a distance
                if motion != "NONE" and dist is not None:
                    detected = True
                    last_seen_t = now

            # hold detected true briefly to reduce blinking
            if not detected and (now - last_seen_t) < HOLD_SECS:
                detected = True

            radar_detected_pub.publish(roslibpy.Message({'data': bool(detected)}))
            radar_dist_pub.publish(roslibpy.Message({'data': float(dist_val)}))
            radar_motion_pub.publish(roslibpy.Message({'data': motion}))

    finally:
        radar_detected_pub.unadvertise()
        radar_dist_pub.unadvertise()
        radar_motion_pub.unadvertise()
        client.terminate()

if __name__ == "__main__":
    main()