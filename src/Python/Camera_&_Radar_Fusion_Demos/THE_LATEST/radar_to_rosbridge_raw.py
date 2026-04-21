import time
from collections import deque
import roslibpy
import radar_module_raw
import json

def main():
    radar_module_raw.initialize()

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    # Original topics for backwards compatibility
    radar_detected_pub = roslibpy.Topic(client, '/radar_detected', 'std_msgs/Bool')
    radar_dist_pub = roslibpy.Topic(client, '/radar_distance', 'std_msgs/Float32')
    radar_motion_pub = roslibpy.Topic(client, '/radar_motion', 'std_msgs/String')
    
    # NEW: Raw radar point cloud topic
    # Sends: { "x": [...], "y": [...], "z": [...], "v": [...], "count": N }
    radar_raw_pub = roslibpy.Topic(client, '/radar_raw', 'std_msgs/String')
    
    # NEW: Last 3 radar points topic
    radar_last3_pub = roslibpy.Topic(client, '/radar_last3', 'std_msgs/String')

    # Queue to store last 3 points
    last3_queue = deque(maxlen=3)

    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ
    last_pub = 0.0

    # Simple persistence
    last_seen_t = 0.0
    HOLD_SECS = 0.6

    try:
        while True:
            data = radar_module_raw.get_radar_data()

            now = time.time()
            if now - last_pub < pub_period:
                continue
            last_pub = now

            detected = False
            dist_val = 999.0
            motion = "NONE"
            
            # Raw arrays to send
            x_array = []
            y_array = []
            z_array = []
            v_array = []

            if data is not None:
                motion = str(data.get("motion", "NONE"))
                dist = data.get("distance", None)

                if dist is not None:
                    dist_val = float(dist)

                if motion != "NONE" and dist is not None:
                    detected = True
                    last_seen_t = now
                    
                # Get raw point cloud from radar
                # We need to modify radar_module.get_radar_data() to also return raw arrays
                # OR we can parse the frame again here
                # For now, let's add a helper function to radar_module

            # Hold detected briefly
            if not detected and (now - last_seen_t) < HOLD_SECS:
                detected = True

            # Publish backwards-compatible topics
            radar_detected_pub.publish(roslibpy.Message({'data': bool(detected)}))
            radar_dist_pub.publish(roslibpy.Message({'data': float(dist_val)}))
            radar_motion_pub.publish(roslibpy.Message({'data': motion}))
            
            # NEW: Publish raw point cloud data
            # We need to get the raw x, y, z, v arrays from the last parsed frame
            # Let's create a modified version that stores this
            
            # For now, we'll call a helper that returns raw arrays
            raw_points = radar_module_raw.get_last_raw_points()
            
            if raw_points:
                raw_data = {
                    "x": raw_points["x"],
                    "y": raw_points["y"],
                    "z": raw_points["z"],
                    "v": raw_points["v"],
                    "count": len(raw_points["x"])
                }
                
                raw_json = json.dumps(raw_data)
                radar_raw_pub.publish(roslibpy.Message({'data': raw_json}))
            else:
                # No points detected, send empty arrays
                empty_data = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "v": [],
                    "count": 0
                }
                raw_json = json.dumps(empty_data)
                radar_raw_pub.publish(roslibpy.Message({'data': raw_json}))
            if raw_points and raw_points["x"]:
                for i in range(len(raw_points["x"])):
                    point = {
                        "x": raw_points["x"][i],
                        "y": raw_points["y"][i],
                        "z": raw_points["z"][i],
                        "v": raw_points["v"][i]
                            }
                    last3_queue.append(point)
                last3_data = {
                "points": list(last3_queue),
                "count": len(last3_queue)
                    }
                radar_last3_pub.publish(roslibpy.Message({'data': json.dumps(last3_data)}))

    finally:
        radar_detected_pub.unadvertise()
        radar_dist_pub.unadvertise()
        radar_motion_pub.unadvertise()
        radar_raw_pub.unadvertise()
        radar_last3_pub.unadvertise()
        client.terminate()

if __name__ == "__main__":
    main()
