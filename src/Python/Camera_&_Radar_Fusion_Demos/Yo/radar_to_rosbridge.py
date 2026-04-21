import time
import numpy as np
import roslibpy
import radar_module

# ---- ROS Bridge Setup ----
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

# Publish raw radar array as Float32MultiArray
radar_array_pub = roslibpy.Topic(client, '/radar_raw_array', 'std_msgs/Float32MultiArray')

# ---- Constants ----
PUBLISH_HZ = 10.0
PUB_PERIOD = 1.0 / PUBLISH_HZ

# ---- Initialize Radar ----
radar_module.initialize()
print("Radar ROS bridge running...")

last_pub = 0.0

try:
    while True:
        now = time.time()
        if now - last_pub < PUB_PERIOD:
            continue
        last_pub = now

        # Get regular raw radar array: shape (N, 4)
        radar_data = radar_module.get_radar_data_raw()

        if radar_data is None or len(radar_data) == 0:
            continue

        points = np.array(radar_data, dtype=np.float32)

        rows = int(points.shape[0])
        cols = int(points.shape[1])  # should be 4

        msg = {
            'layout': {
                'dim': [
                    {
                        'label': 'points',
                        'size': rows,
                        'stride': rows * cols
                    },
                    {
                        'label': 'fields',
                        'size': cols,
                        'stride': cols
                    }
                ],
                'data_offset': 0
            },
            'data': points.flatten().tolist()
        }

        radar_array_pub.publish(roslibpy.Message(msg))
        print("Published raw radar array:")
        print(points)

except KeyboardInterrupt:
    print("Shutting down radar ROS bridge...")

finally:
    radar_array_pub.unadvertise()
    client.terminate()