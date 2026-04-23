import time
import roslibpy
import camera_module  # your file

def main():
    # 1) Start camera + mediapipe
    camera_module.initialize()

    # 2) Connect to rosbridge (WSL)
    # If localhost doesn't work, replace with WSL IP (see note below)
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    person_pub = roslibpy.Topic(client, '/person_detected', 'std_msgs/Bool')
    dist_pub   = roslibpy.Topic(client, '/person_distance', 'std_msgs/Float32')

    # Optional: slow down publish rate a bit
    last_pub = 0.0
    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ

    try:
        while True:
            data = camera_module.get_camera_data()
            if data is None:
                continue
            if data == "QUIT":
                break

            now = time.time()
            if now - last_pub < pub_period:
                continue
            last_pub = now

            person = bool(data.get("person", False))
            distance = data.get("distance", None)

            # Publish person_detected
            person_pub.publish(roslibpy.Message({'data': person}))

            # Publish distance (Float32 must be a number)
            # If distance is None, publish a big number so brain keeps GO
            if distance is None:
                distance_val = 999.0
            else:
                distance_val = float(distance)

            dist_pub.publish(roslibpy.Message({'data': distance_val}))

    finally:
        person_pub.unadvertise()
        dist_pub.unadvertise()
        client.terminate()

if __name__ == "__main__":
    main()