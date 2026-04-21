import time
import roslibpy
import camera_module_raw
import numpy as np

def main():
    # 1) Start camera + mediapipe
    camera_module_raw.initialize()

    # 2) Connect to rosbridge (WSL)
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    # Original topics for backwards compatibility
    person_pub = roslibpy.Topic(client, '/person_detected', 'std_msgs/Bool')
    dist_pub = roslibpy.Topic(client, '/person_distance', 'std_msgs/Float32')
    
    # NEW: Raw camera data topic
    # Sends: { "frame": [...], "shape": [h, w, c], "dtype": "uint8" }
    camera_raw_pub = roslibpy.Topic(client, '/camera_raw', 'std_msgs/String')

    # Slow down publish rate
    last_pub = 0.0
    PUB_HZ = 10.0
    pub_period = 1.0 / PUB_HZ

    try:
        while True:
            # Get camera data and the raw frame
            data = camera_module_raw.get_camera_data()
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

            # Publish person_detected (backwards compatibility)
            person_pub.publish(roslibpy.Message({'data': person}))

            # Publish distance
            if distance is None:
                distance_val = 999.0
            else:
                distance_val = float(distance)
            dist_pub.publish(roslibpy.Message({'data': distance_val}))

            # NEW: Publish raw color frame
            # Get the frame from camera_module
            # We need to modify camera_module to return the frame,
            # or we can access the pipeline directly here
            
            # Get fresh frame for raw data
            frames = camera_module_raw.pipeline.wait_for_frames()
            aligned = camera_module_raw.align.process(frames)
            color_frame = aligned.get_color_frame()
            
            if color_frame:
                color_img = np.asanyarray(color_frame.get_data())
                
                # Convert numpy array to JSON-serializable format
                raw_data = {
                    "frame": color_img.flatten().tolist(),  # Flatten and convert to list
                    "shape": list(color_img.shape),         # [height, width, channels]
                    "dtype": str(color_img.dtype)           # "uint8"
                }
                
                # Convert to JSON string
                import json
                raw_json = json.dumps(raw_data)
                
                camera_raw_pub.publish(roslibpy.Message({'data': raw_json}))

    finally:
        person_pub.unadvertise()
        dist_pub.unadvertise()
        camera_raw_pub.unadvertise()
        client.terminate()

if __name__ == "__main__":
    main()
