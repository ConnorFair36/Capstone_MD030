import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
import numpy as np
import json


class BrainNode(Node):
    """
    Subscribes to:
      Original topics (for backwards compatibility):
        /person_detected   (std_msgs/Bool)
        /person_distance   (std_msgs/Float32)
        /radar_detected    (std_msgs/Bool)
        /radar_distance    (std_msgs/Float32)
        /radar_motion      (std_msgs/String)
      
      NEW Raw data topics:
        /camera_raw        (std_msgs/String) - Full color frame as JSON
        /radar_raw         (std_msgs/String) - Point cloud arrays as JSON

    Publishes:
      /vehicle_cmd       (std_msgs/String) -> "SAFE" / "CAUTION" / "STOP"
    """

    def __init__(self):
        super().__init__('brain_node')

        # ---- TUNABLE THRESHOLDS ----
        self.STOP_DIST_M = 1.2
        self.CAUTION_DIST_M = 2.5

        # Stale timeouts
        self.PERSON_STALE_S = 1.0
        self.RADAR_STALE_S = 1.0

        # ---- STATE (original) ----
        self.person_detected = False
        self.person_distance = 999.0
        self._last_person_msg_t = 0.0
        self._last_dist_msg_t = 0.0

        self.radar_detected = False
        self.radar_distance = 999.0
        self.radar_motion = "NONE"
        self._last_radar_msg_t = 0.0

        # ---- NEW: Raw data state ----
        self.camera_frame = None  # Will be numpy array (H, W, 3)
        self._last_camera_raw_t = 0.0
        
        self.radar_points = {
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([]),
            "v": np.array([])
        }
        self._last_radar_raw_t = 0.0

        # ---- SUBSCRIBERS (original) ----
        self.create_subscription(Bool, '/person_detected', self.person_cb, 10)
        self.create_subscription(Float32, '/person_distance', self.dist_cb, 10)
        self.create_subscription(Bool, '/radar_detected', self.radar_cb, 10)
        self.create_subscription(Float32, '/radar_distance', self.radar_dist_cb, 10)
        self.create_subscription(String, '/radar_motion', self.radar_motion_cb, 10)

        # ---- NEW SUBSCRIBERS (raw data) ----
        self.create_subscription(String, '/camera_raw', self.camera_raw_cb, 10)
        self.create_subscription(String, '/radar_raw', self.radar_raw_cb, 10)

        # ---- PUBLISHER ----
        self.cmd_pub = self.create_publisher(String, '/vehicle_cmd', 10)

        # Tick rate: 10 Hz
        self.timer = self.create_timer(0.1, self.tick)

    # ---------- Original Callbacks ----------
    def person_cb(self, msg: Bool):
        self.person_detected = bool(msg.data)
        self._last_person_msg_t = time.monotonic()

    def dist_cb(self, msg: Float32):
        self.person_distance = float(msg.data)
        self._last_dist_msg_t = time.monotonic()

    def radar_cb(self, msg: Bool):
        self.radar_detected = bool(msg.data)
        self._last_radar_msg_t = time.monotonic()

    def radar_dist_cb(self, msg: Float32):
        self.radar_distance = float(msg.data)
        self._last_radar_msg_t = time.monotonic()

    def radar_motion_cb(self, msg: String):
        self.radar_motion = str(msg.data)
        self._last_radar_msg_t = time.monotonic()

    # ---------- NEW: Raw Data Callbacks ----------
    def camera_raw_cb(self, msg: String):
        """
        Receives JSON string containing:
          { "frame": [...], "shape": [h, w, c], "dtype": "uint8" }
        Reconstructs numpy array.
        """
        try:
            data = json.loads(msg.data)
            frame_list = data["frame"]
            shape = tuple(data["shape"])
            dtype = data["dtype"]
            
            # Reconstruct numpy array
            self.camera_frame = np.array(frame_list, dtype=dtype).reshape(shape)
            self._last_camera_raw_t = time.monotonic()
            
            self.get_logger().info(
                f"Received camera frame: shape={self.camera_frame.shape}, "
                f"dtype={self.camera_frame.dtype}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error parsing camera_raw: {e}")

    def radar_raw_cb(self, msg: String):
        """
        Receives JSON string containing:
          { "x": [...], "y": [...], "z": [...], "v": [...], "count": N }
        Reconstructs numpy arrays.
        """
        try:
            data = json.loads(msg.data)
            
            # Convert lists to numpy arrays
            self.radar_points = {
                "x": np.array(data["x"], dtype=np.float32),
                "y": np.array(data["y"], dtype=np.float32),
                "z": np.array(data["z"], dtype=np.float32),
                "v": np.array(data["v"], dtype=np.float32)
            }
            self._last_radar_raw_t = time.monotonic()
            
            count = data["count"]
            self.get_logger().info(
                f"Received radar points: count={count}, "
                f"shapes=({len(self.radar_points['x'])}, "
                f"{len(self.radar_points['y'])}, "
                f"{len(self.radar_points['z'])}, "
                f"{len(self.radar_points['v'])})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error parsing radar_raw: {e}")

    # ---------- Decision Logic ----------
    def tick(self):
        now = time.monotonic()

        # Check if inputs are stale
        person_ok = (now - self._last_person_msg_t) <= self.PERSON_STALE_S
        dist_ok = (now - self._last_dist_msg_t) <= self.PERSON_STALE_S
        radar_ok = (now - self._last_radar_msg_t) <= self.RADAR_STALE_S

        person = self.person_detected if person_ok else False
        dist = self.person_distance if dist_ok else 999.0

        radar_present = (self.radar_detected if radar_ok else False)
        radar_motion = (self.radar_motion if radar_ok else "NONE")

        # ---- Core SAFE/CAUTION/STOP logic ----
        if person and dist < self.STOP_DIST_M:
            state = "STOP"
        elif person and dist < self.CAUTION_DIST_M:
            state = "CAUTION"
        else:
            # No close person seen by camera
            # Use radar as secondary confirmation
            if radar_present or (radar_motion not in ("NONE", "", "STATIONARY")):
                state = "CAUTION"
            else:
                state = "SAFE"

        out = String()
        out.data = state
        self.cmd_pub.publish(out)

        # ---- Log raw data availability ----
        camera_raw_ok = (now - self._last_camera_raw_t) <= 1.0
        radar_raw_ok = (now - self._last_radar_raw_t) <= 1.0
        
        self.get_logger().info(
            f"vehicle_cmd={state} | person={person} dist={dist:.2f} | "
            f"radar={radar_present} motion={radar_motion} | "
            f"camera_raw={'OK' if camera_raw_ok else 'STALE'} "
            f"radar_raw={'OK' if radar_raw_ok else 'STALE'}"
        )

    # ---------- Helper Methods for Processing Raw Data ----------
    def get_camera_frame(self):
        """Returns the latest camera frame as numpy array, or None if stale/unavailable."""
        now = time.monotonic()
        if (now - self._last_camera_raw_t) <= 1.0:
            return self.camera_frame
        return None

    def get_radar_points(self):
        """Returns the latest radar point cloud as dict of numpy arrays, or None if stale/unavailable."""
        now = time.monotonic()
        if (now - self._last_radar_raw_t) <= 1.0:
            return self.radar_points
        return None


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
