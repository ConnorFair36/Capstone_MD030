import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class BrainNode(Node):
    """
    Subscribes:
      /person_detected   (std_msgs/Bool)
      /person_distance   (std_msgs/Float32)
      /radar_detected    (std_msgs/Bool)      [NEW]
      /radar_distance    (std_msgs/Float32)   [NEW, optional]
      /radar_motion      (std_msgs/String)    [NEW, optional]

    Publishes:
      /vehicle_cmd       (std_msgs/String) -> "SAFE" / "CAUTION" / "STOP"
    """

    def __init__(self):
        super().__init__('brain_node')

        # ---- TUNABLE THRESHOLDS (demo-friendly defaults) ----
        self.STOP_DIST_M = 1.2
        self.CAUTION_DIST_M = 2.5

        # "stale" timeouts: if we haven't heard recently, treat as false
        self.PERSON_STALE_S = 1.0
        self.RADAR_STALE_S = 1.0

        # ---- STATE ----
        self.person_detected = False
        self.person_distance = 999.0
        self._last_person_msg_t = 0.0
        self._last_dist_msg_t = 0.0

        self.radar_detected = False
        self.radar_distance = 999.0
        self.radar_motion = "NONE"
        self._last_radar_msg_t = 0.0

        # ---- SUBS ----
        self.create_subscription(Bool, '/person_detected', self.person_cb, 10)
        self.create_subscription(Float32, '/person_distance', self.dist_cb, 10)

        # Radar topics (new)
        self.create_subscription(Bool, '/radar_detected', self.radar_cb, 10)
        self.create_subscription(Float32, '/radar_distance', self.radar_dist_cb, 10)
        self.create_subscription(String, '/radar_motion', self.radar_motion_cb, 10)

        # ---- PUB ----
        self.cmd_pub = self.create_publisher(String, '/vehicle_cmd', 10)

        # Tick rate: 10 Hz looks responsive on-screen
        self.timer = self.create_timer(0.1, self.tick)

    # ---------- Callbacks ----------
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
        # treat distance update as radar activity too
        self._last_radar_msg_t = time.monotonic()

    def radar_motion_cb(self, msg: String):
        self.radar_motion = str(msg.data)
        self._last_radar_msg_t = time.monotonic()

    # ---------- Decision ----------
    def tick(self):
        now = time.monotonic()

        # If inputs are stale, treat them as "not detected"
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
            # No (close) person seen by camera.
            # Use radar only as a secondary confidence channel.
            # If radar is confidently seeing something (or motion != NONE), escalate to CAUTION.
            if radar_present or (radar_motion not in ("NONE", "", "STATIONARY")):
                state = "CAUTION"
            else:
                state = "SAFE"

        out = String()
        out.data = state
        self.cmd_pub.publish(out)

        self.get_logger().info(
            f"vehicle_cmd={state} | person={person} dist={dist:.2f} | radar={radar_present} motion={radar_motion}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()