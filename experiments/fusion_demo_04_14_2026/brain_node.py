import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")

        self.STOP_DIST_M = 1.2
        self.CAUTION_DIST_M = 2.5
        self.RADAR_STOP_DIST_M = 1.0
        self.RADAR_CAUTION_DIST_M = 2.8

        self.CAM_CONF_GOOD = 0.75
        self.CAM_CONF_POOR = 0.40
        self.RADAR_CONF_GOOD = 0.65

        self.CAMERA_STALE_S = 1.0
        self.RADAR_STALE_S = 1.0

        self.camera_detected = False
        self.camera_label = ""
        self.camera_distance = 999.0
        self.camera_confidence = 0.0
        self._last_cam_t = 0.0
        self._last_cam_conf_t = 0.0

        self.radar_detected = False
        self.radar_distance = 999.0
        self.radar_motion = "NONE"
        self.radar_confidence = 0.0
        self._last_radar_t = 0.0
        self._last_radar_conf_t = 0.0

        self.create_subscription(Bool, "/camera_detected", self.camera_detected_cb, 10)
        self.create_subscription(String, "/camera_label", self.camera_label_cb, 10)
        self.create_subscription(Float32, "/camera_distance", self.camera_distance_cb, 10)
        self.create_subscription(Float32, "/camera_confidence", self.camera_conf_cb, 10)

        self.create_subscription(Bool, "/radar_detected", self.radar_detected_cb, 10)
        self.create_subscription(Float32, "/radar_distance", self.radar_distance_cb, 10)
        self.create_subscription(String, "/radar_motion", self.radar_motion_cb, 10)
        self.create_subscription(Float32, "/radar_confidence", self.radar_conf_cb, 10)

        self.cmd_pub = self.create_publisher(String, "/vehicle_cmd", 10)
        self.debug_pub = self.create_publisher(String, "/fusion_debug", 10)

        self.timer = self.create_timer(0.1, self.tick)

    def camera_detected_cb(self, msg: Bool):
        self.camera_detected = bool(msg.data)
        self._last_cam_t = time.monotonic()

    def camera_label_cb(self, msg: String):
        self.camera_label = str(msg.data)
        self._last_cam_t = time.monotonic()

    def camera_distance_cb(self, msg: Float32):
        self.camera_distance = float(msg.data)
        self._last_cam_t = time.monotonic()

    def camera_conf_cb(self, msg: Float32):
        self.camera_confidence = max(0.0, min(1.0, float(msg.data)))
        self._last_cam_conf_t = time.monotonic()

    def radar_detected_cb(self, msg: Bool):
        self.radar_detected = bool(msg.data)
        self._last_radar_t = time.monotonic()

    def radar_distance_cb(self, msg: Float32):
        self.radar_distance = float(msg.data)
        self._last_radar_t = time.monotonic()

    def radar_motion_cb(self, msg: String):
        self.radar_motion = str(msg.data).strip().upper()
        self._last_radar_t = time.monotonic()

    def radar_conf_cb(self, msg: Float32):
        self.radar_confidence = max(0.0, min(1.0, float(msg.data)))
        self._last_radar_conf_t = time.monotonic()

    def compute_camera_trust(self, camera_ok: bool) -> float:
        if not camera_ok:
            return 0.0
        return max(0.0, min(1.0, self.camera_confidence))

    def compute_radar_trust(self, radar_ok: bool) -> float:
        if not radar_ok:
            return 0.0
        trust = self.radar_confidence
        if self.radar_motion == "APPROACHING":
            trust += 0.20
        elif self.radar_motion == "MOVING_AWAY":
            trust += 0.05
        elif self.radar_motion == "STATIONARY":
            trust -= 0.10
        elif self.radar_motion == "NONE":
            trust -= 0.25
        return max(0.0, min(1.0, trust))

    def tick(self):
        now = time.monotonic()

        camera_ok = (now - self._last_cam_t) <= self.CAMERA_STALE_S
        cam_conf_ok = (now - self._last_cam_conf_t) <= self.CAMERA_STALE_S
        radar_ok = (now - self._last_radar_t) <= self.RADAR_STALE_S
        radar_conf_ok = (now - self._last_radar_conf_t) <= self.RADAR_STALE_S

        cam_trust = self.compute_camera_trust(camera_ok or cam_conf_ok)
        radar_trust = self.compute_radar_trust(radar_ok or radar_conf_ok)

        person = camera_ok and self.camera_detected and (self.camera_label == "Person")
        person_dist = self.camera_distance if camera_ok else 999.0
        radar_present = self.radar_detected if radar_ok else False
        radar_dist = self.radar_distance if radar_ok else 999.0
        radar_motion = self.radar_motion if radar_ok else "NONE"

        state = "SAFE"
        reason = "default_safe"

        if person and person_dist < self.STOP_DIST_M and cam_trust >= 0.35:
            state = "STOP"
            reason = "camera_close_person"
        elif person and person_dist < self.CAUTION_DIST_M and cam_trust >= 0.35:
            state = "CAUTION"
            reason = "camera_midrange_person"
        else:
            if radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_STOP_DIST_M and radar_trust >= self.RADAR_CONF_GOOD:
                state = "STOP"
                reason = "radar_close_approaching"
            elif radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_CAUTION_DIST_M:
                state = "CAUTION"
                reason = "radar_approaching"
            elif radar_present and radar_trust >= self.RADAR_CONF_GOOD and cam_trust < self.CAM_CONF_POOR:
                state = "CAUTION"
                reason = "radar_confident_camera_weak"
            else:
                state = "SAFE"
                reason = "good_visual_no_threat"

        if (
            person and person_dist < self.CAUTION_DIST_M and cam_trust >= self.CAM_CONF_GOOD
            and radar_present and radar_motion == "APPROACHING"
            and radar_dist < self.RADAR_CAUTION_DIST_M and radar_trust >= self.RADAR_CONF_GOOD
        ):
            state = "STOP"
            reason = "camera_radar_agree_close_threat"

        out = String()
        out.data = state
        self.cmd_pub.publish(out)

        dbg = String()
        dbg.data = (
            f"state={state} reason={reason} | "
            f"cam_trust={cam_trust:.2f} label={self.camera_label} detected={self.camera_detected} dist={person_dist:.2f} | "
            f"radar_trust={radar_trust:.2f} present={radar_present} radar_dist={radar_dist:.2f} motion={radar_motion}"
        )
        self.debug_pub.publish(dbg)
        self.get_logger().info(dbg.data)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
