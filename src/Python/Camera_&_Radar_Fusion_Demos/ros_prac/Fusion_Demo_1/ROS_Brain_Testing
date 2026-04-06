import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class BrainNode(Node):
    """
    Subscribes:
      /person_detected      (std_msgs/Bool)
      /person_distance      (std_msgs/Float32)
      /camera_confidence    (std_msgs/Float32)   # 0.0 to 1.0
      /camera_low_light     (std_msgs/Bool)
      /camera_bad_weather   (std_msgs/Bool)      # rain/fog/glare/etc.

      /radar_detected       (std_msgs/Bool)
      /radar_distance       (std_msgs/Float32)
      /radar_motion         (std_msgs/String)    # APPROACHING / MOVING_AWAY / STATIONARY / NONE
      /radar_confidence     (std_msgs/Float32)   # 0.0 to 1.0

    Publishes:
      /vehicle_cmd          (std_msgs/String) -> SAFE / CAUTION / STOP
      /fusion_debug         (std_msgs/String) -> optional human-readable reason
    """

    def __init__(self):
        super().__init__('brain_node')

        # -----------------------------
        # TUNABLE THRESHOLDS
        # -----------------------------
        self.STOP_DIST_M = 1.2
        self.CAUTION_DIST_M = 2.5

        # Radar-only thresholds
        self.RADAR_STOP_DIST_M = 1.0
        self.RADAR_CAUTION_DIST_M = 2.8

        # Confidence thresholds
        self.CAM_CONF_GOOD = 0.75
        self.CAM_CONF_POOR = 0.40
        self.RADAR_CONF_GOOD = 0.65
        self.RADAR_CONF_POOR = 0.30

        # stale timeouts
        self.PERSON_STALE_S = 1.0
        self.CAMERA_STALE_S = 1.0
        self.RADAR_STALE_S = 1.0

        # -----------------------------
        # CAMERA STATE
        # -----------------------------
        self.person_detected = False
        self.person_distance = 999.0
        self.camera_confidence = 0.0
        self.camera_low_light = False
        self.camera_bad_weather = False

        self._last_person_msg_t = 0.0
        self._last_dist_msg_t = 0.0
        self._last_cam_conf_t = 0.0
        self._last_cam_env_t = 0.0

        # -----------------------------
        # RADAR STATE
        # -----------------------------
        self.radar_detected = False
        self.radar_distance = 999.0
        self.radar_motion = "NONE"
        self.radar_confidence = 0.0

        self._last_radar_msg_t = 0.0
        self._last_radar_conf_t = 0.0

        # -----------------------------
        # SUBSCRIBERS
        # -----------------------------
        self.create_subscription(Bool, '/person_detected', self.person_cb, 10)
        self.create_subscription(Float32, '/person_distance', self.dist_cb, 10)
        self.create_subscription(Float32, '/camera_confidence', self.camera_conf_cb, 10)
        self.create_subscription(Bool, '/camera_low_light', self.camera_low_light_cb, 10)
        self.create_subscription(Bool, '/camera_bad_weather', self.camera_bad_weather_cb, 10)

        self.create_subscription(Bool, '/radar_detected', self.radar_cb, 10)
        self.create_subscription(Float32, '/radar_distance', self.radar_dist_cb, 10)
        self.create_subscription(String, '/radar_motion', self.radar_motion_cb, 10)
        self.create_subscription(Float32, '/radar_confidence', self.radar_conf_cb, 10)

        # -----------------------------
        # PUBLISHERS
        # -----------------------------
        self.cmd_pub = self.create_publisher(String, '/vehicle_cmd', 10)
        self.debug_pub = self.create_publisher(String, '/fusion_debug', 10)

        self.timer = self.create_timer(0.1, self.tick)

    # =========================================================
    # CALLBACKS
    # =========================================================
    def person_cb(self, msg: Bool):
        self.person_detected = bool(msg.data)
        self._last_person_msg_t = time.monotonic()

    def dist_cb(self, msg: Float32):
        self.person_distance = float(msg.data)
        self._last_dist_msg_t = time.monotonic()

    def camera_conf_cb(self, msg: Float32):
        self.camera_confidence = max(0.0, min(1.0, float(msg.data)))
        self._last_cam_conf_t = time.monotonic()

    def camera_low_light_cb(self, msg: Bool):
        self.camera_low_light = bool(msg.data)
        self._last_cam_env_t = time.monotonic()

    def camera_bad_weather_cb(self, msg: Bool):
        self.camera_bad_weather = bool(msg.data)
        self._last_cam_env_t = time.monotonic()

    def radar_cb(self, msg: Bool):
        self.radar_detected = bool(msg.data)
        self._last_radar_msg_t = time.monotonic()

    def radar_dist_cb(self, msg: Float32):
        self.radar_distance = float(msg.data)
        self._last_radar_msg_t = time.monotonic()

    def radar_motion_cb(self, msg: String):
        self.radar_motion = str(msg.data).strip().upper()
        self._last_radar_msg_t = time.monotonic()

    def radar_conf_cb(self, msg: Float32):
        self.radar_confidence = max(0.0, min(1.0, float(msg.data)))
        self._last_radar_conf_t = time.monotonic()

    # =========================================================
    # TRUST MODEL
    # =========================================================
    def compute_camera_trust(self, camera_ok: bool) -> float:
        """
        Camera trust should drop in:
        - stale data
        - poor confidence
        - low light
        - bad weather / glare / fog / rain
        """
        if not camera_ok:
            return 0.0

        trust = self.camera_confidence

        if self.camera_low_light:
            trust -= 0.30

        if self.camera_bad_weather:
            trust -= 0.25

        # Clamp
        trust = max(0.0, min(1.0, trust))
        return trust

    def compute_radar_trust(self, radar_ok: bool) -> float:
        """
        Radar trust is less affected by light, but we still avoid blindly
        trusting stationary clutter.
        """
        if not radar_ok:
            return 0.0

        trust = self.radar_confidence

        # motion-based boost/penalty
        if self.radar_motion == "APPROACHING":
            trust += 0.20
        elif self.radar_motion == "MOVING_AWAY":
            trust += 0.05
        elif self.radar_motion == "STATIONARY":
            trust -= 0.15
        elif self.radar_motion == "NONE":
            trust -= 0.25

        trust = max(0.0, min(1.0, trust))
        return trust

    # =========================================================
    # DECISION LOGIC
    # =========================================================
    def tick(self):
        now = time.monotonic()

        # -----------------------------
        # Freshness / stale handling
        # -----------------------------
        person_ok = (now - self._last_person_msg_t) <= self.PERSON_STALE_S
        dist_ok = (now - self._last_dist_msg_t) <= self.CAMERA_STALE_S
        cam_conf_ok = (now - self._last_cam_conf_t) <= self.CAMERA_STALE_S
        cam_env_ok = (now - self._last_cam_env_t) <= self.CAMERA_STALE_S

        radar_ok = (now - self._last_radar_msg_t) <= self.RADAR_STALE_S
        radar_conf_ok = (now - self._last_radar_conf_t) <= self.RADAR_STALE_S

        # Camera usable if detections are fresh enough
        camera_ok = person_ok or dist_ok or cam_conf_ok or cam_env_ok

        # Radar usable if radar messages are fresh
        radar_stream_ok = radar_ok or radar_conf_ok

        # -----------------------------
        # Effective values
        # -----------------------------
        person = self.person_detected if person_ok else False
        person_dist = self.person_distance if dist_ok else 999.0

        radar_present = self.radar_detected if radar_ok else False
        radar_dist = self.radar_distance if radar_ok else 999.0
        radar_motion = self.radar_motion if radar_ok else "NONE"

        cam_trust = self.compute_camera_trust(camera_ok)
        radar_trust = self.compute_radar_trust(radar_stream_ok)

        # Useful flags
        visual_bad = self.camera_low_light or self.camera_bad_weather or (cam_trust < self.CAM_CONF_POOR)
        radar_strong = radar_trust >= self.RADAR_CONF_GOOD
        camera_strong = cam_trust >= self.CAM_CONF_GOOD

        # -----------------------------
        # FUSION DECISION
        # -----------------------------
        state = "SAFE"
        reason = "default_safe"

        # 1) Highest priority: camera sees close person with decent trust
        if person and person_dist < self.STOP_DIST_M and cam_trust >= 0.35:
            state = "STOP"
            reason = "camera_close_person"

        elif person and person_dist < self.CAUTION_DIST_M and cam_trust >= 0.35:
            state = "CAUTION"
            reason = "camera_midrange_person"

        else:
            # 2) If visual conditions are bad, allow radar to carry more weight
            if visual_bad:
                if radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_STOP_DIST_M and radar_strong:
                    state = "STOP"
                    reason = "radar_close_approaching_in_bad_visual"
                elif radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_CAUTION_DIST_M:
                    state = "CAUTION"
                    reason = "radar_approaching_in_bad_visual"
                elif radar_present and radar_strong:
                    state = "CAUTION"
                    reason = "radar_present_in_bad_visual"
                else:
                    state = "SAFE"
                    reason = "bad_visual_but_no_strong_radar"

            else:
                # 3) Good visual conditions:
                #    camera remains primary, radar is supportive only
                if radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_CAUTION_DIST_M:
                    if not person:
                        state = "CAUTION"
                        reason = "radar_approaching_support"
                    else:
                        state = "CAUTION"
                        reason = "camera_radar_both_support"
                elif radar_present and radar_strong and not camera_strong:
                    state = "CAUTION"
                    reason = "radar_confident_camera_weak"
                else:
                    state = "SAFE"
                    reason = "good_visual_no_threat"

        # Optional extra rule:
        # If both sensors strongly agree that something is close, force STOP
        if (
            person and person_dist < self.CAUTION_DIST_M and camera_strong and
            radar_present and radar_motion == "APPROACHING" and radar_dist < self.RADAR_CAUTION_DIST_M and radar_strong
        ):
            state = "STOP"
            reason = "camera_radar_agree_close_threat"

        # -----------------------------
        # Publish outputs
        # -----------------------------
        out = String()
        out.data = state
        self.cmd_pub.publish(out)

        dbg = String()
        dbg.data = (
            f"state={state} reason={reason} | "
            f"cam_trust={cam_trust:.2f} person={person} person_dist={person_dist:.2f} "
            f"low_light={self.camera_low_light} bad_weather={self.camera_bad_weather} | "
            f"radar_trust={radar_trust:.2f} radar_present={radar_present} "
            f"radar_dist={radar_dist:.2f} radar_motion={radar_motion}"
        )
        self.debug_pub.publish(dbg)

        self.get_logger().info(dbg.data)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
