import base64
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import roslibpy
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FX = 641.60986328125
FY = 640.8302001953125
CX = 650.4204711914062
CY = 405.4923400878906
IMG_W = 1280
IMG_H = 720

T_RADAR_TO_CAM = np.array([0.04571, 0.03764, 0.02418], dtype=np.float32)
R_RADAR_TO_CAM = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float32)


@dataclass
class AppState:
    camera_detected: bool = False
    camera_label: str = ""
    camera_distance: float | None = None
    camera_confidence: float = 0.0
    detections: list[dict[str, Any]] = field(default_factory=list)
    frame_bgr: np.ndarray | None = None
    depth_frame_bgr: np.ndarray | None = None
    last_camera_t: float = 0.0

    radar_detected: bool = False
    radar_distance: float | None = None
    radar_motion: str = "NONE"
    radar_confidence: float = 0.0
    radar_points: list[dict[str, Any]] = field(default_factory=list)
    last_radar_t: float = 0.0

    vehicle_state: str = "SAFE"
    fusion_debug: str = "Waiting for /fusion_debug"
    last_vehicle_t: float = 0.0

    rosbridge_connected: bool = False

    show_boxes: bool = True
    show_labels: bool = True
    show_distance: bool = True
    show_confidence: bool = True
    show_center_point: bool = True
    show_legend: bool = True
    show_nothing_detected: bool = True
    show_radar_summary_box: bool = True
    show_projected_radar_points: bool = True
    show_radar_match_text: bool = True
    show_camera_velocity: bool = False
    show_camera_velocity_components: bool = False

    show_bev_velocity_arrows: bool = True
    show_bev_camera_detections: bool = True

    show_person: bool = True
    show_backpack: bool = True
    show_chair: bool = True
    show_laptop: bool = True
    show_phone: bool = True


CLASS_KEYS = {
    "Person": "show_person",
    "Backpack": "show_backpack",
    "Chair": "show_chair",
    "Laptop": "show_laptop",
    "Phone": "show_phone",
}


class RosListener:
    def __init__(self, state: AppState):
        self.state = state
        self.client = roslibpy.Ros(host="localhost", port=9090)
        self.client.run()
        self.state.rosbridge_connected = bool(self.client.is_connected)

        self.subs = [
            roslibpy.Topic(self.client, "/camera_detected", "std_msgs/Bool"),
            roslibpy.Topic(self.client, "/camera_label", "std_msgs/String"),
            roslibpy.Topic(self.client, "/camera_distance", "std_msgs/Float32"),
            roslibpy.Topic(self.client, "/camera_confidence", "std_msgs/Float32"),
            roslibpy.Topic(self.client, "/camera_detections_json", "std_msgs/String"),
            roslibpy.Topic(self.client, "/camera_frame_b64", "std_msgs/String"),
            roslibpy.Topic(self.client, "/camera_depth_b64", "std_msgs/String"),
            roslibpy.Topic(self.client, "/radar_detected", "std_msgs/Bool"),
            roslibpy.Topic(self.client, "/radar_distance", "std_msgs/Float32"),
            roslibpy.Topic(self.client, "/radar_motion", "std_msgs/String"),
            roslibpy.Topic(self.client, "/radar_confidence", "std_msgs/Float32"),
            roslibpy.Topic(self.client, "/radar_points_json", "std_msgs/String"),
            roslibpy.Topic(self.client, "/vehicle_cmd", "std_msgs/String"),
            roslibpy.Topic(self.client, "/fusion_debug", "std_msgs/String"),
        ]

        self.subs[0].subscribe(lambda msg: self._setattr("camera_detected", bool(msg.get("data", False)), "camera"))
        self.subs[1].subscribe(lambda msg: self._setattr("camera_label", str(msg.get("data", "")), "camera"))
        self.subs[2].subscribe(lambda msg: self._setattr("camera_distance", float(msg.get("data", 0.0)), "camera"))
        self.subs[3].subscribe(lambda msg: self._setattr("camera_confidence", float(msg.get("data", 0.0)), "camera"))
        self.subs[4].subscribe(self._on_detections)
        self.subs[5].subscribe(self._on_frame)
        self.subs[6].subscribe(self._on_depth_frame)
        self.subs[7].subscribe(lambda msg: self._setattr("radar_detected", bool(msg.get("data", False)), "radar"))
        self.subs[8].subscribe(lambda msg: self._setattr("radar_distance", float(msg.get("data", 0.0)), "radar"))
        self.subs[9].subscribe(lambda msg: self._setattr("radar_motion", str(msg.get("data", "NONE")), "radar"))
        self.subs[10].subscribe(lambda msg: self._setattr("radar_confidence", float(msg.get("data", 0.0)), "radar"))
        self.subs[11].subscribe(self._on_radar_points)
        self.subs[12].subscribe(lambda msg: self._setattr("vehicle_state", str(msg.get("data", "SAFE")).upper(), "vehicle"))
        self.subs[13].subscribe(self._on_fusion_debug)

    def _setattr(self, key: str, value: Any, kind: str):
        setattr(self.state, key, value)
        now = time.time()
        if kind == "camera":
            self.state.last_camera_t = now
        elif kind == "radar":
            self.state.last_radar_t = now
        elif kind == "vehicle":
            self.state.last_vehicle_t = now

    def _on_detections(self, msg):
        try:
            self.state.detections = json.loads(msg.get("data", "[]"))
        except Exception:
            self.state.detections = []
        self.state.last_camera_t = time.time()

    def _on_radar_points(self, msg):
        try:
            self.state.radar_points = json.loads(msg.get("data", "[]"))
        except Exception:
            self.state.radar_points = []
        self.state.last_radar_t = time.time()

    def _on_frame(self, msg):
        data = msg.get("data", "")
        if not data:
            return
        try:
            jpg = base64.b64decode(data)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.state.frame_bgr = frame
                self.state.last_camera_t = time.time()
        except Exception:
            pass

    def _on_depth_frame(self, msg):
        data = msg.get("data", "")
        if not data:
            return
        try:
            jpg = base64.b64decode(data)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.state.depth_frame_bgr = frame
                self.state.last_camera_t = time.time()
        except Exception:
            pass

    def _on_fusion_debug(self, msg):
        self.state.fusion_debug = str(msg.get("data", ""))
        self.state.last_vehicle_t = time.time()

    def close(self):
        for sub in self.subs:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self.client.terminate()
        self.state.rosbridge_connected = False


def radar_point_to_pixel(x_r: float, y_r: float, z_r: float):
    p_r = np.array([x_r, y_r, z_r], dtype=np.float32)
    Xc, Yc, Zc = (R_RADAR_TO_CAM @ p_r + T_RADAR_TO_CAM).tolist()
    if Zc <= 1e-6:
        return None
    u = int(round(FX * (Xc / Zc) + CX))
    v = int(round(FY * (Yc / Zc) + CY))
    if 0 <= u < IMG_W and 0 <= v < IMG_H:
        return {"u": u, "v": v, "Xc": Xc, "Yc": Yc, "Zc": Zc}
    return None


def velocity_to_motion(v: float) -> str:
    if v < -0.05:
        return "APPROACHING"
    if v > 0.05:
        return "MOVING_AWAY"
    return "STATIONARY"


def velocity_to_color(v: float):
    if v < -0.05:
        return (0, 0, 255)
    if v > 0.05:
        return (255, 0, 0)
    return (255, 255, 255)


def visible_detections(state: AppState) -> list[dict[str, Any]]:
    out = []
    for det in state.detections:
        key = CLASS_KEYS.get(det.get("label", ""))
        if key and not getattr(state, key):
            continue
        out.append(det)
    return out


def visible_primary_detection(state: AppState) -> dict[str, Any] | None:
    dets = visible_detections(state)
    if not dets:
        return None
    best = None
    best_metric = None
    for det in dets:
        dist = det.get("distance")
        conf = float(det.get("confidence", 0.0))
        metric = (0, float(dist), -conf) if dist is not None else (1, -conf)
        if best is None or metric < best_metric:
            best = det
            best_metric = metric
    return best


def projected_radar_points(state: AppState) -> list[dict[str, Any]]:
    points = []
    for pt in state.radar_points:
        try:
            x = float(pt["x"])
            y = float(pt["y"])
            z = float(pt["z"])
            v = float(pt.get("v", 0.0))
            r = float(pt.get("range", 0.0))
        except Exception:
            continue
        proj = radar_point_to_pixel(x, y, z)
        if proj is None:
            continue
        points.append({
            **proj,
            "vel": v,
            "motion": velocity_to_motion(v),
            "range": r,
            "color": velocity_to_color(v),
        })
    return points


def attach_radar_to_detections(detections: list[dict[str, Any]], proj_points: list[dict[str, Any]]):
    out = []
    for det in detections:
        det = dict(det)
        matches = []
        x1, y1, x2, y2 = det["bbox"]
        for rp in proj_points:
            if x1 <= rp["u"] <= x2 and y1 <= rp["v"] <= y2:
                matches.append(rp)
        matches.sort(key=lambda p: (-abs(p["vel"]), p["range"]))
        det["best_radar"] = matches[0] if matches else None
        out.append(det)
    return out


BEV_W = 420
BEV_H = 420
BEV_X_RANGE = 5.0
BEV_Y_RANGE = 10.0
BEV_BG = (20, 20, 20)
BEV_GRID = (50, 50, 50)
BEV_ARROW_SCALE = 18.0

R_CAM_TO_RADAR = R_RADAR_TO_CAM.T
T_CAM_TO_RADAR = -R_CAM_TO_RADAR @ T_RADAR_TO_CAM


def bev_project(x_m: float, y_m: float) -> tuple[int, int] | None:
    """Map radar-frame (x, y) metres to BEV canvas pixel coords.

    Origin (radar) is at bottom-centre; +y goes up (forward from radar).
    Returns None when the point falls outside the visible area.
    """
    px = int(round((x_m / BEV_X_RANGE + 1.0) * 0.5 * BEV_W))
    py = int(round((1.0 - y_m / BEV_Y_RANGE) * BEV_H))
    if 0 <= px < BEV_W and 0 <= py < BEV_H:
        return (px, py)
    return None


class BirdEyeViewWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Bird's-Eye View")
        self.resize(BEV_W + 40, BEV_H + 60)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.image_label = QLabel("Waiting for radar data")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(BEV_W, BEV_H)
        layout.addWidget(self.image_label)

    def _draw_grid(self, img: np.ndarray):
        for ym in range(1, int(BEV_Y_RANGE) + 1):
            pt = bev_project(0.0, float(ym))
            if pt is None:
                continue
            cv2.line(img, (0, pt[1]), (BEV_W - 1, pt[1]), BEV_GRID, 1, cv2.LINE_AA)
            cv2.putText(img, f"{ym}m", (4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)

        for xm_int in range(-int(BEV_X_RANGE) + 1, int(BEV_X_RANGE)):
            pt = bev_project(float(xm_int), 0.0)
            if pt is None:
                continue
            cv2.line(img, (pt[0], 0), (pt[0], BEV_H - 1), BEV_GRID, 1, cv2.LINE_AA)

        origin = bev_project(0.0, 0.0)
        if origin:
            cv2.drawMarker(img, origin, (0, 200, 200), cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
            cv2.putText(img, "RADAR", (origin[0] + 10, origin[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1, cv2.LINE_AA)

    def refresh(self):
        img = np.full((BEV_H, BEV_W, 3), BEV_BG, dtype=np.uint8)
        self._draw_grid(img)

        if self.state.show_bev_camera_detections:
            for det in visible_detections(self.state):
                xyz = det.get("xyz")
                if xyz is None or len(xyz) != 3:
                    continue
                cam = np.array(xyz, dtype=np.float32)
                radar_pos = R_CAM_TO_RADAR @ cam + T_CAM_TO_RADAR
                pt = bev_project(float(radar_pos[0]), float(radar_pos[1]))
                if pt is None:
                    continue
                color = tuple(det.get("color", [0, 255, 0]))
                cv2.rectangle(img, (pt[0] - 5, pt[1] - 5), (pt[0] + 5, pt[1] + 5), color, 2, cv2.LINE_AA)
                label = det.get("label", "")
                if label:
                    cv2.putText(img, label, (pt[0] + 8, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

        for rp in self.state.radar_points:
            try:
                x = float(rp["x"])
                y = float(rp["y"])
                v = float(rp.get("v", 0.0))
            except Exception:
                continue
            pt = bev_project(x, y)
            if pt is None:
                continue
            color = velocity_to_color(v)
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)

            if self.state.show_bev_velocity_arrows and abs(v) > 0.05:
                dist_xy = math.sqrt(x * x + y * y)
                if dist_xy < 1e-6:
                    continue
                vx = v * x / dist_xy
                vy = v * y / dist_xy
                tip = bev_project(x + vx * BEV_ARROW_SCALE / BEV_X_RANGE,
                                  y + vy * BEV_ARROW_SCALE / BEV_Y_RANGE)
                if tip:
                    cv2.arrowedLine(img, pt, tip, color, 2, cv2.LINE_AA, tipLength=0.3)

        legend_y = 14
        for label, color in [("Approaching", (0, 0, 255)), ("Stationary", (255, 255, 255)), ("Moving Away", (255, 0, 0))]:
            cv2.circle(img, (BEV_W - 100, legend_y), 5, color, -1)
            cv2.putText(img, label, (BEV_W - 90, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            legend_y += 18

        cv2.putText(img, f"Pts: {len(self.state.radar_points)}", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class DashboardWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Sensor Fusion Dashboard")
        self.resize(700, 780)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        title = QLabel("Senior Design Sensor Fusion Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addWidget(QLabel("Camera + radar + brain-node view"))

        grid = QGridLayout()
        layout.addLayout(grid)

        self.system_box = self._build_system_box()
        self.detect_box = self._build_detection_box()
        self.vehicle_box = self._build_vehicle_box()
        self.overlay_box = self._build_overlay_box()
        self.filter_box = self._build_filter_box()
        self.radar_box = self._build_radar_box()

        grid.addWidget(self.system_box, 0, 0)
        grid.addWidget(self.detect_box, 0, 1)
        grid.addWidget(self.vehicle_box, 1, 0)
        grid.addWidget(self.overlay_box, 1, 1)
        grid.addWidget(self.filter_box, 2, 0)
        grid.addWidget(self.radar_box, 2, 1)

        btn_row = QHBoxLayout()
        close_btn = QPushButton("Quit")
        close_btn.clicked.connect(QApplication.instance().quit)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _build_system_box(self):
        box = QGroupBox("System Status")
        form = QFormLayout(box)
        self.lbl_cam = QLabel("Waiting")
        self.lbl_ros = QLabel("Waiting")
        self.lbl_brain = QLabel("Waiting")
        self.lbl_radar_stream = QLabel("Waiting")
        form.addRow("Camera Stream:", self.lbl_cam)
        form.addRow("ROS Bridge:", self.lbl_ros)
        form.addRow("Brain Node:", self.lbl_brain)
        form.addRow("Radar Stream:", self.lbl_radar_stream)
        return box

    def _build_detection_box(self):
        box = QGroupBox("Current Detection")
        form = QFormLayout(box)
        self.lbl_det_status = QLabel("Nothing Detected")
        self.lbl_det_obj = QLabel("--")
        self.lbl_det_dist = QLabel("--")
        self.lbl_det_conf = QLabel("--")
        self.lbl_det_vel = QLabel("--")
        self.lbl_det_status.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.lbl_det_obj.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        form.addRow("Status:", self.lbl_det_status)
        form.addRow("Object:", self.lbl_det_obj)
        form.addRow("Distance:", self.lbl_det_dist)
        form.addRow("Confidence:", self.lbl_det_conf)
        form.addRow("Camera Velocity:", self.lbl_det_vel)
        return box

    def _build_vehicle_box(self):
        box = QGroupBox("Vehicle State")
        layout = QVBoxLayout(box)
        self.lbl_vehicle_state = QLabel("SAFE")
        self.lbl_vehicle_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vehicle_state.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.lbl_vehicle_reason = QLabel("Waiting for /fusion_debug")
        self.lbl_vehicle_reason.setWordWrap(True)
        self.lbl_vehicle_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_vehicle_state)
        layout.addWidget(self.lbl_vehicle_reason)
        return box

    def _build_overlay_box(self):
        box = QGroupBox("Overlay Controls")
        layout = QVBoxLayout(box)
        self.cb_show_boxes = self._make_checkbox("Show Bounding Boxes", "show_boxes")
        self.cb_show_labels = self._make_checkbox("Show Labels", "show_labels")
        self.cb_show_distance = self._make_checkbox("Show Distance", "show_distance")
        self.cb_show_confidence = self._make_checkbox("Show Confidence", "show_confidence")
        self.cb_show_center = self._make_checkbox("Show Center Point", "show_center_point")
        self.cb_show_legend = self._make_checkbox("Show Legend", "show_legend")
        self.cb_show_none = self._make_checkbox("Show Nothing Detected", "show_nothing_detected")
        self.cb_show_cam_vel = self._make_checkbox("Show Camera Velocity", "show_camera_velocity")
        self.cb_show_cam_vel_comp = self._make_checkbox("Show Velocity Components", "show_camera_velocity_components")
        self.cb_show_radar_summary = self._make_checkbox("Show Radar Summary Box", "show_radar_summary_box")
        self.cb_show_radar_points = self._make_checkbox("Show Projected Radar Points", "show_projected_radar_points")
        self.cb_show_radar_text = self._make_checkbox("Show Radar Match Text", "show_radar_match_text")
        self.cb_show_bev_arrows = self._make_checkbox("BEV Velocity Arrows", "show_bev_velocity_arrows")
        self.cb_show_bev_cam = self._make_checkbox("BEV Camera Detections", "show_bev_camera_detections")
        for w in [
            self.cb_show_boxes, self.cb_show_labels, self.cb_show_distance, self.cb_show_confidence,
            self.cb_show_center, self.cb_show_legend, self.cb_show_none, self.cb_show_cam_vel,
            self.cb_show_cam_vel_comp, self.cb_show_radar_summary, self.cb_show_radar_points,
            self.cb_show_radar_text, self.cb_show_bev_arrows, self.cb_show_bev_cam,
        ]:
            layout.addWidget(w)
        return box

    def _build_filter_box(self):
        box = QGroupBox("Object Filters")
        layout = QVBoxLayout(box)
        self.cb_person = self._make_checkbox("Person", "show_person")
        self.cb_backpack = self._make_checkbox("Backpack", "show_backpack")
        self.cb_chair = self._make_checkbox("Chair", "show_chair")
        self.cb_laptop = self._make_checkbox("Laptop", "show_laptop")
        self.cb_phone = self._make_checkbox("Phone", "show_phone")
        for w in [self.cb_person, self.cb_backpack, self.cb_chair, self.cb_laptop, self.cb_phone]:
            layout.addWidget(w)
        return box

    def _build_radar_box(self):
        box = QGroupBox("Radar Status")
        form = QFormLayout(box)
        self.lbl_radar_status = QLabel("Waiting")
        self.lbl_radar_dist = QLabel("--")
        self.lbl_radar_motion = QLabel("--")
        self.lbl_radar_conf = QLabel("--")
        self.lbl_radar_points = QLabel("0")
        form.addRow("Detected:", self.lbl_radar_status)
        form.addRow("Distance:", self.lbl_radar_dist)
        form.addRow("Motion:", self.lbl_radar_motion)
        form.addRow("Confidence:", self.lbl_radar_conf)
        form.addRow("Raw Points:", self.lbl_radar_points)
        return box

    def _make_checkbox(self, label: str, attr: str):
        cb = QCheckBox(label)
        cb.setChecked(getattr(self.state, attr))
        cb.toggled.connect(lambda checked, a=attr: setattr(self.state, a, bool(checked)))
        return cb

    def refresh(self):
        now = time.time()
        cam_ok = (now - self.state.last_camera_t) < 2.0
        radar_ok = (now - self.state.last_radar_t) < 2.0
        brain_ok = (now - self.state.last_vehicle_t) < 2.0

        self.lbl_cam.setText("Connected" if cam_ok else "Waiting")
        self.lbl_ros.setText("Connected" if self.state.rosbridge_connected else "Disconnected")
        self.lbl_brain.setText("Active" if brain_ok else "Waiting")
        self.lbl_radar_stream.setText("Active" if radar_ok else "Waiting")

        primary = visible_primary_detection(self.state)
        if primary is None:
            self.lbl_det_status.setText("Nothing Detected")
            self.lbl_det_obj.setText("--")
            self.lbl_det_dist.setText("--")
            self.lbl_det_conf.setText("--")
            self.lbl_det_vel.setText("--")
        else:
            self.lbl_det_status.setText("Detected")
            self.lbl_det_obj.setText(primary.get("label", "--"))
            dist = primary.get("distance")
            conf = primary.get("confidence")
            vel_mag = primary.get("velocity_mag")
            self.lbl_det_dist.setText("--" if dist is None else f"{float(dist):.2f} m")
            self.lbl_det_conf.setText(f"{float(conf):.2f}")
            self.lbl_det_vel.setText("--" if vel_mag is None else f"{float(vel_mag):.2f} m/s")

        self.lbl_radar_status.setText("Detected" if self.state.radar_detected else "No Target")
        self.lbl_radar_dist.setText("--" if self.state.radar_distance is None else f"{float(self.state.radar_distance):.2f} m")
        self.lbl_radar_motion.setText(self.state.radar_motion)
        self.lbl_radar_conf.setText(f"{float(self.state.radar_confidence):.2f}")
        self.lbl_radar_points.setText(str(len(self.state.radar_points)))

        state = self.state.vehicle_state
        self.lbl_vehicle_state.setText(state)
        colors = {
            "SAFE": "background:#26a269; color:white; border-radius:14px; padding:18px;",
            "CAUTION": "background:#c99a00; color:black; border-radius:14px; padding:18px;",
            "STOP": "background:#c01c28; color:white; border-radius:14px; padding:18px;",
        }
        self.lbl_vehicle_state.setStyleSheet(colors.get(state, colors["SAFE"]))
        self.lbl_vehicle_reason.setText(self.state.fusion_debug)


class LiveViewWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Live Stream")
        self.resize(1100, 760)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.image_label = QLabel("Waiting for /camera_frame_b64")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        layout.addWidget(self.image_label)

    def refresh(self):
        frame = self.state.frame_bgr
        if frame is None:
            self.image_label.setText("Waiting for /camera_frame_b64")
            return

        img = frame.copy()
        dets = attach_radar_to_detections(visible_detections(self.state), projected_radar_points(self.state))

        if self.state.show_projected_radar_points:
            for rp in projected_radar_points(self.state):
                cv2.circle(img, (rp["u"], rp["v"]), 6, rp["color"], -1)
                cv2.putText(img, f"v={rp['vel']:.2f}", (rp["u"] + 8, rp["v"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rp["color"], 1, cv2.LINE_AA)

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            color = tuple(det.get("color", [0, 255, 0]))

            if self.state.show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if self.state.show_center_point:
                cv2.circle(img, (cx, cy), 4, color, -1)

            text_lines = []
            if self.state.show_labels:
                text_lines.append(det["label"])
            if self.state.show_confidence:
                text_lines.append(f"conf={float(det['confidence']):.2f}")
            if self.state.show_distance:
                d = det.get("distance")
                text_lines.append("d=-- m" if d is None else f"d={float(d):.2f} m")
            if self.state.show_camera_velocity:
                mag = det.get("velocity_mag")
                if mag is None:
                    text_lines.append("vel=-- m/s")
                else:
                    text_lines.append(f"vel={float(mag):.2f} m/s")
                    if self.state.show_camera_velocity_components:
                        vel = det.get("velocity")
                        if vel is not None and len(vel) == 3:
                            text_lines.append(f"Vx={vel[0]:+.2f} Vy={vel[1]:+.2f} Vz={vel[2]:+.2f}")

            if text_lines:
                for i, line in enumerate(text_lines):
                    cv2.putText(img, line, (x1, max(y1 - 12 - 18 * i, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if self.state.show_radar_match_text and det.get("best_radar") is not None:
                br = det["best_radar"]
                radar_text = f"RADAR {br['motion']} v={br['vel']:.2f} r={br['range']:.2f}"
                cv2.putText(img, radar_text, (x1, min(y2 + 22, img.shape[0] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, br["color"], 2, cv2.LINE_AA)
                if self.state.show_projected_radar_points:
                    cv2.line(img, (cx, cy), (br["u"], br["v"]), br["color"], 1, cv2.LINE_AA)

        if self.state.show_nothing_detected and not dets:
            cv2.putText(img, "Nothing Detected", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if self.state.show_legend:
            legend_items = [(name, key) for name, key in CLASS_KEYS.items() if getattr(self.state, key)]
            legend_x = img.shape[1] - 190
            legend_y = 30
            colors = {
                "Person": (0, 255, 0),
                "Backpack": (255, 0, 0),
                "Chair": (0, 165, 255),
                "Laptop": (255, 0, 255),
                "Phone": (0, 255, 255),
            }
            for name, _ in legend_items:
                color = colors[name]
                cv2.rectangle(img, (legend_x, legend_y - 12), (legend_x + 16, legend_y + 4), color, -1)
                cv2.putText(img, name, (legend_x + 22, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                legend_y += 24

        if self.state.show_radar_summary_box:
            lines = [
                f"Radar: {'DETECTED' if self.state.radar_detected else 'NONE'}",
                f"R-Dist: {'--' if self.state.radar_distance is None else f'{self.state.radar_distance:.2f} m'}",
                f"Motion: {self.state.radar_motion}",
                f"Conf: {self.state.radar_confidence:.2f}",
                f"Pts: {len(self.state.radar_points)}",
            ]
            x, y = 18, 70
            cv2.rectangle(img, (10, 45), (290, 170), (25, 25, 25), -1)
            cv2.rectangle(img, (10, 45), (290, 170), (180, 180, 180), 1)
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, f"Vehicle: {self.state.vehicle_state}", (18, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class DepthHeatMapWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Depth Heat Map")
        self.resize(1100, 760)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.image_label = QLabel("Waiting for /camera_depth_b64")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        layout.addWidget(self.image_label)

    def refresh(self):
        frame = self.state.depth_frame_bgr
        if frame is None:
            self.image_label.setText("Waiting for /camera_depth_b64")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class FusionApp:
    def __init__(self):
        self.qt_app = QApplication([])
        self.state = AppState()
        self.ros = RosListener(self.state)
        self.dashboard = DashboardWindow(self.state)
        self.live = LiveViewWindow(self.state)
        self.bev = BirdEyeViewWindow(self.state)
        self.depth = DepthHeatMapWindow(self.state)
        self.dashboard.show()
        self.live.show()
        self.bev.show()
        self.depth.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(80)
        self.qt_app.aboutToQuit.connect(self.ros.close)

    def refresh(self):
        self.dashboard.refresh()
        self.live.refresh()
        self.bev.refresh()
        self.depth.refresh()

    def run(self):
        return self.qt_app.exec()


if __name__ == "__main__":
    app = FusionApp()
    raise SystemExit(app.run())
