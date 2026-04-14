import base64
import json
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

pipeline = None
align = None
model = None

CONF_THRESHOLD = 0.50
JPEG_QUALITY = 80

DETECT_CLASSES = {
    0:  ("Person", (0, 255, 0)),
    24: ("Backpack", (255, 0, 0)),
    56: ("Chair", (0, 165, 255)),
    63: ("Laptop", (255, 0, 255)),
    67: ("Phone", (0, 255, 255)),
}

# camera intrinsics used for 3D back-projection at detection center
# these match the user's calibration.json color intrinsics
COLOR_INTRINSICS = {
    "fx": 641.60986328125,
    "fy": 640.8302001953125,
    "ppx": 650.4204711914062,
    "ppy": 405.4923400878906,
}

_previous_tracks: dict[int, dict[str, Any]] = {}
_previous_frame_t: float | None = None


def initialize(model_path: str = "yolo26n.pt") -> None:
    global pipeline, align, model, _previous_tracks, _previous_frame_t

    model = YOLO(model_path)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    _previous_tracks = {}
    _previous_frame_t = None


def shutdown() -> None:
    global pipeline, align, model, _previous_tracks, _previous_frame_t

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass

    pipeline = None
    align = None
    model = None
    _previous_tracks = {}
    _previous_frame_t = None


def _safe_depth(depth_frame: rs.depth_frame, cx: int, cy: int, window: int = 2) -> float | None:
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    samples: list[float] = []

    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            px = max(0, min(cx + dx, w - 1))
            py = max(0, min(cy + dy, h - 1))
            d = depth_frame.get_distance(px, py)
            if 0.1 <= d <= 20.0:
                samples.append(float(d))

    if not samples:
        return None
    return float(np.median(samples))


def _img_to_pts3d(x_pixel: int, y_pixel: int, depth_m: float) -> list[float]:
    cx, cy = COLOR_INTRINSICS["ppx"], COLOR_INTRINSICS["ppy"]
    fx, fy = COLOR_INTRINSICS["fx"], COLOR_INTRINSICS["fy"]

    x_pts = (x_pixel - cx) * depth_m / fx
    y_pts = (y_pixel - cy) * depth_m / fy
    z_pts = depth_m
    return [float(x_pts), float(y_pts), float(z_pts)]


def _compute_velocity(track_id: int | None, xyz: list[float] | None, now_t: float) -> tuple[list[float] | None, float | None]:
    global _previous_tracks

    if track_id is None or xyz is None:
        return None, None

    prev = _previous_tracks.get(track_id)
    if prev is None:
        _previous_tracks[track_id] = {"xyz": xyz, "t": now_t}
        return None, None

    dt = max(now_t - float(prev["t"]), 1e-6)
    prev_xyz = np.array(prev["xyz"], dtype=np.float32)
    cur_xyz = np.array(xyz, dtype=np.float32)
    vel = (cur_xyz - prev_xyz) / dt
    speed = float(np.linalg.norm(vel))

    _previous_tracks[track_id] = {"xyz": xyz, "t": now_t}
    return vel.tolist(), speed


def _cleanup_old_tracks(now_t: float, max_age_s: float = 1.0) -> None:
    global _previous_tracks
    _previous_tracks = {
        tid: info for tid, info in _previous_tracks.items()
        if (now_t - float(info.get("t", now_t))) <= max_age_s
    }


def _primary_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None

    best = None
    best_metric = None
    for det in detections:
        dist = det.get("distance")
        conf = float(det.get("confidence", 0.0))
        metric = (0, float(dist), -conf) if dist is not None else (1, -conf)
        if best is None or metric < best_metric:
            best = det
            best_metric = metric
    return best


def get_camera_data() -> dict[str, Any] | None:
    global pipeline, align, model, _previous_frame_t

    if pipeline is None or align is None or model is None:
        raise RuntimeError("Camera module not initialized. Call initialize() first.")

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    now_t = time.time()
    _cleanup_old_tracks(now_t)

    results = model.track(color_img, persist=True, verbose=False)
    detections: list[dict[str, Any]] = []

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in DETECT_CLASSES or conf < CONF_THRESHOLD:
                continue

            label, color = DETECT_CLASSES[cls_id]
            track_id = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id[0]) if hasattr(box.id, "__len__") else int(box.id)
                except Exception:
                    track_id = None

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(color_img.shape[1] - 1, x2)
            y2 = min(color_img.shape[0] - 1, y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dist = _safe_depth(depth_frame, cx, cy, window=2)
            xyz = _img_to_pts3d(cx, cy, dist) if dist is not None else None
            velocity_vec, velocity_mag = _compute_velocity(track_id, xyz, now_t)

            detections.append({
                "object_id": track_id,
                "class_id": cls_id,
                "label": label,
                "confidence": float(conf),
                "distance": float(dist) if dist is not None else None,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(cx), int(cy)],
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "xyz": xyz,
                "velocity": velocity_vec,
                "velocity_mag": float(velocity_mag) if velocity_mag is not None else None,
            })

    primary = _primary_detection(detections)
    return {
        "frame": color_img,
        "detected": bool(detections),
        "label": primary["label"] if primary else None,
        "distance": primary["distance"] if primary else None,
        "confidence": float(primary["confidence"]) if primary else 0.0,
        "detections": detections,
    }


def encode_frame_to_b64(frame: np.ndarray, jpeg_quality: int = JPEG_QUALITY) -> str | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


if __name__ == "__main__":
    initialize("yolo26n.pt")
    try:
        while True:
            data = get_camera_data()
            if data is None:
                continue
            print(json.dumps({
                "detected": data["detected"],
                "label": data["label"],
                "distance": data["distance"],
                "confidence": data["confidence"],
                "n_detections": len(data["detections"]),
            }))
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
