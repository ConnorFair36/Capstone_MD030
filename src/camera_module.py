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
DEPTH_HEATMAP_MAX_M = 6.0
DEPTH_HEATMAP_MIN_M = 0.1

DETECT_CLASSES = {
    0:  ("Person", (0, 255, 0)),
    24: ("Backpack", (255, 0, 0)),
    56: ("Chair", (0, 165, 255)),
    63: ("Laptop", (255, 0, 255)),
    67: ("Phone", (0, 255, 255)),
}


CALIBRATION_PATHS = ["./calibration.json", "./calibration(3).json"]
CALIBRATION = None
for _cal_path in CALIBRATION_PATHS:
    try:
        with open(_cal_path, "r") as f:
            CALIBRATION = json.load(f)
        break
    except FileNotFoundError:
        continue

if CALIBRATION is None:
    raise FileNotFoundError("Could not find calibration.json or calibration(3).json")

# camera intrinsics used for 3D back-projection at detection center
COLOR_INTRINSICS = CALIBRATION["color_intrinsics"]

DEPTH_INTRINSICS = CALIBRATION["depth_intrinsics"]
depth_to_color_extrinsics_4x4 = np.array(CALIBRATION["depth_to_color_extrinsics_4x4"])

_previous_tracks: dict[int, dict[str, Any]] = {}
_previous_frame_t: float | None = None


def initialize(model_path: str = "yolo26n.pt") -> None:
    global pipeline, align, model, _previous_tracks, _previous_frame_t

    model = YOLO(model_path)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
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


def _safe_depth(depth_frame: np.array, cx: int, cy: int, window: int = 2) -> float | None:

    # create a window x window frame around our target pixel  
    frame = depth_frame[max(cy-window, 0):cy+window+1, max(cx-window, 0):cx+window+1]
    if not bool(frame.size):
        return None
    frame = frame.flatten()
    frame = frame[(frame >= 0.1) & (frame <= 20.0)]
    
    if not frame.size:
        return None
    return float(np.median(frame))

def depth_to_3d(depth_map: np.array, depth_intrinsics: dict):
    """Converts the depth measurements from the depth camera into 3d points."""
    h, w = depth_map.shape

    #assert (h == depth_intrinsics['height']) and \
    #       (w == depth_intrinsics['width'])
    
    cx, cy = depth_intrinsics['ppx'], depth_intrinsics['ppy']
    fx, fy = depth_intrinsics['fx'], depth_intrinsics['fy']

    x_map, y_map = np.meshgrid(range(w), range(h))

    x_map = (x_map - cx) * depth_map / fx
    y_map = (y_map - cy) * depth_map / fy
    

    xs, ys, zs = x_map.flatten(), y_map.flatten(), depth_map.flatten()
    
    msk = zs > 0

    xs, ys, zs = xs[msk], ys[msk], zs[msk]

    pts = np.stack([xs,ys,zs], axis=0)  

    return pts

def pts3d_to_img(pts, intrinsic_dict):
 
    cx, cy = intrinsic_dict['ppx'], intrinsic_dict['ppy']
    fx, fy = intrinsic_dict['fx'], intrinsic_dict['fy']
    h, w = intrinsic_dict['height'], intrinsic_dict['width']
    xs, ys, zs = pts[0,:], pts[1,:], pts[2,:]

    x_pixels = fx * xs / zs + cx
    y_pixels = fy * ys / zs + cy

    msk_in = (x_pixels>=0) & (x_pixels<=w-1) & (y_pixels>=0) & (y_pixels<=h-1)

    x_pixels, y_pixels, d_pixels = x_pixels[msk_in], y_pixels[msk_in], zs[msk_in]

    return x_pixels, y_pixels, d_pixels

def transform_pts(trans_matrix, p0):
    """Transforms the depth point origin into the camera's origin."""
    p0 = p0.astype(float)
    p1 = np.dot(trans_matrix, np.vstack((p0, np.ones(p0.shape[1],dtype=float))))
    p1 = p1[:3,:]
    
    return p1

def _img_to_pts3d(x_pixel: int, y_pixel: int, depth_m: float) -> list[float]:
    cx, cy = COLOR_INTRINSICS["ppx"], COLOR_INTRINSICS["ppy"]
    fx, fy = COLOR_INTRINSICS["fx"], COLOR_INTRINSICS["fy"]

    x_pts = (x_pixel - cx) * depth_m / fx
    y_pts = (y_pixel - cy) * depth_m / fy
    z_pts = depth_m
    return [float(x_pts), float(y_pts), float(z_pts)]

def _transform_depth_to_img(raw_depth: np.array) -> np.array:
    raw_depth *= CALIBRATION["depth_scale_m_per_unit"]
    depth_frame = depth_to_3d(raw_depth, DEPTH_INTRINSICS)
    depth_frame = transform_pts(depth_to_color_extrinsics_4x4, depth_frame)
    x_px, y_px, d_px = pts3d_to_img(depth_frame, COLOR_INTRINSICS)
    h_c = COLOR_INTRINSICS["height"]
    w_c = COLOR_INTRINSICS["width"]
    depth_color_img = np.zeros((h_c, w_c), dtype=np.float32)
    xi = np.clip(np.round(x_px).astype(int), 0, w_c - 1)
    yi = np.clip(np.round(y_px).astype(int), 0, h_c - 1)
    depth_color_img[yi, xi] = d_px
    return depth_color_img


def depth_to_heatmap_bgr(depth_img_m: np.ndarray, min_m: float = DEPTH_HEATMAP_MIN_M, max_m: float = DEPTH_HEATMAP_MAX_M) -> np.ndarray:
    """Convert a depth image in metres to a colored heat map for display."""
    depth = depth_img_m.copy()
    valid = (depth >= min_m) & (depth <= max_m)

    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth[valid], min_m, max_m)
        scaled = 255.0 * (1.0 - ((clipped - min_m) / max(max_m - min_m, 1e-6)))
        normalized[valid] = scaled.astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap[~valid] = (0, 0, 0)

    if heatmap.shape[1] > 0 and heatmap.shape[0] > 0:
        legend_text = f"Depth Heat Map ({min_m:.1f}m - {max_m:.1f}m)"
        cv2.putText(heatmap, legend_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(heatmap, "Near", (16, heatmap.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(heatmap, "Far", (heatmap.shape[1] - 60, heatmap.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)

    return heatmap

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
    # aligned_frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    # transforms the origin for the depth map into the camera based on our measured intrinsics
    raw_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_frame = _transform_depth_to_img(raw_depth)

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
    depth_heatmap_bgr = depth_to_heatmap_bgr(depth_frame)

    return {
        "frame": color_img,
        "depth_frame": depth_frame,
        "depth_heatmap_bgr": depth_heatmap_bgr,
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
