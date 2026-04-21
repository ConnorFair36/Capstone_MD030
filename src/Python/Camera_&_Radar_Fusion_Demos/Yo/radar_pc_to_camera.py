import numpy as np
import cv2


def radar_to_cam(radar_points: np.array, radar_to_cam_extrinsic: np.array):
    """
    Transforms the origin of the point clouds to the origin of the camera
    Args:
        radar_points:           Takes in the radar points with shape: (4, n_points)
        radar_to_cam_extrinsic: encodes the difference between the camera and radar in meters.
    
    Returns:
        transformed pointcloud (6, n_points) 
    """
    # transform radial velocity into x, y and z components to preserve direction before changing the origin
    N = radar_points.shape[1]
    pc_positions = radar_points[:3]
    vel_mag =  radar_points[3]
    norm = np.linalg.norm(pc_positions, axis=0, keepdims=True)
    vel_direction = pc_positions / norm
    vel_components = vel_direction * vel_mag

    # Transform to camera frame
    ones = np.ones((1, N))
    radar_hom = np.vstack([pc_positions, ones])
    cam_points = (radar_to_cam_extrinsic @ radar_hom)[:3]
    return np.vstack([cam_points, vel_components])

def build_extrinsic_matrix(tx, ty, tz, roll, pitch, yaw):
    """
    Args:
        tx, ty, tz:       translation in meters (radar origin → camera origin)
        roll, pitch, yaw: rotation in degrees
    
    Returns:
        (4, 4) extrinsic matrix
    """
    
    T = np.array([tx, ty, tz])
    
    extrinsic = np.eye(4)
    extrinsic[:3, 3]  = T
    
    return extrinsic


def build_intrinsic_matrix(fx, fy, cx, cy):
    """
    Standard pinhole camera intrinsic matrix.
    Args:
        fx, fy: focal lengths in pixels
        cx, cy: principal point (typically image_width/2, image_height/2)
    Returns:
        (3, 3) K matrix
    """
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def project_to_image(cam_points: np.ndarray, K: np.ndarray):
    """
    Projects 3D camera-frame points onto the 2D image plane.
    Args:
        cam_points: (3, N) array of points in camera frame [X, Y, Z]
        K:          (3, 3) intrinsic matrix
    Returns:
        pixels:     (2, N) array of [u, v] pixel coordinates
        valid_mask: (N,)   boolean mask — True where point is in front of camera (Z > 0)
    """
    # Only keep points in front of the camera
    valid_mask = cam_points[2, :] > 0

    # K @ cam_points → (3, N) homogeneous image coords [u*Z, v*Z, Z]
    proj = K @ cam_points          # shape (3, N)

    # Perspective divide by Z to get pixel coords
    pixels = proj[:2, :] / proj[2, :]   # shape (2, N)

    return pixels, valid_mask


def overlay_radar_on_image(image: np.ndarray, pixels: np.ndarray,
                            valid_mask: np.ndarray, depths: np.ndarray = None):
    """
    Draws radar points onto the image.
    Args:
        image:      HxWx3 BGR image (numpy array, e.g. from cv2.imread)
        pixels:     (2, N) projected pixel coords [u, v]
        valid_mask: (N,) boolean — which points are in front of camera
        depths:     (N,) optional Z values for depth-based coloring
    Returns:
        Annotated image copy
    """
    h, w = image.shape[:2]
    out = image.copy()

    pts = pixels[:, valid_mask]          # (2, M) — only valid points
    Z   = depths[valid_mask] if depths is not None else None

    # Normalize depth for coloring (closer = red, farther = blue)
    if Z is not None:
        z_min, z_max = Z.min(), Z.max()
        z_norm = (Z - z_min) / (z_max - z_min + 1e-6)   # 0..1

    for i in range(pts.shape[1]):
        u, v = int(round(pts[0, i])), int(round(pts[1, i]))

        # Skip points outside the image bounds
        if not (0 <= u < w and 0 <= v < h):
            continue

        # Color by depth if available, otherwise white
        if Z is not None:
            # BGR: blue=far, red=close
            color = (
                int(255 * (1 - z_norm[i])),   # B
                50,                            # G
                int(255 * z_norm[i])           # R
            )
        else:
            color = (255, 255, 255)

        cv2.circle(out, (u, v), radius=4, color=color, thickness=-1)

    return out

# Example — if your radar is 10cm to the left, 5cm below, 
# 2cm behind the camera, with no angular difference:
extrinsic = build_extrinsic_matrix(
    tx= -0.0333,   # 10cm left
    ty= -0.0349,   # 5cm below
    tz= -0.0190,   # 2cm behind
    roll=0, pitch=0, yaw=0
)

if __name__ == "__main__":
    pc_test = np.array([[10, 12, 13, 55]]).T
    pc_test2 = radar_to_cam(pc_test, extrinsic)
    print(pc_test2)
