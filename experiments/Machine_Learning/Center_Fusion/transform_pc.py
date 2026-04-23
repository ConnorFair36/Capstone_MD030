import numpy as np
from scipy.spatial.transform import Rotation

from utils.pointcloud import map_pointcloud_to_image
from utils.image import getAffineTransform
from utils.transform_pointcloud import PointCloudProcessor

# Applied to your full point cloud (vectorized)
def decompose_radial_velocity_pc(radar_pc, v_radial):
    """
    Args:
        radar_pc: (3, N) array of [x, y, z] in camera coords
        v_radial: (N,) array of radial velocities
    
    Returns:
        radar_pc_with_vel: (5, N) array [x, y, z, vx_comp, vz_comp]
    """
    x, y, z = radar_pc[0], radar_pc[1], radar_pc[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r == 0, 1e-6, r)  # avoid division by zero
    
    vx = v_radial * (x / r)
    vz = v_radial * (z / r)
    
    return np.vstack([radar_pc, vx, vz])  # (5, N)

def generate_pc_hm(pc_dep, config):
    """
    Generate radar point cloud heatmap from depth map.
    Closer objects have higher values.

    Args:
        pc_dep: (3, 112, 200) depth map [depth, vx, vz]
        config: config object

    Returns:
        pc_hm: (3, 112, 200) normalized heatmap
    """
    maxDist = int(config.DATASET.MAX_PC_DIST)
    pc_hm = pc_dep.copy()

    if config.DATASET.ONE_HOT_PC:
        pc_hm[:int(maxDist)] /= maxDist
        pc_hm[:int(maxDist)] = 1 - pc_hm[:int(maxDist)]
    else:
        pc_hm[0] /= maxDist
        pc_hm[0] = 1 - pc_hm[0]

    return pc_hm

## Updated Full Pipeline

# Your Radar PC          Decompose         Coordinate        Project to        Build
# (x, y, z, v_rad)  →   velocity     →    Transform    →    Image 2D     →   Heatmap
#                        (step 1)         (step 2)          (step 3)         (step 4)

def radar_to_model_input(radar_points, v_radial, radar_to_cam_extrinsic, camera_intrinsic, img, config):
    # Step 1: Transform to camera frame
    N = radar_points.shape[1]
    ones = np.ones((1, N))
    radar_hom = np.vstack([radar_points, ones])
    cam_points = (radar_to_cam_extrinsic @ radar_hom)[:3]

    # Step 2: Decompose radial velocity
    cam_points_with_vel = decompose_radial_velocity_pc(cam_points, v_radial)

    # Step 3: Project to image plane
    img_shape = img.shape[-2:]  # (height, width)
    pc_2d, mask = map_pointcloud_to_image(
        cam_points_with_vel,
        camera_intrinsic,
        img_shape=img_shape
    )
    pc_3d = cam_points_with_vel[:, mask]

    # Step 4: Build affine transforms
    input_h, input_w   = config.MODEL.INPUT_SIZE
    output_h, output_w = config.MODEL.OUTPUT_SIZE
    center = np.array([img_shape[1] / 2, img_shape[0] / 2])  # (cx, cy)
    scale  = max(img_shape)

    transMatInput  = getAffineTransform(center, scale, 0, [input_w, input_h])
    transMatOutput = getAffineTransform(center, scale, 0, [output_w, output_h])

    # Step 5: Build img_info and splat into depthmap
    img_info = {
        "calib": camera_intrinsic,
        "width": img_shape[1],
        "height": img_shape[0]
    }
    pc_processor = PointCloudProcessor(config)
    _, _, pc_dep = pc_processor.processPointCloud(
        pc_2d, pc_3d, img, transMatInput, transMatOutput, img_info
    )

    # Step 6: create the heatmap using the depthmap
    pc_hm = generate_pc_hm(pc_dep, config)

    return pc_dep, pc_hm  # (3, 112, 200), (3, 112, 200)

def build_extrinsic_matrix(tx, ty, tz, roll, pitch, yaw):
    """
    Args:
        tx, ty, tz:       translation in meters (radar origin → camera origin)
        roll, pitch, yaw: rotation in degrees
    
    Returns:
        (4, 4) extrinsic matrix
    """
    R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T = np.array([tx, ty, tz])
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3]  = T
    
    return extrinsic

# Example — if your radar is 10cm to the left, 5cm below, 
# 2cm behind the camera, with no angular difference:
extrinsic = build_extrinsic_matrix(
    tx=-0.10,   # 10cm left
    ty= 0.05,   # 5cm below
    tz=-0.02,   # 2cm behind
    roll=0, pitch=0, yaw=0
)