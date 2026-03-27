import numpy as np

from config.utils import updateDatasetAndModelConfig
from config.default import _Cfg

from transform_pc import extrinsic, radar_to_model_input

import time

config_file = "centerfusion_debug.yaml"

# fake dataset, lol
class DatasetStub:
    num_categories = 7          # nuScenes has 7 detection classes
    default_resolution = (448, 800)

# load in the config files
cfg = _Cfg
cfg.merge_from_file(config_file)

updateDatasetAndModelConfig(cfg, DatasetStub())



# example matrices
N_points = 100

img_shape = (448, 800)

# these values are garbage, get the correct ones >:()
focal_length_mm = 4.0            # from camera datasheet
sensor_width_mm = 6.4            # from camera datasheet
image_width_px  = img_shape[1]   # your image width
image_height_px  = img_shape[0]  # your image height
fx = focal_length_mm * (image_width_px / sensor_width_mm)
fy = fx                      # usually equal for square pixels
cx = image_width_px  / 2     # = 400
cy = image_height_px / 2     # = 224
dummy_image = np.ones((1, 3, image_height_px, image_width_px))

radar_points = np.ones((3, N_points)) 
v_radial = np.ones((N_points))
radar_to_cam_extrinsic = extrinsic
camera_intrinsic = np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0]
    ], dtype=np.float32)

start = time.monotonic()

example_output = radar_to_model_input(
    radar_points, 
    v_radial, 
    radar_to_cam_extrinsic, 
    camera_intrinsic, 
    dummy_image,   # pass image not img_shape
    cfg
)

print(f"Depthmap: {example_output[0].shape}\nHeatmap: {example_output[1].shape}")

print(f"Time to run: {time.monotonic() - start}")

# for the future, this is how we could get the camera intrinsic matrix
# import cv2
# 
# # After collecting checkerboard images
# ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints,
#     (image_width, image_height),
#     None, None
# )
# # K is your 3x3 intrinsic matrix
# # Add the 4th column of zeros to make it 3x4
# K_34 = np.hstack([K, np.zeros((3, 1))])