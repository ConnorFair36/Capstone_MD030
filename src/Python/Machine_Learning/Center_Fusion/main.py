import torch
from torchinfo import summary
import numpy as np

from yacs.config import CfgNode as CN
from model.model import getModel, loadModel
from config.utils import updateDatasetAndModelConfig
from config.default import _Cfg

from model.decode import fusionDecode

from transform_pc import radar_to_model_input, extrinsic

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

# get the model
cf_model = getModel(cfg)

# load in the model weights
cf_model = loadModel(cf_model, cfg)[1]

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cf_model = cf_model.to(device)

B = 1
H, W = 448, 800
outH, outW = 112, 200  # INPUT_SIZE // 4
pc_channels = 3
n_points = 100

# these values are garbage, get the correct ones >:()
focal_length_mm = 4.0            # from camera datasheet
sensor_width_mm = 6.4            # from camera datasheet
fx = focal_length_mm * (W / sensor_width_mm)
fy = fx                      # usually equal for square pixels
cx = W  / 2     
cy = H / 2     
camera_intrinsic = np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0]
    ], dtype=np.float32)

dummy_image = torch.zeros(B, 3, H, W).to(device)           # RGB image
dummy_pc = np.zeros((4, n_points))                         # radar pointclouds
dummy_pc_dep, dummy_pc_hm = radar_to_model_input(
    dummy_pc[:3, :],    # (x, y, z)
    dummy_pc[3, :],     # radial velocity
    extrinsic, 
    camera_intrinsic,
    dummy_image[0],     
    cfg
)
dummy_pc_hm = torch.tensor(dummy_pc_hm).reshape(1, 3, outH, outW).to(device)
dummy_pc_dep = torch.tensor(dummy_pc_dep).reshape(1, 3, outH, outW).to(device)
# dummy_pc_hm = torch.zeros(B, pc_channels, outH, outW).to(device)     # radar heatmap at output resolution
# dummy_pc_dep = torch.zeros(B, pc_channels, outH, outW).to(device)    # radar depth at output resolution
# dummy_calib = torch.zeros(B, 3, 4).to(device)              # 3x4 camera intrinsic matrix
dummy_calib = torch.tensor(camera_intrinsic).reshape(1, 3, 4).to(device)

cf_model.eval()
with torch.no_grad():
    for i in range(5):
        start = time.monotonic()
        output = cf_model(
            dummy_image,
            pc_hm=dummy_pc_hm,
            pc_dep=dummy_pc_dep,
            calib=dummy_calib
        )
        boxes = fusionDecode(output)
        print(f"Runtime: {time.monotonic() - start}")
        if i == 4:
            for key, value in boxes.items():
                print(f"{key}: {value.shape}")
            print("----------------")
            for key, value in output[0].items():
                print(f"{key}: {value.shape}")
#for name, param in cf_model.named_parameters():
#    print(name, param)