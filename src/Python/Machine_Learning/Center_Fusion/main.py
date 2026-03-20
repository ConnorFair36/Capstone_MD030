import torch
from torchinfo import summary

from yacs.config import CfgNode as CN
from model.model import getModel, loadModel
from config.utils import updateDatasetAndModelConfig
from config.default import _Cfg

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


print("Done!")
# TODO Figure out what tensor input sizes the model takes
dummy = torch.zeros(1, 3, 448, 800)
cf_model.eval()
with torch.no_grad():
    print((cf_model(dummy, pc_hm=dummy, pc_dep=dummy, calib=dummy)).shape)

#for name, param in cf_model.named_parameters():
#    print(name, param)