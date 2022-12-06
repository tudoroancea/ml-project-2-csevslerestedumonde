import copy
from load_data import *
import torch
import torchvision
from PIL import Image

data = TrainRoadsDataset(
    path="data/training", image_idx=list(range(1, 101)), device="cpu"
)
s = torch.sum(data.gt_images).item()
print("road pixels proportion: ", s / (400 * 400 * 100))
print("pos_weight: ", (400 * 400 * 100 - s) / s)
