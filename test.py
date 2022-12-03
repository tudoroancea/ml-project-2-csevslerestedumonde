import torch
from model import UNet
from utils import *
from load_data import RoadsDataset
from matplotlib import pyplot as plt
import numpy as np

device = "cuda"

training_data = RoadsDataset(
    root="data_augmented/training", image_idx = list(range(1,11)), device=device
)

loss_fun = torch.nn.BCELoss()

# Classifying ================================================
unet_model = UNet(3, 2).to(device)
unet_model.load_state_dict(
    torch.load("unet_model_iou.pth", map_location=torch.device("cpu"))
)
unet_model.eval()
index = 2
X, Y, Y_one_hot = (
    training_data.images[index],
    training_data.gt_images[index],
    training_data.gt_images_one_hot[index],
)
X = X.to(device)
Y = Y.to(device)
Y_one_hot = Y_one_hot.to(device)
with torch.no_grad():
    Y_pred = torch.squeeze(unet_model(torch.unsqueeze(X, 0)))
    print(loss_fun(Y_pred, Y_one_hot))
    Y_pred = Y_pred[1, :, :]
    Y_pred = torch.unsqueeze(Y_pred, 0)
    Y_pred *= 255

plt.imsave("Input_{}.png".format(index), np.moveaxis((X * 255).to("cpu").numpy(), 0, 2))

print(Y.shape)
print(torch.squeeze(Y).shape)
plt.imsave("GT_{}.png".format(index), torch.squeeze(Y).to("cpu").numpy(), cmap="gray")

plt.imsave(
    "Pred_{}.png".format(index), torch.squeeze(Y_pred).to("cpu").numpy(), cmap="gray"
)
