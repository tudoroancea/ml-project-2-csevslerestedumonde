"""
    Training : 
        - décommenter la section training et commenter la section classifying
        - upload le script et le dossier data sur izar 
        - module load gcc cuda py-torch py-torchvision
        - srun --gres=gpu:1 python train_script.py

    Classifying :
        - décommenter la section classifying et commenter la section training
        - Sur le serveur :
            - rien à faire pour faire tourner 
            - après récupèrer les .png avec scp
        - Sur le client : 
            - scp les paramètres du modèle "unet_model<x>.pth" 
            (scp <nom>@izar.epfl.ch:<path>/unet_model3.pth .)
            - puis exécuter : juste besoin de torch et torchvision pas cuda (device = "cpu")

    Pour modifier gardez en tête qu'on utilise pytorch 1.6 donc certaines fonctions sont
    pas disponibles, des fois faut utiliser les autres librairies genre numpy.

"""


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision
from PIL import Image
from torch.nn import functional as F

from load_data import RoadsDataset
from model import UNet


device = "cuda"
torch.manual_seed(127)
print(torch.cuda.is_available())

training_data = RoadsDataset(
    root="data_augmented/training", num_images=800, device=device
)

# Showing/saving images ===================================
# index = torch.randint(0, len(training_data), (1,)).item()
# image = training_data.images[index].to(device="cpu")
# gt_image = training_data.gt_images[index].to(device="cpu")
# plt.subplot(121)
# save_image(image*255, "1.png")
# plt.imsave("plt1.png", np.moveaxis(image.numpy()*255, 0, 2))
# plt.subplot(122)
# plt.imshow(torch.Tensor(np.moveaxis(gt_image.numpy(),0,2)), cmap="gray")
# torchvision.utils.save_image(gt_image, "2.png")
# plt.tight_layout()
# =========================================================


def train(model: nn.Module, loss_fun, batch_size, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = tdata.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        size = len(dataloader.dataset)
        model.train()
        for batch_num, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # Compute prediction and loss
            pred = model(X_batch)
            loss = loss_fun(pred, Y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            loss, current = loss.item(), (batch_num + 1) * len(X_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # compute metrics

        print()


# Training ================================================
unet_model = UNet(n_channels=3, n_classes=2).to(device)
loss_fun = nn.MSELoss()
# train(unet_model, loss_fun, batch_size=20, lr=0.001, epochs=40)
# print("Done training!")
# model_file_name = "unet_model.pth"
# torch.save(unet_model.state_dict(), model_file_name)
# print("Saved PyTorch Model State to " + model_file_name)
# ==========================================================

# Classifying ================================================
unet_model = UNet(3, 2).to(device)
unet_model.load_state_dict(
    torch.load("unet_model.pth", map_location=torch.device("cpu"))
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
# ==========================================================
