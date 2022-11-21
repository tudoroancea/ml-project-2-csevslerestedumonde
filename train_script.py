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

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import save_image

from PIL import Image

# select the line depending on the platform
device = "cpu"
# device = "cuda"

torch.manual_seed(127)
print(torch.cuda.is_available())

read_image = torchvision.transforms.ToTensor()

class RoadsDataset(tdata.Dataset):
    root: str
    num_images: int
    images: list
    gt_images: list
    gt_images_one_hot: list

    def __init__(self, root: str, num_images=20, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.num_images = num_images
        assert 10 <= num_images <= 100
        self.images = []
        self.gt_images = []
        self.gt_images_one_hot = []
        for i in range(num_images):
            image_path = os.path.join(self.root, "image/image_%.5d.png" % (i + 1))
            img = read_image(Image.open(image_path)).type(torch.float32).to(device)
            img /= 255.0
            self.images.append(img)
            gt_image_path = os.path.join(
                self.root, "groundtruth/ground_truth_%.5d.png" % (i + 1)
            )
            gt_image = read_image(Image.open(gt_image_path))
            # print(gt_image)
            gt_image_one_hot = torch.Tensor(np.moveaxis(F.one_hot(
                    torch.div(
                        torch.squeeze(gt_image),
                        255,
                    ).type(torch.int64),
                    2,
                ).numpy(), 2, 0)).to(dtype=torch.float32).to(device)
            gt_image_one_hot = torch.cat((gt_image/255, -gt_image/255 + 1))
            # print(gt_image_one_hot.shape)
            self.gt_images.append(gt_image)
            self.gt_images_one_hot.append(gt_image_one_hot)

        print("Loaded {} images from {}".format(num_images, root))

    def __len__(self):
        return self.num_images

    def __getitem__(self, item: int) -> tuple:
        if self.transform:
            image = self.transform(self.images[item])
        else:
            image = self.images[item]

        if self.target_transform:
            gt_image_one_hot = self.target_transform(self.gt_images_one_hot[item])
        else:
            gt_image_one_hot = self.gt_images_one_hot[item]

        return image, gt_image_one_hot

training_data = RoadsDataset(root="data/train", num_images=100)
training_dataloader = tdata.DataLoader(training_data, batch_size=5, shuffle=True)


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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


unet_model = UNet(n_channels=3, n_classes=2).to(device)

def loss_fun(input, target):
    # plt.subplot(121)
    # plt.imshow(input[0]*255, cmap="gray")
    # plt.subplot(122)
    # plt.imshow(target[0]*255, cmap="gray")
    # plt.show()
    return nn.BCELoss()(input, target)

optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)


def train(dataloader: tdata.DataLoader, model: nn.Module, loss_fun, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_num, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        # Compute prediction error
        pred = model(X_batch)
        loss = loss_fun(pred, Y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch_num + 1) * len(X_batch)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Training ================================================
# epochs = 20
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(training_dataloader, unet_model, loss_fun, optimizer)
# print("Done training!")
# torch.save(unet_model.state_dict(), "unet_model2.pth")
# print("Saved PyTorch Model State to unet_model2.pth")
#==========================================================

# Classifying ================================================
unet_model = UNet(3, 2).to(device)
unet_model.load_state_dict(torch.load("unet_model3.pth", map_location=torch.device('cpu')))
unet_model.eval()
index = 2
X, Y, Y_one_hot = training_data.images[index], training_data.gt_images[index], training_data.gt_images_one_hot[index]
X = X.to(device)
Y = Y.to(device)
Y_one_hot = Y_one_hot.to(device)
with torch.no_grad():
    Y_pred = torch.squeeze(unet_model(torch.unsqueeze(X, 0)))
    print(loss_fun(Y_pred, Y_one_hot))
    Y_pred = Y_pred[1,:,:]
    Y_pred = torch.unsqueeze(Y_pred, 0)
    Y_pred *= 255

plt.imsave("Input_{}.png".format(index), np.moveaxis((X*255).to("cpu").numpy(), 0, 2))

print(Y.shape)
print(torch.squeeze(Y).shape)
plt.imsave("GT_{}.png".format(index), torch.squeeze(Y).to("cpu").numpy(), cmap="gray")

plt.imsave("Pred_{}.png".format(index), torch.squeeze(Y_pred).to("cpu").numpy() , cmap="gray")
#==========================================================

