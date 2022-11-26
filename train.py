import torch
import torch.nn as nn
import torch.utils.data as tdata
import matplotlib.pyplot as plt

from load_data import RoadsDataset
from model import UNet
from utils import *


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
    size = len(dataloader.dataset)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
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

        # compute metrics on the whole dataset
        model.eval()
        with torch.no_grad():
            pred = model(training_data.images)
            print(
                "Dice coeff: {}, Jaccard index: {}".format(
                    dice_coeff(pred, training_data.gt_images_one_hot),
                    jaccard_index(pred, training_data.gt_images_one_hot),
                )
            )


# Training ================================================
unet_model = UNet(n_channels=3, n_classes=2).to(device)
loss_fun = dice_loss
train(unet_model, loss_fun, batch_size=40, lr=1e-4, epochs=40)
print("Done training!")
model_file_name = "unet_model.pth"
torch.save(unet_model.state_dict(), model_file_name)
print("Saved PyTorch Model State to " + model_file_name)
# ==========================================================
