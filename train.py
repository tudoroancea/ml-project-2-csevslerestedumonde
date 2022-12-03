import argparse
import os
from tkinter.tix import Tree
import torch
import torch.nn as nn
import torch.utils.data as tdata

from load_data import RoadsDataset
from model import UNet
from utils import *


def train(
    model: nn.Module,
    loss_fun,
    train_data_: RoadsDataset,
    validation_data_: RoadsDataset,
    batch_size: int = 20,
    lr: float = 1e-4,
    epochs: int = 80,
    model_file_name: str = "unet_model.pth",
    device: str = "cuda",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataloader = tdata.DataLoader(train_data_, batch_size=batch_size)
    eval_dataloader = tdata.DataLoader(validation_data_, batch_size=batch_size)
    size = len(train_dataloader.dataset)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train()
        for batch_num, (X_batch, Y_batch) in enumerate(train_dataloader):
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
            a = 0.0
            b = 0.0
            for X_batch, Y_batch in eval_dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                a += dice_coeff(pred, Y_batch)
                b += jaccard_index(pred, Y_batch)

            a /= len(eval_dataloader)
            b /= len(eval_dataloader)
            print("Dice coeff: {}, Jaccard index: {}".format(a, b))

        torch.save(model.state_dict(), model_file_name)
        print("Saved PyTorch Model State to " + model_file_name)


def main():
    # add arg parser for loss function, batch size, learning rate, epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="dice", required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    assert args.loss in [
        "bce",
        "dice",
        "iou",
    ], "Loss function must be one of bce, dice, iou"
    device = "cuda"

    train_idx, validation_idx = tdata.random_split(
        range(1, 801), [640, 160], generator=torch.Generator().manual_seed(127)
    )
    train_data = RoadsDataset(
        root="data_augmented/training", image_idx=train_idx, device=device
    )
    validation_data = RoadsDataset(
        root="data_augmented/training", image_idx=validation_idx, device=device
    )
    model_file_name = "unet_model_{}.pth".format(args.loss)

    # Training ================================================
    unet_model = UNet(n_channels=3, n_classes=2).to(device)
    if os.path.exists(model_file_name):
        unet_model.load_state_dict(torch.load(model_file_name))
        print("Loaded model from " + model_file_name)

    if args.loss == "dice":
        loss_fun = dice_loss
    elif args.loss == "bce":
        loss_fun = bce_loss
    elif args.loss == "iou":
        loss_fun = iou_loss
    else:
        raise NotImplementedError("Loss function not recognized")

    train(
        model=unet_model,
        loss_fun=loss_fun,
        train_data_=train_data,
        validation_data_=validation_data,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )
    print("Done training!")


if __name__ == "__main__":
    main()
