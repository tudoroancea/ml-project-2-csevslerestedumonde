import argparse
import os
import sys
from tkinter.tix import Tree
import torch
import torch.nn as nn
import torch.utils.data as tdata

from load_data import TrainRoadsDataset
from model import UNet
from utils import *


def train(
    model: nn.Module,
    loss_fun,
    train_data_: TrainRoadsDataset,
    validation_data_: TrainRoadsDataset,
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
    best_dice = 0
    best_iou = 0
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
        dice = 0.0
        iou = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in eval_dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                dice += dice_coeff(pred, Y_batch)
                iou += jaccard_index(pred, Y_batch)

        dice /= len(eval_dataloader)
        iou /= len(eval_dataloader)
        print("Dice coeff: {}, Jaccard index: {}".format(dice, iou))
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), model_file_name)
            print("Saved model to " + model_file_name)
        if iou > best_iou:
            best_iou = iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="dice", required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    print("Training with args: ", args)
    assert args.loss in [
        "bce",
        "dice",
        "iou",
    ], "Loss function must be one of bce, dice, iou"
    device = "cuda"

    train_idx, validation_idx = tdata.random_split(
        range(1, 801), [640, 160], generator=torch.Generator().manual_seed(127)
    )
    sys.stdout.write("Loading training data... ")
    train_data = TrainRoadsDataset(
        path="data_augmented/training", image_idx=train_idx, device=device
    )
    sys.stdout.write("Loading validation data... ")
    validation_data = TrainRoadsDataset(
        path="data_augmented/training", image_idx=validation_idx, device=device
    )
    model_file_name = "unet_model_{}.pth".format(args.loss)

    # Training ================================================
    unet_model = UNet(n_channels=3, n_classes=1, sigmoid=args.loss != "bce").to(device)
    if os.path.exists(model_file_name):
        unet_model.load_state_dict(torch.load(model_file_name))
        print("Loaded model from " + model_file_name)

    if args.loss == "dice":
        loss_fun = dice_loss
    elif args.loss == "bce":
        weight = (400 * 400 * (len(train_data) + len(validation_data))) / (
            torch.sum(train_data.gt_images) + torch.sum(validation_data.gt_images)
        ) - 1
        print("using pos_weight={} for BCE loss".format(weight))
        loss_fun = nn.BCEWithLogitsLoss(pos_weight=weight)
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
        model_file_name=model_file_name,
    )
    print("Done training!")


if __name__ == "__main__":
    main()
