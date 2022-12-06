import argparse
from email import generator
import os
from random import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt

from load_data import TrainRoadsDataset
from model import UNet
from utils import *


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get the model file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    model_filename = args.model

    # load (part of) the train set to test the model
    data = TrainRoadsDataset(
        path="data_augmented/training", image_idx=list(range(1, 21)), device=device
    )

    # create folder for output
    output_path = "test_output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    # load the model
    unet_model = UNet(
        n_channels=3,
        n_classes=1,
        sigmoid=True,
    ).to(device)

    unet_model.load_state_dict(torch.load(model_filename))
    unet_model.eval()

    # apply the model to each image, save the masks and print dice coefficient and jaccard index
    dice = []
    jaccard = []
    for i, j in enumerate(data.idx):
        X = data.images[i]
        Y = data.gt_images[i]

        with torch.no_grad():
            Y_pred = unet_model(X.unsqueeze(0))
            print("max Y_pred: ", torch.max(Y_pred[0, 0, :, :]).item())
            Y_mask = proba_to_mask(Y_pred, args.threshold)

        dice.append(dice_coeff(Y_mask, Y.unsqueeze(0)).item())
        jaccard.append(jaccard_index(Y_mask, Y.unsqueeze(0)).item())
        print(
            "Image {}: Dice coeff = {}, Jaccard index = {}".format(
                j, dice[-1], jaccard[-1]
            )
        )
        # recast the tensors to 0-255 range for saving
        Y_mask = 255.0 * Y_mask[0, 0, :, :].cpu().numpy()  # shape (400, 400)
        Y = 255.0 * torch.squeeze(Y).cpu().numpy()  # shape (400, 400)
        X = 255.0 * np.moveaxis(X.cpu().numpy(), 0, -1)  # shape (400, 400, 3)

        plt.imsave(
            os.path.join(output_path, "{}_pred.png".format(j)), Y_mask, cmap="gray"
        )
        plt.imsave(os.path.join(output_path, "{}_gt.png".format(j)), Y, cmap="gray")
        plt.imsave(os.path.join(output_path, "{}_img.png".format(j)), X)

    print(
        "Mean Dice coeff = {}, mean Jaccard index = {}".format(
            np.mean(dice), np.mean(jaccard)
        )
    )


def compute_roc():
    device = "cuda"

    # get the model file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    model_filename = args.model

    # load the model
    unet_model = UNet(3, 1).to(device)
    unet_model.load_state_dict(torch.load(model_filename))
    unet_model.eval()

    # load (part of) the train set to test the model
    a, _ = torch.utils.data.random_split(
        list(range(1, 801)), [400, 400], generator=torch.Generator().manual_seed(127)
    )
    data = TrainRoadsDataset(path="data_augmented/training", image_idx=a, device=device)
    dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=20)
    # compute ROC curve for different thresholds
    thresholds = np.logspace(-4, -2, 20)
    fpr_values = []
    tpr_values = []
    for threshold in thresholds:
        fpr = 0.0
        tpr = 0.0
        for X, Y_one_hot in dataloader:
            # X, Y_one_hot = next(iter(dataloader))
            with torch.no_grad():
                Y_pred = unet_model(X)
                Y_pred = proba_to_mask(Y_pred, threshold)
            fpr += fpr_score(Y_pred, Y_one_hot).item()
            tpr += tpr_score(Y_pred, Y_one_hot).item()

        fpr_values.append(fpr / len(dataloader))
        tpr_values.append(tpr / len(dataloader))
    fpr_values = np.array(fpr_values)
    tpr_values = np.array(tpr_values)

    print("fpr_values: ", fpr_values)
    print("tpr_values: ", tpr_values)
    plt.plot(fpr_values, tpr_values)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("roc.png", dpi=300)

    # find best threshold
    best_threshold = thresholds[np.argmax(tpr_values - fpr_values)]
    print("Best threshold: ", best_threshold)


if __name__ == "__main__":
    main()
    # compute_roc()
