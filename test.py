import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from load_data import RoadsDataset
from model import UNet
from utils import *
from utils import proba_to_mask


def main():
    # get the model file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    model_filename = args.model

    device = "cuda"

    # load (part of) the train set to test the model
    data = RoadsDataset(
        root="data_augmented/training", image_idx=list(range(1, 21)), device=device
    )

    # create folder for output
    output_path = "test_output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    # load the model
    unet_model = UNet(3, 2).to(device)
    unet_model.load_state_dict(torch.load(model_filename))
    unet_model.eval()

    # apply the model to each image, save the masks and print dice coefficient and jaccard index
    dice = []
    jaccard = []
    for i, j in enumerate(data.idx):
        X = data.images[i]
        Y = data.gt_images[i]
        Y_one_hot = data.gt_images_one_hot[i]
        with torch.no_grad():
            Y_pred = unet_model(X.unsqueeze(0))
            print("max Y_pred: ", torch.max(Y_pred[0, 0, :, :]).item())
            Y_pred = proba_to_mask(Y_pred, 1.0e-3)

        dice.append(dice_coeff(Y_pred, Y_one_hot.unsqueeze(0)).item())
        jaccard.append(jaccard_index(Y_pred, Y_one_hot.unsqueeze(0)).item())
        print(
            "Image {}: Dice coeff = {}, Jaccard index = {}".format(
                j, dice[-1], jaccard[-1]
            )
        )
        Y_pred = 255.0 * Y_pred[0, 0, :, :].cpu().numpy()
        Y = 255.0 * torch.squeeze(Y).cpu().detach().numpy()
        X = 255.0 * np.moveaxis(X.cpu().detach().numpy(), 0, -1)

        plt.imsave(
            os.path.join(output_path, "{}_pred.png".format(j)), Y_pred, cmap="gray"
        )
        plt.imsave(os.path.join(output_path, "{}_gt.png".format(j)), Y, cmap="gray")
        plt.imsave(os.path.join(output_path, "{}_img.png".format(j)), X)

    print(
        "Mean Dice coeff = {}, mean Jaccard index = {}".format(
            np.mean(dice), np.mean(jaccard)
        )
    )


if __name__ == "__main__":
    main()
