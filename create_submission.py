import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from mask_to_submission import masks_to_submission
from model import UNet
from utils import *
from utils import proba_to_mask


def main():
    # get the model file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    model_filename = args.model

    device = "cuda"  # select "cpu" if you don't have a GPU

    # load the model
    unet_model = UNet(3, 2).to(device)
    unet_model.load_state_dict(
        torch.load(model_filename, map_location=torch.device(device))
    )
    unet_model.eval()

    # create folder for submission
    submission_path = "submission"
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)
    else:
        for file in os.listdir(submission_path):
            os.remove(os.path.join(submission_path, file))

    # apply the model to each test image and save the masks
    masks_file_names = []
    to_tensor = torchvision.transforms.ToTensor()
    for i in range(1, 51):
        img = (
            to_tensor(
                Image.open("data/test_set_images/test_{}/test_{}.png".format(i, i))
            )
            .type(torch.float32)
            .to(device)
        )
        img /= 255.0

        with torch.no_grad():
            Y_pred = torch.squeeze(unet_model(img.unsqueeze(0)))  # shape (2, 608, 608)
            Y_pred = Y_pred[0, :, :]  # shape (608, 608)

        mask = proba_to_mask(Y_pred, 1.0e-3)
        masks_file_names.append(
            os.path.join(submission_path, "mask_" + str(i).zfill(3) + ".png")
        )
        plt.imsave(
            masks_file_names[-1],
            (mask * 255).cpu().numpy(),
            cmap="gray",
        )

    # create the submission file from all the saved masks
    masks_to_submission(
        os.path.join(submission_path, "submission.csv"), *masks_file_names
    )


if __name__ == "__main__":
    main()
