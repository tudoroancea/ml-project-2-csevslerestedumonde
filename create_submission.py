import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from mask_to_submission import masks_to_submission
from model import UNet
from utils import *
from load_data import rotate_symmetry


def main():
    # get the model file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.001)
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
        image = Image.open("data/test_set_images/test_{}/test_{}.png".format(i, i))
        transformed_images = rotate_symmetry(image)
        probas = []
        for transformed_image in transformed_images:
            img = to_tensor(transformed_image).type(torch.float32).to(device)
            img /= 255.0
            with torch.no_grad():
                Y_pred = torch.squeeze(
                    unet_model(img.unsqueeze(0))
                )  # shape (2, 608, 608)
                Y_pred = Y_pred[0, :, :]  # shape (608, 608)

            probas.append(Y_pred)

        probas[1] = probas[1].rot90(-1)
        probas[2] = probas[2].rot90(-2)
        probas[3] = probas[3].rot90(-3)
        probas[4] = probas[4].flip(1)
        probas[5] = probas[5].flip(1).rot90(-1)
        probas[6] = probas[6].flip(1).rot90(-2)
        probas[7] = probas[7].flip(1).rot90(-3)
        probas = torch.mean(torch.stack(probas), dim=0)

        mask = proba_to_mask(probas, args.threshold)
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
