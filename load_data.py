from PIL import Image
import torch
import torch.utils.data as tdata
import torchvision
from torch.nn import functional as F
import numpy as np
import os


def symmetry(images: list) -> list:
    for i in range(len(images)):
        images.append(images[i].transpose(0))
    return images


def rotate(image: Image) -> list:
    images = [image]
    images.append(image.rotate(90))
    images.append(image.rotate(180))
    images.append(image.rotate(270))
    return images


def rotate_symmetry(image: Image) -> list:
    images = symmetry(rotate(image))
    return images


def data_augmentation():
    source_path = "data/training/"
    destination_path = "data_augmented/training/"
    for i in range(1, 101):
        image_name = "satImage_" + str(i).zfill(3) + ".png"
        image = Image.open(source_path + "images/" + image_name)
        gt = Image.open(source_path + "groundtruth/" + image_name)
        images = rotate_symmetry(image)
        gts = rotate_symmetry(gt)
        for im in range(len(images)):
            image_name = "satImage_" + str(im * 100 + i).zfill(4) + ".png"
            images[im].save(destination_path + "images/" + image_name)
            gts[im].save(destination_path + "groundtruth/" + image_name)


def check() -> bool:
    # check if the directory data_augmented exists
    if not os.path.exists("data_augmented"):
        # create the directory data_augmented
        os.makedirs("data_augmented")
        # create the directory data_augmented/training
        os.makedirs("data_augmented/training")
        # create the directory data_augmented/training/images
        os.makedirs("data_augmented/training/images")
        # create the directory data_augmented/training/groundtruth
        os.makedirs("data_augmented/training/groundtruth")
        return False
    return True


class RoadsDataset(tdata.Dataset):
    root: str
    num_images: int
    images: torch.Tensor
    gt_images: torch.Tensor
    gt_images_one_hot: torch.Tensor

    def __init__(
        self,
        root: str,
        num_images=20,
        device="cuda",
    ):
        self.root = root
        self.num_images = num_images
        assert 10 <= num_images <= 800
        self.images = []
        self.gt_images = []
        self.gt_images_one_hot = []
        image_to_tensor = torchvision.transforms.ToTensor()
        for i in range(num_images):
            image_path = os.path.join(
                self.root, "images/satImage_" + str(i + 1).zfill(4) + ".png"
            )
            img = image_to_tensor(Image.open(image_path)).type(torch.float32).to(device)
            img /= 255.0
            img = torch.unsqueeze(img, 0)
            self.images.append(img)

            gt_image_path = os.path.join(
                self.root, "groundtruth/satImage_" + str(i + 1).zfill(4) + ".png"
            )
            gt_image = image_to_tensor(Image.open(gt_image_path)).to(device)
            gt_image = torch.unsqueeze(gt_image, 0)
            self.gt_images.append(gt_image)

            gt_image_one_hot = torch.cat((gt_image / 255, 1 - gt_image / 255))
            gt_image_one_hot = torch.unsqueeze(gt_image_one_hot, 0)
            self.gt_images_one_hot.append(gt_image_one_hot)

        self.images = torch.cat(self.images, 0)
        self.gt_images = torch.cat(self.gt_images, 0)
        self.gt_images_one_hot = torch.cat(self.gt_images_one_hot, 0)

        print("Loaded {} images from {}".format(num_images, root))

    def __len__(self):
        return self.num_images

    def __getitem__(self, item: int) -> tuple:
        return self.images[item, :, :, :], self.gt_images_one_hot[item, :, :, :]


if __name__ == "__main__":
    if not check():
        data_augmentation()