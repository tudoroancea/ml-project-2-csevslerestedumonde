import _thread
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image


class RoadDataset(torch.utils.data.Dataset):
    """
    The class RoadDataset loads the data and executes the pre-processing operations on it.
    More specifically, it re-applies the specified transform every time data is fetched via a dataloader.
    """

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        transform,
    ):
        self.transform = transform
        self.images = self.load_images(image_path)
        self.masks = self.load_images(mask_path)
        self.images_augmented = []
        self.masks_augmented = []

        # Data augmentation
        for i in range(len(self.images)):
            output = self.transform(image=self.images[i], mask=self.masks[i])
            self.images_augmented.append(output["image"])
            self.masks_augmented.append(output["mask"])

    def get_images(self):
        return self.images, self.masks

    @staticmethod
    def load_images(image_path):
        """This method loads the images from the given path"""
        images = []
        for img in os.listdir(image_path):
            path = os.path.join(image_path, img)
            image = Image.open(path)
            images.append(np.asarray(image))
            # images.append(cv2.imread(path))
        return np.asarray(images)

    def augment(self, index):
        """This method applies data augmentation to the images"""
        output = self.transform(image=self.images[index], mask=self.masks[index])
        self.images_augmented[index] = output["image"]
        self.masks_augmented[index] = output["mask"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """This method returns the image at a certain position and its mask"""
        image = self.images_augmented[index]
        mask = self.masks_augmented[index]
        _thread.start_new_thread(self.augment, (index,))
        return (image / 255), (mask.unsqueeze(0) > 100).float()


def get_loader(
    data_path: str,
    mask_path: str,
    transform,
    batch_size: int = 4,
) -> torch.utils.data.DataLoader:
    """Create the DataLoader class"""
    dataset = RoadDataset(
        data_path,
        mask_path,
        transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(127),
    )
