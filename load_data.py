from PIL import Image
import torch
import torch.utils.data as tdata
import torchvision
import os


class RoadsDataset(tdata.Dataset):
    """
    This class is used to load the training dataset and to store 3 type of arrays in
    memory:
    - images: the images of the dataset as torch tensors of shape (1, 3, 400, 400) with
        float values in [0, 1]
    - gt_images: the ground truth images of the dataset as torch tensors of shape
        (1, 400, 400) with float values in [0, 1]
    - gt_images_one_hot: the ground truth images of the dataset as torch tensors of
        shape (1, 2, 400, 400) with float values in {0, 1}
    """

    root: str
    idx: list
    images: torch.Tensor
    gt_images: torch.Tensor
    gt_images_one_hot: torch.Tensor

    def __init__(
        self,
        root: str,
        image_idx: list = None,
        device="cuda",
    ):
        self.root = root

        self.idx = image_idx if image_idx is not None else list(range(1, 801))
        self.images = []
        self.gt_images = []
        self.gt_images_one_hot = []
        image_to_tensor = torchvision.transforms.ToTensor()
        for i in self.idx:
            image_path = os.path.join(
                self.root, "images/satImage_" + str(i).zfill(3) + ".png"
            )
            img = image_to_tensor(Image.open(image_path)).type(torch.float32).to(device)
            img /= 255.0
            img = torch.unsqueeze(img, 0)
            self.images.append(img)

            gt_image_path = os.path.join(
                self.root, "groundtruth/satImage_" + str(i).zfill(3) + ".png"
            )
            gt_image = image_to_tensor(Image.open(gt_image_path)).to(device)
            gt_image /= 255.0

            gt_image_one_hot = torch.cat((gt_image, 1 - gt_image))
            gt_image_one_hot = torch.unsqueeze(gt_image_one_hot, 0)
            gt_image = torch.unsqueeze(gt_image, 0)
            self.gt_images.append(gt_image)
            self.gt_images_one_hot.append(gt_image_one_hot)

        self.images = torch.cat(self.images, 0)
        self.gt_images = torch.cat(self.gt_images, 0)
        self.gt_images_one_hot = torch.cat(self.gt_images_one_hot, 0)

        print("Loaded {} images from {}".format(len(self.idx), root))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item: int) -> tuple:
        return self.images[item, :, :, :], self.gt_images_one_hot[item, :, :, :]


def rotate(image: Image) -> list:
    images = [image]
    images.append(image.rotate(90))
    images.append(image.rotate(180))
    images.append(image.rotate(270))
    return images


def symmetry(images: list) -> list:
    for i in range(len(images)):
        images.append(images[i].transpose(0))
    return images


def rotate_symmetry(image: Image) -> list:
    images = symmetry(rotate(image))
    return images


def augment_data():
    source_path = "data/training/"
    destination_path = "data_augmented/training/"
    for i in range(1, 101):
        image_name = "satImage_" + str(i).zfill(3) + ".png"
        image = Image.open(source_path + "images/" + image_name)
        gt = Image.open(source_path + "groundtruth/" + image_name)
        images = rotate_symmetry(image)
        gts = rotate_symmetry(gt)
        for im in range(len(images)):
            image_name = "satImage_" + str(im * 100 + i).zfill(3) + ".png"
            images[im].save(destination_path + "images/" + image_name)
            gts[im].save(destination_path + "groundtruth/" + image_name)
