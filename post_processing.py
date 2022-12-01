import torch
from mask_to_submission import masks_to_submission
from torchvision.utils import save_image
# torchvision.utils.save_image(gt_image, "2.png")


def proba_to_zeros_ones(proba: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """Converts a probability tensor to a binary tensor.
    Args:
        proba (torch.Tensor): A tensor of probabilities.
        threshold (float): A threshold to convert probabilities to binary values.
    Returns:
        torch.Tensor: A binary tensor.
    """
    return (proba > threshold).long()


def save_images(images: list, path: str):
    """Saves a list of images to a path.
    Args:
        images (list): A list of images.
        path (str): A path to save the images to.
    """
    for i, image in enumerate(images):
        save_image(image, path + "image_" + str(i + 1).zfill(3) + ".png")


def to_submission(path: str):
    """Converts a folder of images to a submission file.
    Args:
        path (str): A path to the folder of images.
    """
    images = [
        Image.open(path + "image_" + str(i + 1).zfill(3) + ".png") for i in range(100)
    ]
    masks_to_submission("submission.csv", *images)


def post_processing(images: list, threshold=0.5, path: str = "post_processed_images/"):
    """Performs post processing on a list of images.
    Args:
        images (list): A list of images.
        threshold (float): A threshold to convert probabilities to binary values.
    Returns:
        list: A list of post processed images.
    """
    post_processed_images = []
    for image in images:
        image = proba_to_zeros_ones(image, threshold)
        post_processed_images.append(image)
    save_images(post_processed_images, path)
    to_submission(path)
