# =================================================================================
# IMPORTS
# =================================================================================
import _thread
from cProfile import run
import os
import re

#to plot and save images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

#pytorch
import torch
import torch.utils.data
import torchvision
from torchvision.transforms.functional import resize

#library segmentation models for UNet, LinkNet and metrics implementations 
from segmentation_models_pytorch import metrics
import segmentation_models_pytorch as smp

#progress bar
import tqdm

from albumentations.pytorch import ToTensorV2
import albumentations

#set the seed espacially for albumentations
import random

# =================================================================================
# GENERAL PARAMETERS
# =================================================================================
random.seed(127)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", DEVICE)
SUBMISSION_THRESHOLD = 0.25 # Given, DO NOT change it
random.seed(127)

# =================================================================================
# PATHS
# =================================================================================
# the 100 training images are divided in two sets : training (1 to 80) and validating (81 to 100)
TRAIN_IMAGE_PATH = "data/training/images"
TRAIN_MASK_PATH = "data/training/groundtruth"
VALIDATION_IMAGE_PATH = "data/validating/images"
VALIDATION_MASK_PATH = "data/validating/groundtruth"

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
        # Remember transforms
        self.transform = transform

        # Load images and masks
        self.images = self.load_images(image_path)
        self.masks = self.load_images(mask_path)

        # Augmented images and masks
        self.images_augmented = []
        self.masks_augmented = []

        # Data augmentation using transforms
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

        return np.asarray(images)

    def augment(self, index):
        """This method applies data augmentation to the images again to change precedent augmentation transformations"""
        output = self.transform(image=self.images[index], mask=self.masks[index])
        self.images_augmented[index] = output["image"]
        self.masks_augmented[index] = output["mask"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """This method returns the image at a certain position and its mask"""
        image = self.images_augmented[index]
        mask = self.masks_augmented[index]

        # Start a new thread to augment the data (thread level parallelism)
        _thread.start_new_thread(self.augment, (index,))

        # Return scaled image and mask
        return (image / 255), (mask.unsqueeze(0) > 200).float()


def get_loader(
    data_path: str,
    mask_path: str,
    transform,
    batch_size: int = 4,
):
    """Create the pytorch DataLoader"""
    # Use our dataset and defined transformations
    dataset = RoadDataset(
        data_path,
        mask_path,
        transform,
    )

    # Use the dataset in the torch dataloader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # Specify batch size
        shuffle=True, # Shuffle the data to avoid learning the order
        pin_memory=True, # Copy tensors to CUDA pinned memory
        generator=torch.Generator().manual_seed(127), # Set seed for reproducibility
    )


def compute_metrics(loader, model, device):
    """Compute the accuracy rate on the validation dataset with the input model"""
    # Set model to evaluation mode
    model.eval()

    # Dictionnary to store metrics
    logs = dict()

    # Parameters to compute metrics
    num_correct = 0
    num_pixels = 0

    # Metrics computation
    f1_score = 0
    precision = 0
    recall = 0

    # Eval mode, so no gradient computation because no training
    with torch.no_grad():
        # Iterate over the dataset
        for x, y in loader:
            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # # Create simple transformed images
            # x = create_simple_transformed_images(x, [0, 90, 180, 270], True)

            # Compute output via the model
            output = model(x)

            # Drop the first dimension of the output (batch size)
            output = output[:, -1, :, :].unsqueeze(1)

            # Apply sigmoid to the output and round it to 0 or 1 to get the prediction for each pixel
            pred: torch.Tensor = (torch.sigmoid(output) >= 0.5)

            # Compute the number of correct pixels and the total number of pixels
            num_correct += torch.sum(pred == y).item()
            num_pixels += torch.numel(pred)
            
            # True positive and negative, false positive and negative using segmentation models functions
            tp, fp, fn, tn = metrics.get_stats(pred, y.int(), mode='binary')

            # Compute F1 score, precision and recall
            f1_score += metrics.f1_score(tp, fp, fn, tn, reduction='micro')
            precision += metrics.precision(tp, fp, fn, tn, reduction='micro')
            recall += metrics.recall(tp, fp, fn, tn, reduction='micro')

    # Add metrics to the dictionnary and multiply by 100 to get a percentage
    logs["acc"] = num_correct / num_pixels * 100
    logs["f1 score"] = f1_score.cpu().numpy() / len(loader) * 100
    logs["precision"] = precision.cpu().numpy() / len(loader) * 100
    logs["recall"] = recall.cpu().numpy() / len(loader) * 100

    # Set model back to training mode
    model.train()

    # Return logs
    return logs


def epoch(model, loader, optimizer, criterion, scaler):
    """Train the model for one epoch"""
    # Total loss for the epoch
    total_loss = 0

    # Iterate over the dataset
    for data, target in loader:
        # Move data to device
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        # Compute output via the model with mixed precision to speed up training
        with torch.cuda.amp.autocast():
            # Compute output
            output = model(data)

            # Define local loss variable
            loss = 0
            # Compute loss for each output
            for i in range(output.shape[1]):
                # Get the output for the current time step, add a dimension to match the target shape
                pred = output[:, i, :, :].unsqueeze(1)

                # Compute the loss for the current time step
                loss += criterion(pred, target)

            # Add the loss for the current time step to the total loss
            total_loss += loss.item()

        # Backpropagation and scaler to avoid vanishing gradient
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Return total loss
    return total_loss


def train(
    model,
    model_name,
    train_loader,
    validation_loader,
    lr_max: float = 1.0e-3, # Learning rate minimal
    lr_min: float = 1.0e-5, # Learning rate minimal
    epochs: int = 10, # Number of epochs
):
    """Train the model"""
    # Create the log file
    log_file_name = os.path.relpath(os.path.join("logs", model_name + ".csv"))
    with open(log_file_name, "w") as f:
        f.write("epoch,loss,f1,iou,accuracy,precision,recall\n")

    # Create a checkpoint file to store the best model
    model_file_name = os.path.join("checkpoints", model_name + ".pth")

    # Define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)

    # Define the scheduler and scaler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr_min)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.cuda.amp.GradScaler()

    # Variables to store current best metrics
    max_f1 = 0
    min_loss = 0.5

    # Train the model, then save the training logs and the best model
    loop = tqdm.tqdm(range(epochs)) 
    for e in loop:
        # Train the model for one epoch and get the loss
        loss = epoch(model, train_loader, optimizer, criterion, scaler)
        # if loss < 0.5:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Take a step in the scheduler
        scheduler.step()

        # Compute the metrics on the validation set
        metrics = compute_metrics(validation_loader, model, DEVICE)

        # Save the model if it surpasses the current best metrics
        # in terms of F1 score
        if metrics["f1 score"] > max_f1:
            max_f1 = metrics["f1 score"]
            if max_f1 > 80.0:
                torch.save(model, model_file_name + ".maxf1")

        # or in terms of loss
        if loss < min_loss:
            torch.save(model, model_file_name + ".minloss")
            min_loss = loss

        # Save the logs into a file
        with open(log_file_name, "a") as f:
            f.write(
                "{},{},{},{},{},{},{},{}\n".format(
                    e,
                    loss,
                    metrics["f1 score"],
                    0,
                    metrics["acc"],
                    metrics["precision"],
                    metrics["recall"],
                    min_loss
                )
            )

        # Update the progress bar
        loop.set_postfix(loss=loss, f1_score=metrics["f1 score"], max_f1=max_f1, min_loss=min_loss)

    # Save the logs into a file
    torch.save(model, model_file_name)


# Define the data augmentation with albumentations for the training set and convert it to tensor
train_transform = albumentations.Compose(
    [
        albumentations.Flip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Rotate(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.CoarseDropout(min_holes= 5, max_holes=20, min_height=5, max_height=20, min_width=5, max_width=20, p=0.5),
        albumentations.OpticalDistortion(p=0.5),
        albumentations.GridDistortion(p=0.5),
        albumentations.ElasticTransform(p=0.5),
        albumentations.PiecewiseAffine(p=0.5),
        ToTensorV2(),
    ]
)

val_transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),
        ToTensorV2(),
    ]
)

# Create the data loader for the validation
val_loader = get_loader(
    data_path=VALIDATION_IMAGE_PATH,
    mask_path=VALIDATION_MASK_PATH,
    transform=val_transform,
    batch_size=4,
)


# our best model
batch_size = 16
train_loader = get_loader(
        data_path=TRAIN_IMAGE_PATH,
        mask_path=TRAIN_MASK_PATH,
        transform=train_transform,
        batch_size=batch_size, # Choose the batch size
    )

train(
    model=smp.Linknet(
        encoder_name="resnet152",
        encoder_depth=4,
        encoder_weights="imagenet",
        in_channels=3,
    ).to(DEVICE),
    model_name="BESTMODEL",
    epochs=150, #increase
    train_loader=train_loader,
    validation_loader=val_loader,
    lr_min=1e-05,
    lr_max=1e-05,
)

def patch_to_label(patch):
    df = np.mean(patch)
    if df > SUBMISSION_THRESHOLD:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    print(image_filename, os.path.basename(image_filename))
    img_number = int(re.search(r"\d+", os.path.basename(image_filename)).group(0))
    print(img_number)
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))


def create_postprocessing_images(images, rotations, transposes):
    """Apply transformations to the image and return different prospectives"""
    ims = []
    for image in images:
        for rotation in rotations:
            ims.append(albumentations.rotate(image, rotation))
            if transposes:
                im = albumentations.hflip(image)
                ims.append(albumentations.rotate(im, rotation))
    ims = np.array(ims)
    ims = torch.tensor(ims).transpose(1, -1).transpose(2, -1).float()
    return ims


def combine_postprocessing_images(images, rotations, transposes):
    """Combine predictions of different prospectives"""
    outputs = []
    index = 0
    while index < len(images):
        output = np.zeros(images[0].shape)
        for rotation in rotations:
            im = images[index, 0]
            output += albumentations.rotate(im, -rotation)
            index += 1
            if transposes:
                im = images[index, 0]
                im = albumentations.rotate(im, -rotation)
                output += albumentations.hflip(im)
                index += 1
        output = output / len(images)
        outputs.append(output)
    return np.array(outputs)


def create_submission(model_name: str):
    model_file_name = os.path.join("checkpoints", model_name + ".pth.maxf1")
    model = torch.load(model_file_name, map_location=torch.device(DEVICE)).to(DEVICE)
    model.eval()

    # Create the directory to store the predictions
    path = "data/test_set_images"
    pred_path = "predictions/" + model_name
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    # For each image, apply postprocessing augmentation, make predictions and save predictions
    for image in tqdm.tqdm(os.listdir(path)):
        img_path = os.path.join(path, image, image + ".png")
        im = np.asarray(Image.open(img_path)) / 255
        ims = create_postprocessing_images(
            [im], rotations=[0, 90, 180, 270], transposes=True
        )

        with torch.no_grad():
            output = model(ims.to(DEVICE))
            predicts = torch.sigmoid(output).cpu().detach()

        predict = combine_postprocessing_images(
            predicts.numpy(), rotations=[0, 90, 180, 270], transposes=True
        ).reshape((608, 608))
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1
        predict *= 255
        Image.fromarray(predict).convert("L").save(
            os.path.join(pred_path, image) + ".png"
        )

    # Generate the submission file
    submission_filename = os.path.join(
        "submissions", "submission_{}.csv".format(model_name)
    )
    image_filenames = []
    for i in range(1, 51):
        image_filename = pred_path + "/test_" + str(i) + ".png"
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)


#finally create the submissions
create_submission("unet_smp")
create_submission("unet_no_preprocessing")
create_submission("linknet18_smp")
create_submission("linknet34_smp")
create_submission("linknet50_smp")
create_submission("linknet101_smp")
create_submission("linknet152_smp")
create_submission("BESTMODEL")