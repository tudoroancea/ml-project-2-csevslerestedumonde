# Project Road Segmentation

## Setup
Once you have unzipped `submission.zip` in a folder `submission`, you should 
download a fresh copy of the data from AIcrowd and place it inside 
`submission/data` in such a way that the following files are present:
```bash
submission/data/test_set_images.zip
submission/data/training.zip
```
Now you can just run `setup.sh` to unzip the data files, partition the training 
data into training and validation sets.

Also make sure to have the following dependencies installed:
- Python >= 3.7
- torch
- torchvision
- albumentations
- Pillow
- tqdm
- segmentation_models_python