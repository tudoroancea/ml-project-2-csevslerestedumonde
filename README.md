# Project Road Segmentation

## Setup instructions
1. Download the data from AIcrowd, place all the contents (`training.zip`, 
`test.zip`, `sample_submission.csv`, `mask_to_submission.py`, 
`submission_to_mask.py`) in a folder named data, and finally unzip the zip files.
Then run once `augment_data.py` which will create a a folder named `augmented_data` with the augmented data.
2. Make sure you have a valid interpreter with python 3.7 or higher and the 
following dependencies:
```
numpy
matplotlib
Pillow
torch
torchvision
```
3. On izar run `module load gcc cuda py-torch py-torchvision` to load the necessary modules.
