# Project Road Segmentation

## Setup instructions
1. Download the data from AIcrowd, place all the contents (`training.zip`, `test.zip`, `sample_submission.csv`, `mask_to_submission.py`, `submission_to_mask.py`) in a folder named data, and finally unzip the zip files.
Execute the following command under windows:
```
python augment_data.py
```
Or under linux:
```
python3 augment_data.py
```
It will create a folder named `augmented_data` with the augmented data.

2. Make sure you have a valid interpreter with python 3.7. Install the requirements by running the command:
```
pip install -r requirements.txt
```
 On izar, run `module load gcc cuda py-torch py-torchvision` to load the necessary modules.
