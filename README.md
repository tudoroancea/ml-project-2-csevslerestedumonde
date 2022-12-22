# Project Road Segmentation

This repo contains the code of the group CSEVsLeResteDuMonde (Tudor Oancea, 
Philippe Servant and Pierre Ancey) for the AIcrowd Road Segmentation challenge.

## Setup
Once you have unzipped `submission.zip` in a folder `submission`, you should 
download a fresh copy of the data from AIcrowd and place it inside 
`submission/data` in such a way that the following files are present:
```bash
submission/data/test_set_images.zip
submission/data/training.zip
```
Now you can just run `submission/setup.sh` to unzip the data files, partition the training 
data into training and validation sets.

Also make sure to have the following dependencies installed:
- Python >= 3.7
- torch  (ML framework use for training the models)
- albumentations (data augmentation library)
- segmentation_models_python  (collection of pre-trained segmentation models)
- Pillow
- tqdm

Then you can either run [`road_segmentation.ipynb`](road_segmentation.ipynb) to 
have all the models and cross-validation or only [`run.py`](run.py) to only train 
the best model.
