from load_data import augment_data
import os

if __name__ == "__main__":
    if not os.path.exists("data_augmented"):
        os.makedirs("data_augmented")
        os.makedirs("data_augmented/training")
        os.makedirs("data_augmented/training/images")
        os.makedirs("data_augmented/training/groundtruth")
        augment_data()
        print("Data has been augmented!")
    else:
        print("Data has already been augmented!")
