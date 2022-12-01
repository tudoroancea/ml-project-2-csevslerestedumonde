import torch
from model import UNet
from utils import *
from load_data import RoadsDataset
from post_processing import *
from PIL import Image
import torchvision

# select the line depending on the platform
# device = "cpu"
device = "cuda"

torch.manual_seed(127)
read_image = torchvision.transforms.ToTensor()

# Classifying ================================================
unet_model = UNet(3, 2).to(device)
unet_model.load_state_dict(torch.load("unet_model2.pth", map_location=torch.device('cpu')))
unet_model.eval()
images = []
for i in range(1, 51):
    print(i)
    img = read_image(Image.open("data/test_set_images/test_{}/test_{}.png".format(i, i))).type(torch.float32).to(device)
    img /= 255.0

    img = img.to(device)
    with torch.no_grad():
        Y_pred = torch.squeeze(unet_model(torch.unsqueeze(img, 0)))
        Y_pred = Y_pred[0,:,:]
        Y_pred = torch.unsqueeze(Y_pred, 0)
    images.append(Y_pred)
post_processing(images)
#==========================================================

