import copy
from load_data import rotate_symmetry
import torch
import torchvision
from PIL import Image

im = Image.open("data/training/images/satImage_001.png")

ims = rotate_symmetry(im)
for i in range(len(ims)):
    ims[i].save("bruh/" + str(i) + ".png")

# ims_bis = copy.deepcopy(ims)
# ims_bis[1] = ims_bis[1].rotate(-90)
# ims_bis[2] = ims_bis[2].rotate(-180)
# ims_bis[3] = ims_bis[3].rotate(-270)
# ims_bis[4] = ims_bis[4].transpose(0)
# ims_bis[5] = ims_bis[5].transpose(0).rotate(-90)
# ims_bis[6] = ims_bis[6].transpose(0).rotate(-180)
# ims_bis[7] = ims_bis[7].transpose(0).rotate(-270)

# for i in range(len(ims_bis)):
#     ims_bis[i].save("bruh/" + str(i) + "bis.png")

image_to_tensor = torchvision.transforms.ToTensor()
tensors = [image_to_tensor(im) for im in ims]
tensors[1] = tensors[1].rot90(-1, [1, 2])
tensors[2] = tensors[2].rot90(-2, [1, 2])
tensors[3] = tensors[3].rot90(-3, [1, 2])
tensors[4] = tensors[4].flip(2)
tensors[5] = tensors[5].flip(2).rot90(-1, [1, 2])
tensors[6] = tensors[6].flip(2).rot90(-2, [1, 2])
tensors[7] = tensors[7].flip(2).rot90(-3, [1, 2])

tensor_to_image = torchvision.transforms.ToPILImage()
for i in range(len(tensors)):
    tensor_to_image(tensors[i]).save("bruh/" + str(i) + "ter.png")
