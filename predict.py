from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms

from main import config
from model import *
from utils import *


def convert_image_to_tensor(args, input_image):
    image = Image.open(input_image)
    image_pt = torch.as_tensor(np.array(image, copy=True))
    image_pt = image_pt.permute((2, 0, 1))   # CHW
    
    image_transforms = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.Grayscale(args.image_channel)
    ])
    
    image_pt = image_transforms(image_pt) / 255
    image_pt = normalize(image_pt, 0.5, 0.5)   # normalize to range [-1, 1]
    return image_pt

def predict():
    pass


if __name__ == "__main__":
    args = config()
    convert_image_to_tensor(args, "dataset/eval/iceland.jpg")