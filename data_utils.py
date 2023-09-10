from PIL import Image

import numpy as np

import torch
import torchvision.transforms as transforms
from utils import *


def read_and_convert_image_to_pt(image_dir) -> torch.Tensor:
    """
    Loads an image with PIL and converts to torch.Tensor in the CHW sequence.
    If the image is in the RGBA format, the alpha channel is deleted.
    """
    image = Image.open(image_dir)
    image_pt = torch.as_tensor(np.array(image, copy=True))   # HWC
    if image_pt.dim() == 2:
        image_pt = image_pt.unsqueeze(dim=-1)
    if image_pt.size(-1) == 4:
        image_pt = image_pt[:, :, :3]   # RGBA to RGB
    image_pt = image_pt.permute((2, 0, 1))   # CHW
    return image_pt


def image_transform(image_np):
    output = to_tensor(image_np, normalize=True)
    # output = output.reshape(output.size(0), -1)
    output = normalize(output, 0.5, 0.5) # normalize to [-1.0, 1.0] range
    return output


def label_transform(label_np):
    output = to_tensor(label_np, normalize=False, dtype=torch.int64)
    return output


def eval_transform(args, image_dir):
    resizer = transforms.Resize((args.image_height, args.image_width))
    grayscaler = transforms.Grayscale(args.image_channel)
    
    image_pt = read_and_convert_image_to_pt(image_dir)
    image_pt = resizer(image_pt)
    image_pt = grayscaler(image_pt) if image_pt.size(0) == 3 else image_pt
    image_pt = 255 - image_pt   # invert background color # https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa
    image_pt = normalize(image_pt / 255, 0.5, 0.5)
    if args.model_type.lower() == "ann":
        image_pt = image_pt.flatten()
    
    return image_pt


def custom_collate_fn(desired_height, desired_width, image_pt, pad_mode="right"):
    """
    If the size of input image (image_pt) is smaller than the desired size,
    return a padded image in the desired size.
    """
    # desired_height = args.image_height
    # desired_width = args.image_width
    c, h, w = image_pt.size()
    
    if h == desired_height and w == desired_width:
        return image_pt
    
    padded_img = torch.FloatTensor(c, desired_height, desired_width).fill_(0)
    height_pad = (desired_height - h) // 2
    width_pad = (desired_width - h) // 2
    
    if pad_mode == "right":
        padded_img[:, height_pad:(desired_height-height_pad), :w] = image_pt
        
    elif pad_mode == "left":
        padded_img[:, height_pad:(desired_height-height_pad), w:] = image_pt
        
    elif pad_mode == "center":
        padded_img[:, height_pad:(desired_height-height_pad), width_pad:(desired_width-width_pad)] = image_pt
    
    return padded_img