import copy
import random

import torch
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def to_tensor(
    input: np.ndarray,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32
):
    output = torch.from_numpy(copy.deepcopy(input))   # avoid non-writeable ndarray error msg
    output = output.type(dtype)   # np.uint8 -> dtype
    if normalize:
        output = output/255
    return output


def normalize(
    input: torch.Tensor,
    mean: float,
    std: float
):
    output = (input-mean)/std
    return output