import copy
import numpy as np
import torch

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