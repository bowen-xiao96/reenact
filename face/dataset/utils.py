import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


def to_tensor(pil_image, normalize):
    # pil_image: h * w * c
    arr = np.array(pil_image)
    arr = np.transpose(arr, (2, 0, 1))
    arr = arr.astype(np.float32)

    if normalize:
        arr /= 255.0

    tensor = torch.from_numpy(arr)
    return tensor


def to_pil_image(torch_tensor, normalize):
    torch_tensor = torch.squeeze(torch_tensor)

    arr = torch_tensor.data.cpu().numpy()  # c * h * w
    arr = np.transpose(arr, (1, 2, 0))

    if normalize:
        arr *= 255.0

    arr = arr.astype(np.uint8)
    pil_image = Image.fromarray(arr, mode='RGB')
    return pil_image
