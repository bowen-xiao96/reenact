import torch
import torch.nn as nn
import torch.nn.functional as F


def set_grad_enabled(model, enabled):
    # disable back propagation of a model
    for param in model.parameters(recurse=True):
        param.requires_grad = enabled

    return model
