import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_layer(in_channel, out_channel, kernel_size, spectral_norm=True):
    conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)

    if spectral_norm:
        conv_layer = torch.nn.utils.spectral_norm(conv_layer)

    return conv_layer


def get_norm_layer(channel, norm_type, affine=True):
    if norm_type == 'batch_norm':
        norm = nn.BatchNorm2d(channel, affine=affine)
    elif norm_type == 'instance_norm':
        norm = nn.InstanceNorm2d(channel, affine=affine)
    else:
        raise NotImplementedError()

    return norm
