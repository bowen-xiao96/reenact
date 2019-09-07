import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import config
import model.network  # to avoid circular import
from model.components import *


def calculate_adain_slice(G_config, input_dim):
    param_count = list()
    in_channel = input_dim

    for c in G_config:
        if c[0] == 'D':  # ResidualBlockDown
            out_channel, kernel_size, downsample_scale = c[1:]
            in_channel = out_channel

        elif c[0] == 'BA':  # ResidualBlockAdaptive
            out_channel, kernel_size = c[1:]
            param_count.append((out_channel, out_channel))
            in_channel = out_channel

        elif c[0] == 'U':  # ResidualBlockUp
            out_channel, kernel_size, upsample_scale = c[1:]
            param_count.append((in_channel, out_channel))
            in_channel = out_channel

        elif c[0] == 'I' or c[0] == 'B':
            continue

    return param_count


def make_layers(layer_config, input_dim):
    layer_list = nn.ModuleList()
    in_channel = input_dim

    for c in layer_config:
        if c[0] == 'D':  # ResidualBlockDown
            out_channel, kernel_size, pooling_kernel = c[1:]
            layer = ResidualBlockDown(in_channel, out_channel, kernel_size, pooling_kernel)
            in_channel = out_channel

        elif c[0] == 'A':  # SelfAttention
            layer = SelfAttention(in_channel)

        elif c[0] == 'B':  # ResidualBlock
            out_channel, kernel_size = c[1:]
            layer = ResidualBlock(in_channel, out_channel, kernel_size)
            in_channel = out_channel

        elif c[0] == 'BA':  # ResidualBlockAdaptive
            out_channel, kernel_size = c[1:]
            layer = ResidualBlockAdaptive(in_channel, out_channel, kernel_size)
            in_channel = out_channel

        elif c[0] == 'U':  # ResidualBlockUp
            out_channel, kernel_size, upsample_scale = c[1:]
            layer = ResidualBlockUp(in_channel, out_channel, kernel_size, upsample_scale, config.upsample_mode)
            in_channel = out_channel

        elif c[0] == 'I':
            layer = nn.InstanceNorm2d(in_channel, affine=True)

        else:
            raise ValueError('Unknown layer type ' + c[0])

        layer_list.append(layer)

    return layer_list


def set_grad_enabled(model, enabled=True):
    # disable back propagation of a model
    for param in model.parameters(recurse=True):
        param.requires_grad = enabled

    return model


def initialize_param(module):
    if isinstance(module, model.network.Discriminator):
        module.embedding.weight.data.uniform_()
        module.w0.data.normal_()
        module.b.data.normal_()

    elif isinstance(module, model.network.Generator):
        module.P.weight.data.normal_(0.0, 0.02)

    param_count = sum([p.data.nelement() for p in module.parameters(recurse=True)])
    return param_count
