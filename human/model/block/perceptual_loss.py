import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import *
from .normalization import adaptive_instance_normalize

# https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
# all batch norm replaced by instance norm


class ResidualBlock(nn.Module):
    # residual block with instance norm

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        # right
        self.conv_r1 = get_conv_layer(in_channel, in_channel, kernel_size, spectral_norm=True)
        self.in1 = get_norm_layer(in_channel, 'instance_norm', affine=True)
        self.conv_r2 = get_conv_layer(in_channel, out_channel, kernel_size, spectral_norm=True)
        self.in2 = get_norm_layer(out_channel, 'instance_norm', affine=True)

    def forward(self, x):
        # left
        left = x

        # right
        right = self.conv_r1(x)
        right = self.in1(right)
        right = F.relu(right, inplace=False)
        right = self.conv_r2(right)
        right = self.in2(right)

        output = left + right
        return output


class ResidualBlockAdaIN(nn.Module):
    # residual block with adaptive instance norm

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(ResidualBlockAdaIN, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        # right
        self.conv_r1 = get_conv_layer(in_channel, in_channel, kernel_size, spectral_norm=True)
        self.conv_r2 = get_conv_layer(in_channel, out_channel, kernel_size, spectral_norm=True)

    def forward(self, x, sigma_1, mu_1, sigma_2, mu_2):
        # left
        left = x

        # right
        right = self.conv_r1(x)
        right = adaptive_instance_normalize(right, sigma_1, mu_1)
        right = F.relu(right, inplace=False)
        right = self.conv_r2(right)
        right = adaptive_instance_normalize(right, sigma_2, mu_2)

        output = left + right
        return output
