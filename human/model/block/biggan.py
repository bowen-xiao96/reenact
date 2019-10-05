import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import *
from .normalization import adaptive_instance_normalize

# https://arxiv.org/pdf/1809.11096.pdf
# all batch norm replaced by adaptive instance norm


class ResidualBlockUp(nn.Module):
    # upsampling residual block

    def __init__(self, in_channel, out_channel, kernel_size=3, upsample_scale=2, upsample_mode='bilinear'):
        super(ResidualBlockUp, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample_scale = upsample_scale
        self.upsample_mode = upsample_mode

        # left
        self.conv_l1 = get_conv_layer(in_channel, out_channel, 1, spectral_norm=True)

        # right
        self.conv_r1 = get_conv_layer(in_channel, in_channel, kernel_size, spectral_norm=True)
        self.conv_r2 = get_conv_layer(in_channel, out_channel, kernel_size, spectral_norm=True)

    def forward(self, x, sigma_1, mu_1, sigma_2, mu_2):
        # left
        left = F.interpolate(x, scale_factor=self.upsample_scale, mode=self.upsample_mode)
        left = self.conv_l1(left)

        # right
        right = adaptive_instance_normalize(x, sigma_1, mu_1)
        right = F.relu(right, inplace=False)
        right = F.interpolate(right, scale_factor=self.upsample_scale, mode=self.upsample_mode)
        right = self.conv_r1(right)
        right = adaptive_instance_normalize(right, sigma_2, mu_2)
        right = F.relu(right, inplace=False)
        right = self.conv_r2(right)

        output = left + right
        return output


class ResidualBlockDown(nn.Module):
    # downsampling residual block

    def __init__(self, in_channel, out_channel, kernel_size=3, pooling_kernel=2):
        super(ResidualBlockDown, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.pooling_kernel = pooling_kernel

        # left
        self.conv_l1 = get_conv_layer(in_channel, out_channel, 1, spectral_norm=True)

        # right
        self.conv_r1 = get_conv_layer(in_channel, in_channel, kernel_size, spectral_norm=True)
        self.conv_r2 = get_conv_layer(in_channel, out_channel, kernel_size, spectral_norm=True)

    def forward(self, x):
        # left
        left = self.conv_l1(x)
        left = F.avg_pool2d(left, self.pooling_kernel)

        # right
        right = F.relu(x, inplace=False)
        right = self.conv_r1(right)
        right = F.relu(right, inplace=False)
        right = self.conv_r2(right)
        right = F.avg_pool2d(right, self.pooling_kernel)

        output = left + right
        return output
