import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import *


def adaptive_instance_normalize(feature_map, sigma, mu, eps=1e-5):
    n, c, h, w = feature_map.size()

    sigma_n, sigma_c = sigma.size()
    mu_n, mu_c = mu.size()

    assert sigma_n == mu_n
    assert c == sigma_c == mu_c

    # instance normalize the original feature map
    feature_map = feature_map.view(n, c, h * w)
    mean = torch.mean(feature_map, dim=-1, keepdim=True)  # (n, c, 1)
    std = torch.std(feature_map, dim=-1, keepdim=True) + eps  # (n, c, 1)
    feature_map = (feature_map - mean) / std

    # rescale the normalized feature map
    sigma = torch.unsqueeze(sigma, dim=-1)
    mu = torch.unsqueeze(mu, dim=-1)
    feature_map = sigma * feature_map + mu

    feature_map = feature_map.view(n, c, h, w)
    return feature_map


class SPADE(nn.Module):
    # https://arxiv.org/pdf/1903.07291.pdf
    # code reference: https://github.com/NVlabs/SPADE

    def __init__(self, x_channel, d_channel, hidden_channel=128, normalization_type='batch_norm', kernel_size=3):
        super(SPADE, self).__init__()

        self.x_channel = x_channel
        self.d_channel = d_channel
        self.hidden_channel = hidden_channel
        self.normalization_type = normalization_type

        self.norm1 = get_norm_layer(x_channel, normalization_type, affine=False)

        self.conv1 = nn.Conv2d(d_channel, hidden_channel, kernel_size, padding=kernel_size // 2)
        self.conv2_gamma = nn.Conv2d(hidden_channel, x_channel, kernel_size, padding=kernel_size // 2)
        self.conv2_beta = nn.Conv2d(hidden_channel, x_channel, kernel_size, padding=kernel_size // 2)

    def forward(self, x, d):
        xx = self.norm1(x)

        dd = self.conv1(d)
        dd = F.relu(dd, inplace=False)  # shared convolution
        gamma = self.conv2_gamma(dd)  # scaling term
        beta = self.conv2_beta(dd)  # bias term

        output = xx * gamma + beta
        output = output + xx  # residual connection
        return output


class AttentionNormalization(nn.Module):
    # https://arxiv.org/pdf/1906.00884.pdf

    def __init__(self, x_channel, d_channel, hidden_channel, normalization_type='batch_norm', kernel_size=3):
        super(AttentionNormalization, self).__init__()

        self.x_channel = x_channel
        self.d_channel = d_channel
        self.hidden_channel = hidden_channel
        self.normalization_type = normalization_type

        self.norm1 = get_norm_layer(x_channel, normalization_type, affine=False)

        self.conv1 = nn.Conv2d(d_channel, hidden_channel, kernel_size, padding=kernel_size // 2)
        self.conv2_alpha = nn.Conv2d(hidden_channel, x_channel, kernel_size, padding=kernel_size // 2)
        self.conv2_beta = nn.Conv2d(hidden_channel, x_channel, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(x_channel, x_channel, kernel_size, padding=kernel_size // 2)

    def forward(self, x, d):
        xx = self.norm1(x)

        dd = self.conv1(d)
        dd = F.relu(dd, inplace=False)  # shared convolution
        alpha = self.conv2_alpha(dd)  # scaling term
        alpha = torch.sigmoid(alpha)  # limit to [0, 1]
        beta = self.conv2_beta(dd)  # bias term

        output = xx * alpha + beta
        output = F.relu(output, inplace=False)
        output = self.conv3(output)
        output = torch.cat((xx, output), dim=1)
        return output
