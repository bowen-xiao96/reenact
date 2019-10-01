import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import *


class SelfAttention(nn.Module):
    # https://arxiv.org/pdf/1805.08318.pdf
    # code reference: https://github.com/heykeetae/Self-Attention-GAN

    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        hidden_channel = in_channel // 8
        self.conv_query = get_conv_layer(in_channel, hidden_channel, 1, spectral_norm=False)
        self.conv_key = get_conv_layer(in_channel, hidden_channel, 1, spectral_norm=False)
        self.conv_value = get_conv_layer(in_channel, in_channel, 1, spectral_norm=False)
        self.gamma = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, x):
        n, c, h, w = x.size()

        query = self.conv_query(x)  # n * hidden_channel * h * w
        query = query.view(n, -1, h * w)  # n * hidden_channel * N
        query = query.permute(0, 2, 1)  # n * N * hidden_channel

        key = self.conv_key(x)  # n * hidden_channel * h * w
        key = key.view(n, -1, h * w)  # n * hidden_channel * N

        energy = torch.bmm(query, key)  # n * N * N
        attention = F.softmax(energy, dim=-1)
        attention = attention.permute(0, 2, 1)  # n * N * N

        value = self.conv_value(x)  # n * in_channel * h * w
        value = value.view(n, -1, h * w)  # n * in_channel * N

        output = torch.bmm(value, attention)  # n * in_channel * N
        output = output.view(n, c, h, w)
        output = self.gamma * output + x
        return output
