import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from model.components import *
from model.network_utils import *
from vgg_model.vgg_network import *


class Generator(nn.Module):
    def __init__(self, G_config, normalize, input_dim=3):
        super(Generator, self).__init__()

        self.normalize = normalize

        self.layers = make_layers(G_config, input_dim)
        self.adain_slice = calculate_adain_slice(G_config, input_dim)

        # calculate the overall number of adain parameters
        self.adain_param_count = np.sum(self.adain_slice)

        self.adain = nn.Parameter(torch.Tensor(1, self.adain_param_count, 2), requires_grad=True)

    def forward(self, x):
        # x: n * input_dim * input_size * input_size

        adain_slice_pos = 0
        adain_start_pos = 0

        for layer in self.layers:
            if isinstance(layer, (ResidualBlockAdaptive, ResidualBlockUp)):
                dim_1, dim_2 = self.adain_slice[adain_slice_pos]

                # 0: (sigma, mu) * (...first layer (dim_1), second layer (dim_2)...)
                slice_1 = self.adain[:, adain_start_pos: adain_start_pos + dim_1, :]
                slice_2 = self.adain[:, adain_start_pos + dim_1: adain_start_pos + dim_1 + dim_2, :]

                sigma_1 = slice_1[:, :, 0].contiguous()
                mu_1 = slice_1[:, :, 1].contiguous()
                sigma_2 = slice_2[:, :, 0].contiguous()
                mu_2 = slice_2[:, :, 1].contiguous()

                x = layer(x, sigma_1, mu_1, sigma_2, mu_2)
                adain_slice_pos += 1
                adain_start_pos += (dim_1 + dim_2)

            else:
                x = layer(x)

        # sanity check
        assert adain_start_pos == self.adain_param_count

        x = torch.sigmoid(x)

        if not self.normalize:
            x *= 255.0

        return x


class Embedder(nn.Module):
    def __init__(self, E_config, embedding_dim, input_dim=6):
        super(Embedder, self).__init__()

        self.embedding_dim = embedding_dim
        self.layers = make_layers(E_config, input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # sum pooling and relu
        w = x.size(-1)

        # sanity check
        assert w == 4

        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), self.embedding_dim)  # n * embedding_dim
        x = F.relu(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, V_config, embedding_dim, input_dim=6):
        super(Discriminator, self).__init__()

        self.embedding_dim = embedding_dim
        self.layers = make_layers(V_config, input_dim)

        self.w = nn.Parameter(torch.Tensor(1, embedding_dim), requires_grad=True)
        self.b = nn.Parameter(torch.as_tensor(0.0), requires_grad=True)

    def forward(self, x):
        # the output of each residual block and the final score
        output = list()

        for layer in self.layers:
            x = layer(x)

            if isinstance(layer, (ResidualBlockDown, ResidualBlock)):
                output.append(x)

        # sum pooling and relu
        w = x.size(-1)

        # sanity check
        assert w == 4

        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), self.embedding_dim)  # n * embedding_dim
        x = F.relu(x)

        scores = torch.sum(x * self.w, dim=1, keepdim=False)
        scores = scores + self.b
        # scores = torch.tanh(scores)

        output.append(scores)
        return output
