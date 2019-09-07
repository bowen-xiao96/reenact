import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm


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
    sigma = torch.unsqueeze(sigma, -1)
    mu = torch.unsqueeze(mu, -1)
    feature_map = sigma * feature_map + mu

    feature_map = feature_map.view(n, c, h, w)
    return feature_map


class ResidualBlockUp(nn.Module):
    # upsampling residual block with adaptive IN
    # used in the upsampling part of G

    def __init__(self, in_channel, out_channel, kernel_size=3, upsample_scale=2, upsample_mode='bilinear'):
        super(ResidualBlockUp, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample_scale = upsample_scale
        self.upsample_mode = upsample_mode

        # left
        self.conv_l = spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

        # right
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2))
        self.conv_r2 = spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))

    def forward(self, x, sigma_1, mu_1, sigma_2, mu_2):
        left = F.interpolate(x, scale_factor=self.upsample_scale, mode=self.upsample_mode)
        left = self.conv_l(left)

        right = adaptive_instance_normalize(x, sigma_1, mu_1)
        right = F.relu(right)
        right = F.interpolate(right, scale_factor=self.upsample_scale, mode=self.upsample_mode)
        right = self.conv_r1(right)
        right = adaptive_instance_normalize(right, sigma_2, mu_2)
        right = F.relu(right)
        right = self.conv_r2(right)

        return left + right


class ResidualBlockDown(nn.Module):
    # downsampling residual block
    # used in the downsampling part of G and E, V

    def __init__(self, in_channel, out_channel, kernel_size=3, pooling_kernel=2):
        super(ResidualBlockDown, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.pooling_kernel = pooling_kernel

        # left
        self.conv_l = spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

        # right
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2))
        self.conv_r2 = spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        left = self.conv_l(x)
        left = F.avg_pool2d(left, self.pooling_kernel)

        right = F.relu(x)
        right = self.conv_r1(right)
        right = F.relu(right)
        right = self.conv_r2(right)
        right = F.avg_pool2d(right, self.pooling_kernel)

        return left + right


class ResidualBlock(nn.Module):
    # residual block with non-adaptive instance normalization
    # used in V

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        # left
        self.conv_1 = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2))
        self.norm_1 = nn.InstanceNorm2d(out_channel, affine=True)
        self.conv_2 = spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))
        self.norm_2 = nn.InstanceNorm2d(out_channel, affine=True)

    def forward(self, x):
        left = self.conv_1(x)
        left = self.norm_1(left)
        left = F.relu(left)
        left = self.conv_2(left)
        left = self.norm_2(left)

        return left + x


class ResidualBlockAdaptive(nn.Module):
    # residual block with adaptive instance normalization
    # used in G

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(ResidualBlockAdaptive, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        # left
        self.conv_1 = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2))
        self.conv_2 = spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))

    def forward(self, x, sigma_1, mu_1, sigma_2, mu_2):
        left = self.conv_1(x)
        left = adaptive_instance_normalize(left, sigma_1, mu_1)
        left = F.relu(left)
        left = self.conv_2(left)
        left = adaptive_instance_normalize(left, sigma_2, mu_2)

        return left + x


class SelfAttention(nn.Module):
    # self-attention block
    # used in the downsampling part of G and E, V
    # adapted from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
