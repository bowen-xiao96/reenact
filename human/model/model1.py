import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block.biggan import ResidualBlockUp, ResidualBlockDown, ResidualBlockDown3D
from .block.perceptual_loss import ResidualBlock, ResidualBlockAdaIN
from .block.sagan import SelfAttention
from .loss.gan_loss import loss_fm, loss_adv
from .external.vgg.vgg_caffe import VGGActivation, VGGNetwork
from .utils import set_grad_enabled

# config of the network strcture
input_size = 256
input_modality = 'skeleton_2d'
depth_voxel_count = 10
input_normalize = True
score_normalize = False

# dimension of the embedding space
embedding_dim = 512

G_config = [
    ('D', 64, 9, 2),  # 128
    ('I',),
    ('D', 128, 3, 2),  # 64
    ('I',),
    ('D', 256, 3, 2),  # 32
    ('I',),
    ('A',),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('I',),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('U', 256, 3, 2),  # 32
    ('U', 128, 3, 2),  # 64
    ('A', 8),  # operate at 64 x 64
    ('U', 64, 3, 2),  # 128
    ('U', 3, 3, 2),  # 256
]

# for embedder, do not use normalization layers
E_config = [
    ('D', 64, 3, 2),  # 128
    ('D', 128, 3, 2),  # 64
    ('D', 256, 3, 2),  # 32
    ('A',),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('D', 512, 3, 2),  # 8
    ('D', 512, 3, 2),  # 4 x 4 spatial resolution
]

# for discriminator, do not use normalization layers
V_config = [
    ('D', 64, 3, 2),  # 128
    ('D', 128, 3, 2),  # 64
    ('D', 256, 3, 2),  # 32
    ('A',),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('D', 512, 3, 2),  # 8
    ('D', 512, 3, 2),  # 4
    ('B', 512, 3)  # 4 x 4 spatial resolution
]

upsample_mode = 'bilinear'

# layers of the VGG19 and VGGFace networks used to compute content loss
vgg19_layers = (0, 5, 10, 19, 28)  # all before relu activation

# weight of each loss
loss_weight = {
    'content_loss': 10,
    'vgg19_loss': 1e-2,
    'adv_loss': 1e0,
    'fm_loss': 1e1,
    'mch_loss': 8e1
}

# vgg mean and std
vgg19_mean = [103.939, 116.779, 123.68]  # BGR
vgg19_std = [1.0, 1.0, 1.0]

# weight file path
vgg19_weight_file = '/userhome/35/rnchen2/human36m/vgg19_caffe.pth'


def _make_layers(layer_config, input_dim, input_depth):
    layer_list = nn.ModuleList()
    in_channel = input_dim

    for i, c in enumerate(layer_config):
        if c[0] == 'D':  # ResidualBlockDown
            out_channel, kernel_size, pooling_kernel = c[1:]
            if i == 0 and input_depth != 1:
                # 3d input
                layer = ResidualBlockDown3D(in_channel, out_channel, input_depth, kernel_size, pooling_kernel)
            else:
                # 2d input
                layer = ResidualBlockDown(in_channel, out_channel, kernel_size, pooling_kernel)
            in_channel = out_channel

        elif c[0] == 'A':  # SelfAttention
            layer = SelfAttention(in_channel)

        elif c[0] == 'B':  # ResidualBlock
            out_channel, kernel_size = c[1:]
            layer = ResidualBlock(in_channel, out_channel, kernel_size)
            in_channel = out_channel

        elif c[0] == 'BA':  # ResidualBlockAdaIN
            out_channel, kernel_size = c[1:]
            layer = ResidualBlockAdaIN(in_channel, out_channel, kernel_size)
            in_channel = out_channel

        elif c[0] == 'U':  # ResidualBlockUp
            out_channel, kernel_size, upsample_scale = c[1:]
            layer = ResidualBlockUp(in_channel, out_channel, kernel_size, upsample_scale, upsample_mode)
            in_channel = out_channel

        elif c[0] == 'I':
            layer = nn.InstanceNorm2d(in_channel, affine=True)

        else:
            raise ValueError('Unknown layer type ' + c[0])

        layer_list.append(layer)

    return layer_list


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        if input_modality == 'skeleton_2d':
            input_dim = 3
            input_depth = 1
        elif input_modality == 'skeleton_rgbd':
            input_dim = 4
            input_depth = 1
        elif input_modality == 'skeleton_3d':
            input_dim = 3
            input_depth = depth_voxel_count
        else:
            raise NotImplementedError()

        self.layers = _make_layers(G_config, input_dim, input_depth)

        # calculate AdaIN slice
        in_channel = input_dim
        param_count = list()

        for c in G_config:
            if c[0] == 'D':  # ResidualBlockDown
                out_channel, kernel_size, downsample_scale = c[1:]
                in_channel = out_channel

            elif c[0] == 'BA' or c[0] == 'U':  # ResidualBlockAdaptive, ResidualBlockUp
                out_channel, kernel_size = c[1:3]
                param_count.append((in_channel, in_channel))
                in_channel = out_channel

            elif c[0] == 'I' or c[0] == 'B':
                continue

        self.adain_slice = param_count

        # calculate the overall number of adain parameters
        self.adain_param_count = np.sum(param_count)

        # projection matrix
        # note that we need both sigma and mu
        self.P = nn.Linear(embedding_dim, 2 * self.adain_param_count, bias=False)

    def forward(self, y, e):
        e_mean = torch.mean(e, dim=1, keepdim=False)
        adain = self.P(e_mean)
        adain = adain.view(adain.size(0), self.adain_param_count, 2)  # n * adain_param_count * 2

        adain_slice_pos = 0
        adain_start_pos = 0

        for layer in self.layers:
            if isinstance(layer, (ResidualBlockAdaIN, ResidualBlockUp)):
                dim_1, dim_2 = self.adain_slice[adain_slice_pos]

                # 0: (sigma, mu) * (...first layer (dim_1), second layer (dim_2)...)
                slice_1 = adain[:, adain_start_pos: adain_start_pos + dim_1, :]
                slice_2 = adain[:, adain_start_pos + dim_1: adain_start_pos + dim_1 + dim_2, :]

                sigma_1 = slice_1[:, :, 0].contiguous()
                mu_1 = slice_1[:, :, 1].contiguous()
                sigma_2 = slice_2[:, :, 0].contiguous()
                mu_2 = slice_2[:, :, 1].contiguous()

                y = layer(y, sigma_1, mu_1, sigma_2, mu_2)
                adain_slice_pos += 1
                adain_start_pos += (dim_1 + dim_2)

            else:
                y = layer(y)

        # sanity check
        assert adain_start_pos == self.adain_param_count

        y = torch.sigmoid(y)

        if not input_normalize:
            y *= 255.0

        return y


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()

        if input_modality == 'skeleton_2d':
            input_dim = 6
            input_depth = 1
        elif input_modality == 'skeleton_rgbd':
            input_dim = 8
            input_depth = 1
        elif input_modality == 'skeleton_3d':
            input_dim = 3
            input_depth = depth_voxel_count + 1
        else:
            raise NotImplementedError()

        self.layers = _make_layers(E_config, input_dim, input_depth)

    def forward(self, x, y):
        if input_modality == 'skeleton_3d':
            # x: n * T * c * h * w
            # y: n * T * c * d * h * w
            x = torch.unsqueeze(x, dim=3)
            xx = torch.cat((x, y), dim=3)

            # (n * T) * c * (d + 1) * h * w
            n, T, c, d, h, w = xx.size()
            xx = xx.view(-1, c, d, h, w)
        else:
            # x: n * T * c * h * w
            # y: n * T * c * h * s
            xx = torch.cat((x, y), dim=2)

            # (n * T) * (2 * c) * h * w
            n, T, c, h, w = xx.size()
            xx = xx.view(-1, c, h, w)

        for layer in self.layers:
            xx = layer(xx)

        # sum pooling and relu
        w = xx.size(-1)

        # sanity check
        assert w == 4

        xx = F.adaptive_max_pool2d(xx, 1)
        xx = xx.view(n, T, embedding_dim)  # n * T * embedding_dim
        xx = F.relu(xx)

        return xx


class Discriminator(nn.Module):
    def __init__(self, sample_count):
        super(Discriminator, self).__init__()

        if input_modality == 'skeleton_2d':
            input_dim = 6
            input_depth = 1
        elif input_modality == 'skeleton_rgbd':
            input_dim = 8
            input_depth = 1
        elif input_modality == 'skeleton_3d':
            input_dim = 3
            input_depth = depth_voxel_count + 1
        else:
            raise NotImplementedError()

        self.layers = _make_layers(V_config, input_dim, input_depth)

        # projection discriminator
        self.embedding = nn.Embedding(sample_count, embedding_dim)

        self.w0 = nn.Parameter(torch.Tensor(1, embedding_dim), requires_grad=True)
        self.b = nn.Parameter(torch.as_tensor(0.0), requires_grad=True)

    def forward(self, x, y, sample_idx):
        if input_modality == 'skeleton_3d':
            # x: n * c * h * w
            # y: n * c * d * h * w
            x = torch.unsqueeze(x, dim=2)
            xx = torch.cat((x, y), dim=2)
        else:
            # x: n * c * h * w
            # y: n * c * h * w
            xx = torch.cat((x, y), dim=1)

        # the output of each residual block and the final score
        features = list()

        for layer in self.layers:
            xx = layer(xx)

            if isinstance(layer, (ResidualBlockDown, ResidualBlock)):
                features.append(xx)

        # sum pooling and relu
        w = xx.size(-1)

        # sanity check
        assert w == 4

        xx = F.adaptive_max_pool2d(xx, 1)
        xx = xx.view(xx.size(0), embedding_dim)  # n * embedding_dim
        xx = F.relu(xx)

        # look up
        ww = self.embedding(sample_idx)  # embedding vectors
        www = ww + self.w0  # n * embedding_dim
        scores = torch.sum(xx * www, dim=1, keepdim=False)
        scores = scores + self.b

        if score_normalize:
            scores = torch.tanh(scores)

        return {'d_scores': scores, 'd_features': features, 'd_w': ww}


class Loss_DSC(nn.Module):
    def __init__(self):
        super(Loss_DSC, self).__init__()

    def forward(self, d_out, d_out_hat):
        D_score = d_out['d_scores']
        D_score_hat = d_out_hat['d_scores']

        loss = F.relu(1.0 - D_score)
        loss_hat = F.relu(1.0 + D_score_hat)

        loss_all = torch.mean(loss + loss_hat)  # average over batch size
        return {'dsc_loss': loss_all}


class Loss_EG(nn.Module):
    def __init__(self):
        super(Loss_EG, self).__init__()

        vgg19_model = VGGNetwork('vgg19', 1000)
        vgg19_model.load_state_dict(torch.load(vgg19_weight_file))
        self.vgg19_activation = VGGActivation(vgg19_model, vgg19_layers, input_normalize, vgg19_mean, vgg19_std)

    def forward(self, x, x_hat, e, d_out, d_out_hat):
        d_features = d_out['d_features']
        w = d_out['d_w']
        d_scores_hat = d_out_hat['d_scores']
        d_features_hat = d_out_hat['d_features']

        # content loss
        content_loss = F.l1_loss(x_hat, x)

        # vgg19 loss
        vgg19_feat = self.vgg19_activation(x)
        vgg19_feat_hat = self.vgg19_activation(x_hat)
        vgg19_loss = loss_fm(vgg19_feat, vgg19_feat_hat)

        # adversarial loss
        adv_loss = loss_adv(d_scores_hat)

        # feature matching loss
        fm_loss = loss_fm(d_features, d_features_hat)

        # matching loss
        w = torch.unsqueeze(w, 1).expand(e.size())
        mch_loss = F.l1_loss(w, e)

        loss_all = content_loss * loss_weight['content_loss'] + \
                   vgg19_loss * loss_weight['vgg19_loss'] + \
                   adv_loss * loss_weight['adv_loss'] + \
                   fm_loss * loss_weight['fm_loss'] + \
                   mch_loss * loss_weight['mch_loss']

        return {'1_content_loss': content_loss, '2_vgg19_loss': vgg19_loss, '3_adv_loss': adv_loss,
                '4_fm_loss': fm_loss, '5_mch_loss': mch_loss, 'eg_loss': loss_all}


def initialize(G, E, D, L_EG, L_DSC):
    set_grad_enabled(L_EG, False)
    set_grad_enabled(L_DSC, False)
    param_count_all = list()

    for module in (G, E, D):
        if isinstance(module, Discriminator):
            module.embedding.weight.data.uniform_()
            module.w0.data.normal_()
            module.b.data.normal_()

        elif isinstance(module, Generator):
            module.P.weight.data.normal_(0.0, 0.02)

        param_count = sum([p.data.nelement() for p in module.parameters(recurse=True)])
        param_count_all.append(param_count)

    return param_count_all
