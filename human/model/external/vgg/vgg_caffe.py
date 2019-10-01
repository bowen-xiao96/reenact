import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

all_networks = ('vgg16', 'vgg19')

network_config = {
    'vgg16': (64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'),
    'vgg19': (64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P')
}

layer_idx = {
    'vgg16': {
        'conv1_1': 1,
        'conv1_2': 3,
        'pool1': 4,
        'conv2_1': 6,
        'conv2_2': 8,
        'pool2': 9,
        'conv3_1': 11,
        'conv3_2': 13,
        'conv3_3': 15,
        'pool3': 16,
        'conv4_1': 18,
        'conv4_2': 20,
        'conv4_3': 22,
        'pool4': 23,
        'conv5_1': 25,
        'conv5_2': 27,
        'conv5_3': 29,
        'pool5': 30,
        'fc6': 33,
        'fc7': 36,
        'fc8': 37
    },

    'vgg19': {
        'conv1_1': 1,
        'conv1_2': 3,
        'pool1': 4,
        'conv2_1': 6,
        'conv2_2': 8,
        'pool2': 9,
        'conv3_1': 11,
        'conv3_2': 13,
        'conv3_3': 15,
        'conv3_4': 17,
        'pool3': 18,
        'conv4_1': 20,
        'conv4_2': 22,
        'conv4_3': 24,
        'conv4_4': 26,
        'pool4': 27,
        'conv5_1': 29,
        'conv5_2': 31,
        'conv5_3': 33,
        'conv5_4': 35,
        'pool5': 36,
        'fc6': 39,
        'fc7': 42,
        'fc8': 43
    }
}


class VGGNetwork(nn.Module):
    # define the Caffe VGG16/VGG19 network

    def __init__(self, network_type, num_classes):
        super(VGGNetwork, self).__init__()

        self.network_type = network_type
        self.num_classes = num_classes

        config = network_config[network_type]
        conv_layers = nn.ModuleList()
        in_channel = 3

        # convolutional and pooling layers
        for c in config:
            if c == 'P':
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_layers.append(nn.Conv2d(in_channel, c, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU(inplace=False))
                in_channel = c

        # linear layers
        linear_layers = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ])

        self.conv_layers = conv_layers
        self.linear_layers = linear_layers

        print(self)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.linear_layers:
            x = layer(x)

        return x


class VGGActivation(nn.Module):
    def __init__(self, vgg_model, extract_layers, normalize, caffe_mean, caffe_std):
        super(VGGActivation, self).__init__()

        self.conv_layers = vgg_model.conv_layers
        self.extract_layers = extract_layers
        self.normalize = normalize

        if caffe_mean is not None:
            self.caffe_mean = nn.Parameter(torch.as_tensor(caffe_mean).view(1, 3, 1, 1))
        else:
            self.register_parameter('caffe_mean', None)

        if caffe_std is not None:
            self.caffe_std = nn.Parameter(torch.as_tensor(caffe_std).view(1, 3, 1, 1))
        else:
            self.register_parameter('caffe_std', None)

    def forward(self, x):
        # x: n * c * h * w, in RGB format
        x = x[:, (2, 1, 0), ...]  # convert to BGR

        if self.normalize:
            # sanity check
            eps = 1e-5
            assert torch.max(x) <= 1 + eps

            x *= 255.0

        if self.caffe_mean is not None:
            x -= self.caffe_mean
        if self.caffe_std is not None:
            x /= self.caffe_std

        # extract intermediate layer output
        feature_maps = list()
        max_index = max(self.extract_layers)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)

            if i in self.extract_layers:
                feature_maps.append(x)
            if i == max_index:
                break

        return feature_maps
