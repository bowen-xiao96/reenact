import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

vgg16_config = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']
vgg19_config = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']

vgg19_mean = [103.939, 116.779, 123.68]  # BGR
vgg19_std = [1.0, 1.0, 1.0]
vggface_mean = [93.5940, 104.7624, 129.1863]  # BGR
vggface_std = [1.0, 1.0, 1.0]

# all after ReLU or dropout
vgg16_layer_idx = {
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
}

vgg19_layer_idx = {
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


def to_caffe_input(x, mean, std, normalize):
    # x: n * c * h * w, in RGB format
    x = x[:, (2, 1, 0), ...]  # convert to BGR

    if normalize:
        # sanity check
        eps = 1e-5
        assert torch.max(x) <= 1 + eps

        x *= 255.0

    x = (x - mean) / std

    return x


class VGGNetwork(nn.Module):
    def __init__(self, type, network_config, num_classes):
        super(VGGNetwork, self).__init__()
        self.type = type

        # replicate the original caffe VGG16/19 model
        conv_layers = nn.ModuleList()
        in_channel = 3

        # convolutional and pooling layers
        for c in network_config:
            if c == 'P':
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_layers.append(nn.Conv2d(in_channel, c, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU(inplace=True))
                in_channel = c

        # linear layers
        linear_layers = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ])

        self.conv_layers = conv_layers
        self.linear_layers = linear_layers

        # print(self)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.linear_layers:
            x = layer(x)

        return x


# used to extract activations of the given layers of a VGG network
class VGGActivation(nn.Module):
    def __init__(self, vgg_model, extract_layer, normalize):
        super(VGGActivation, self).__init__()

        self.conv_layers = vgg_model.conv_layers
        self.extract_layer = extract_layer
        self.normalize = normalize

        if vgg_model.type == 'vgg19':
            self.mean = nn.Parameter(torch.as_tensor(vgg19_mean).view(1, 3, 1, 1))
            self.std = nn.Parameter(torch.as_tensor(vgg19_std).view(1, 3, 1, 1))
        else:
            self.mean = nn.Parameter(torch.as_tensor(vggface_mean).view(1, 3, 1, 1))
            self.std = nn.Parameter(torch.as_tensor(vggface_std).view(1, 3, 1, 1))

    def forward(self, x):
        feature_maps = list()
        max_index = max(self.extract_layer)

        x = to_caffe_input(x, self.mean, self.std, self.normalize)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)

            if i in self.extract_layer:
                feature_maps.append(x)
            if i == max_index:
                break

        return feature_maps
