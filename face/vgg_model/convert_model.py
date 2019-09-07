import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg_network import *

os.environ['GLOG_minloglevel'] = '2'

CAFFE_PATH = '/home/ephemeroptera/caffe/python'
PROTOTXT = 'VGG_ILSVRC_19_layers_deploy.prototxt'
MODEL = 'VGG_ILSVRC_19_layers.caffemodel'

sys.path = [CAFFE_PATH] + sys.path
import caffe


def initialize_model(pytorch_model, caffe_net, vgg_layer_index):
    conv_layer_count = len(pytorch_model.conv_layers)

    # transfer weight layer by layer
    for layer_name, idx in vgg_layer_index.items():
        if layer_name.startswith('conv'):
            # convolutional layer
            params = caffe_net.params[layer_name]
            idx -= 1  # ignore the relu layer

            assert pytorch_model.conv_layers[idx].weight.data.numpy().shape == params[0].data.shape
            pytorch_model.conv_layers[idx].weight.data = torch.from_numpy(params[0].data)

            assert pytorch_model.conv_layers[idx].bias.data.numpy().shape == params[1].data.shape
            pytorch_model.conv_layers[idx].bias.data = torch.from_numpy(params[1].data)

        if layer_name.startswith('fc'):
            # linear layer
            params = caffe_net.params[layer_name]
            idx -= conv_layer_count
            if layer_name != 'fc8':  # if not the last layer
                idx -= 2  # ignore the relu and dropout layers

            assert pytorch_model.linear_layers[idx].weight.data.numpy().shape == params[0].data.shape
            pytorch_model.linear_layers[idx].weight.data = torch.from_numpy(params[0].data)

            assert pytorch_model.linear_layers[idx].bias.data.numpy().shape == params[1].data.shape
            pytorch_model.linear_layers[idx].bias.data = torch.from_numpy(params[1].data)

    return pytorch_model


if __name__ == '__main__':
    caffe.set_mode_cpu()
    net = caffe.Net(PROTOTXT, MODEL, caffe.TEST)

    # load caffe weights into the pytorch VGG model
    model = VGGNetwork('vgg19', vgg19_config, 1000)
    # model = VGGNetwork('vgg16', vgg16_config, 2622)  # for VGGFace
    model = model.cpu()
    model = model.eval()

    # transfer weight
    model = initialize_model(model, net, vgg19_layer_idx)
    # model = initialize_model(model, net, vgg16_layer_idx)  # for VGGFace

    # test
    input_data = torch.randn(2, 3, 224, 224).float()

    net.blobs['data'].reshape(2, 3, 224, 224)
    net.blobs['data'].data[...] = input_data.data.numpy()
    net.forward()
    caffe_result = net.blobs['prob'].data

    pytorch_result = F.softmax(model(input_data))
    pytorch_result = pytorch_result.data.numpy()

    diff = caffe_result - pytorch_result
    print(diff)
    print(np.max(np.abs(diff)))

    # save model
    torch.save(model.state_dict(), 'vgg19_caffe.pth')
    # torch.save(model.state_dict(), 'vggface_caffe.pth')  # for VGGFace
