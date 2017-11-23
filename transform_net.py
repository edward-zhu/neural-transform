import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

from utils import get_mean_var

import os.path

class AdaInstanceNormalization(nn.Module):
    '''
    Implement Adaptive Instance Normalization Layer

    ref: https://arxiv.org/pdf/1703.06868.pdf

    input:  [content feature map (CxHxW), style feature map (CxHxW)]
    output: BxCxHxW
    '''

    def __init__(self, eps=1e-6):
        super(AdaInstanceNormalization, self).__init__()
        self.eps = eps

    def forward(self, c, s):
        c_mean, c_var = get_mean_var(c)
        s_mean, s_var = get_mean_var(s)

        return s_var * (c - c_mean) / (c_var + self.eps) + s_mean


class DecoderLayer(nn.Module):
    '''
    Decoder Layer

    Decode AdaInstanceNorm output to generated image

    ref:
    https://distill.pub/2016/deconv-checkerboard/
    https://github.com/ceshine/fast-neural-style/blob/master/transformer_net.py
    '''

    def __init__(self):
        super(DecoderLayer, self).__init__()
        # num_layer, num_features
        conf = [
            (1, 256),  # 512, 256
            'U',  # 32 > 64
            (3, 256),  # 256, 256
            'U',  # 64 > 128
            (1, 128),  # 256, 128
            (1, 64),  # 128, 64
            'U',  # 128 > 256
            (1, 64),  # 64, 64
            (1, 3)  # 64, 3
        ]
        self.features = self._make_layers(conf)

    def _make_layers(self, conf):
        layers = []
        in_channels = 512
        for block in conf:
            if block == 'U':
                layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
                continue
            n_layer, n_feat = block
            for i in range(0, n_layer):
                layers += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels, n_feat,
                              kernel_size=3, stride=1),
                    nn.ReLU()]
                in_channels = n_feat
        layers.pop()

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class EncoderLayer(nn.Module):
    '''
    EncoderLayer

    part of VGG19 (through relu_4_1)

    ref:
    https://arxiv.org/pdf/1703.06868.pdf (sec. 6)
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''

    def __init__(self, batch_norm):
        super(EncoderLayer, self).__init__()
        conf = models.vgg.cfg['E'][:12]  # VGG through relu_4_1
        self.features = models.vgg.make_layers(conf, batch_norm=batch_norm)

    def forward(self, x):
        return self.features(x)


def make_encoder(model_file, batch_norm=True):
    '''
    make a pretrained partial VGG-19 network
    '''
    VGG_TYPE = 'vgg19_bn' if batch_norm else 'vgg19'

    enc = EncoderLayer(batch_norm)

    if model_file and os.path.isfile(model_file):
        # load weights from pre-saved model file
        enc.load_state_dict(torch.load(model_file))
    else:
        # load weights from pretrained VGG model
        vgg_weights = model_zoo.load_url(models.vgg.model_urls[VGG_TYPE])
        w = {}
        for key in enc.state_dict().keys():
            w[key] = vgg_weights[key]
        enc.load_state_dict(w)
        if not model_file:
            model_file = "encoder.model"
        torch.save(enc.state_dict(), model_file)

    return enc

def make_decoder(model_file):
    '''
    make a pretrained partial VGG-19 network
    '''

    dec = DecoderLayer()

    if model_file and os.path.isfile(model_file):
        # load weights from pre-saved model file
        dec.load_state_dict(torch.load(model_file))
    else:
        raise ValueError('Decoder model is not found!')

    return dec
