import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils import get_mean_var


class PerceptualLoss(nn.Module):
    '''
    Implement Perceptual Loss in a VGG network

    ref:
    https://github.com/ceshine/fast-neural-style/blob/master/style-transfer.ipynb
    https://arxiv.org/abs/1603.08155

    input: BxCxHxW, BxCxHxW
    output: loss type Variable
    '''

    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.features

        # use relu_1_1, 2_1, 3_1, 4_1
        self.use_layer = set(['2', '9', '16', '29'])

    def forward(self, g, s):
        loss = 0

        for name, module in self.vgg_layers._modules.items():
            g, s = module(g), module(s)
            if name in self.use_layer:
                g_mean, g_var = get_mean_var(g)
                s_mean, s_var = get_mean_var(s)
                s_mean = Variable(s_mean.data, requires_grad=False)
                s_var = Variable(s_var.data, requires_grad=False)
                loss += F.mse_loss(g_mean, s_mean, size_average=False) + \
                        F.mse_loss(g_var, s_var, size_average=False)
        return loss
