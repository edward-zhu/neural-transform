import torch
from torchvision import transforms

import numpy as np
from PIL import Image

def get_mean_var(c):
    n_batch, n_ch, h, w = c.size()

    c_view = c.view(n_batch, n_ch, h * w)
    c_mean = c_view.mean(2)

    c_mean = c_mean.view(n_batch, n_ch, 1, 1).expand_as(c)
    c_var = c_view.var(2)
    c_var = c_var.view(n_batch, n_ch, 1, 1).expand_as(c)
    # c_var = c_var * (h * w - 1) / float(h * w)  # unbiased variance

    return c_mean, c_var

def save_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()

    def recover(t):
        t = t.cpu().numpy()[0].transpose(1, 2, 0) * 255.
        t = t.clip(0, 255).astype(np.uint8)
        return t

    result = Image.fromarray(recover(tensor_transformed))
    orig = Image.fromarray(recover(tensor_orig))
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0, 0))
    new_im.paste(result, (result.size[0] + 5, 0))
    new_im.save(filename)

def recover_from_ImageNet(img):
    '''
    recover from ImageNet normalized rep to real img rep [0, 1]
    '''
    img *= torch.Tensor([0.229, 0.224, 0.225]
                        ).view(1, 3, 1, 1).expand_as(img)
    img += torch.Tensor([0.485, 0.456, 0.406]
                        ).view(1, 3, 1, 1).expand_as(img)

    return img
