import os
from PIL import Image
import numpy as np

img_path = '/Users/edward/Downloads/val2017/'

imgs = os.listdir(img_path)


def isBlackWhite(fn):
    if fn.split('.')[-1] != 'jpg':
        return False

    with open(img_path + fn, 'rb') as f:
        img = Image.open(f).convert('RGB')

    im = np.array(img).reshape(-1, 3)
    return np.sum(np.var(im, axis=1)) == 0


for fn in imgs:
    if isBlackWhite(fn):
        print(fn)
