import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, datasets
from torchvision.utils import save_image

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from torch.autograd import Variable

import numpy as np
from PIL import Image

from loss import PerceptualLoss
from transform_net import make_encoder, DecoderLayer, AdaInstanceNormalization

import logging
import datetime

import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("content_folder", help="path to content dataset")
parser.add_argument("style_folder", help="path to style dataset")
args = parser.parse_args()

print("Content folder:", args.content_folder)
print("Style folder:", args.style_folder)

# Logging setup
start_time = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '_')
logfile_name = "logfile_%s.txt" % start_time
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile_name)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

IMAGE_SIZE = 256
BATCH_SIZE = 4
CONTENT_WEIGHT = 0
STYLE_WEIGHT = 1
MAX_ITER = 100000
lr = 1e-4
CUDA = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

style_transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

content_dataset = datasets.ImageFolder(args.content_folder, transform)
content_train_loader = DataLoader(content_dataset, batch_size=BATCH_SIZE)
content_test_loader = DataLoader(content_dataset, batch_size=BATCH_SIZE)

style_dataset = datasets.ImageFolder(args.style_folder, transform)
style_train_loader = DataLoader(style_dataset, batch_size=1)
style_test_loader = DataLoader(style_dataset, batch_size=1)


enc = make_encoder()
adaIN = AdaInstanceNormalization()
dec = DecoderLayer()
perceptual_loss = PerceptualLoss(enc)

for param in dec.parameters():
    if param.dim() < 2:
        continue
    torch.nn.init.xavier_normal(param.data)

optimizer = Adam(dec.parameters(), lr)

SEED = 1080
torch.manual_seed(SEED)

if CUDA:
    enc.cuda()
    adaIN.cuda()
    dec.cuda()
    perceptual_loss.cuda()
    torch.cuda.manual_seed(SEED)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # kwargs = {'num_workers': 4, 'pin_memory': True}


dec.train()
enc.eval()

logger.debug("Decoder Layer:\n", dec)
logger.debug("Encoder Layer:\n", enc)

scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

def save_debug_image(tensor_orig, tensor_transformed, filename):
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

def train(epoch):
    dec.train()
    for i, (x, _) in enumerate(content_train_loader):
        scheduler.step()
        x =  Variable(x)
        if CUDA:
            x = x.cuda()
        avg_closs = avg_sloss = avg_loss = 0

        for j, (s, _) in enumerate(style_train_loader):
            s = Variable(s)
            if CUDA:
                s = s.cuda()

            optimizer.zero_grad()

            fc, fs = enc(x), enc(s)
            t = adaIN(fc, fs)
            gt = dec(t)
            ft = enc(gt)

            content_loss = F.mse_loss(ft, t, size_average=False)
            style_loss = perceptual_loss(gt, s.expand_as(gt))
            loss = content_loss + 0.01 * style_loss

            avg_closs += content_loss.data.sum() / len(x)
            avg_sloss += style_loss.data.sum() / len(x)
            avg_loss += loss.data.sum() / len(x)

            loss.backward()
            optimizer.step()

        avg_closs /= len(style_train_loader.dataset)
        avg_sloss /= len(style_train_loader.dataset)
        avg_loss /= len(style_train_loader.dataset)

        logger.debug("Train Epoch %d: ITER %d content: %.6f style: %.6f loss: %.6f" %
              (epoch, i, avg_closs, avg_sloss, avg_loss))

        def recover(img):
            '''
            recover from ImageNet normalized rep to real img rep [0, 1]
            '''
            img *= torch.Tensor([0.229, 0.224, 0.225]
                                ).view(1, 3, 1, 1).expand_as(img)
            img += torch.Tensor([0.485, 0.456, 0.406]
                                ).view(1, 3, 1, 1).expand_as(img)

            return img

        stacked = torch.stack(
            [x.data, s.data.expand_as(x.data), gt.data]).view(-1, 3, 256, 256)
        # save_image(recover(x.data), 'origin.png')
        # save_image(stacked, 'debug.png', nrow=8, range=(0.0, 1.0))
        save_debug_image(
            recover(x.data), recover(gt.data), 'debug.png')
        # save_image(recover(s.data), 'style.png')

def test():
    dec.eval()
    avg_closs = avg_sloss = avg_loss = 0
    for i, (x, _) in enumerate(content_test_loader):
        x =  Variable(x)
        if CUDA:
            x = x.cuda()

        for j, (s, _) in enumerate(style_test_loader):
            s = Variable(s)
            if CUDA:
                s = s.cuda()
            
            fc, fs = enc(x), enc(s)
            t = adaIN(fc, fs)
            gt = dec(t)
            ft = enc(gt)

            content_loss = F.mse_loss(ft, t, size_average=False)
            style_loss = perceptual_loss(gt, s.expand_as(gt))
            loss = content_loss + 0.01 * style_loss

            avg_closs += content_loss.data.sum() / len(x)
            avg_sloss += style_loss.data.sum() / len(x)
            avg_loss += loss.data.sum() / len(x)

    avg_closs /= len(style_test_loader.dataset)
    avg_sloss /= len(style_test_loader.dataset)
    avg_loss /= len(style_test_loader.dataset)

    logger.debug('\nTest set: Average content loss: %.4f, Average style loss: %.4f, Average loss: %.4f\n' % (
        avg_closs, avg_sloss, avg_loss))

if __name__ == '__main__':
    for epoch in range(200):
        train(epoch)
        test()
