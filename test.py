import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import numpy as np

from loss import PerceptualLoss
from transform_net import make_encoder, make_decoder, AdaInstanceNormalization
from utils import save_image, recover_from_ImageNet

import logging
import argparse
import random

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 1
CUDA = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--content_folder", required=True, help="path to content dataset")
parser.add_argument("--style_folder", required=True, help="path to style dataset")
parser.add_argument("--output_folder", help="path to output the style-transferred images")
parser.add_argument("--model_encoder", help="path to the saved encoder model")
parser.add_argument("--model_decoder", required=True, help="path to the saved decoder model")
parser.add_argument("--job_id", help="used to distinguish debug and log file name. If not specified, a random number will be used")
args = parser.parse_args()

# Logging setup
job_id = args.job_id
if not job_id:
    job_id = random.randrange(9999999999)
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Transforms
image_transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Content and style loader
content_test_dataset = datasets.ImageFolder("%s/test" % args.content_folder, image_transform)
content_test_loader = DataLoader(content_test_dataset, batch_size=BATCH_SIZE)

style_test_dataset = datasets.ImageFolder("%s/test" % args.style_folder, image_transform)
style_test_loader = DataLoader(style_test_dataset, batch_size=1)

# Initialize models
enc = make_encoder(model_file=args.model_encoder)
adaIN = AdaInstanceNormalization()
dec = make_decoder(model_file=args.model_decoder)
perceptual_loss = PerceptualLoss(enc)

if CUDA:
    enc.cuda()
    adaIN.cuda()
    dec.cuda()
    perceptual_loss.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

dec.eval()
enc.eval()

def test():
    avg_closs = avg_sloss = avg_loss = 0
    for i, (xx, _) in enumerate(content_test_loader):
        for j, (ss, _) in enumerate(style_test_loader):
            x = Variable(xx)
            s = Variable(ss)
            if CUDA:
                x = x.cuda()
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

            save_image(recover_from_ImageNet(x.data), recover_from_ImageNet(gt.data), '%s/%i_%i.png' % (args.output_folder, i, j))

        avg_closs /= len(style_test_loader.dataset)
        avg_sloss /= len(style_test_loader.dataset)
        avg_loss /= len(style_test_loader.dataset)

        logger.info('\nAverage loss: Image %d, Content: %.4f, Style: %.4f, Total: %.4f\n' % (
            i, avg_closs, avg_sloss, avg_loss))

if __name__ == '__main__':
    logger.info("Content folder:"+ args.content_folder)
    logger.info("Style folder:" + args.style_folder)

    test()

    logger.info("Testing finished and images saved!\n")
