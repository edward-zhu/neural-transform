import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, datasets

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from torch.autograd import Variable

import numpy as np
from PIL import Image

from loss import PerceptualLoss
from transform_net import make_encoder, DecoderLayer, AdaInstanceNormalization

from utils import save_image, recover_from_ImageNet
import logging
import argparse
import random


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epoch",
                    default=100, help="epoch")
parser.add_argument("--batch_size",
                    default=8, help="batch size")
parser.add_argument("--content_folder", required=True,
                    help="path to content dataset")
parser.add_argument("--style_folder", required=True,
                    help="path to style dataset")
parser.add_argument("--model_encoder", help="path to the saved encoder model")
parser.add_argument(
    "--job_id", help="used to distinguish debug and log file name. If not specified, a random number will be used")
args = parser.parse_args()

# Constants
EPOCH = args.epoch
IMAGE_SIZE = 256
BATCH_SIZE = args.batch_size
lr = 1e-4
CUDA = torch.cuda.is_available()
TORCH_SEED = 1080

SAVE_DEBUG_IMAGE_PER_NBATCH = 10
SAVE_MODEL_PER_NEPOCH = 1
VALIDATE_PER_NBATCH = 10


# Logging setup
# start_time = str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '-')
job_id = args.job_id
if not job_id:
    job_id = random.randrange(9999999999)
# logfile_name = "logfile_%s.txt" % job_id
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
# fh = logging.FileHandler(logfile_name)
# fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# logger.addHandler(fh)
logger.addHandler(ch)

# Transforms
image_transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

style_transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image_transform_nocrop = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Content and style loader
content_train_dataset = datasets.ImageFolder(
    "%s/train" % args.content_folder, image_transform)
content_train_loader = DataLoader(
    content_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
content_validation_dataset = datasets.ImageFolder(
    "%s/val" % args.content_folder, image_transform)
content_validation_loader = DataLoader(
    content_validation_dataset, batch_size=BATCH_SIZE)

style_train_dataset = datasets.ImageFolder(
    "%s/train" % args.style_folder, image_transform)
style_train_loader = DataLoader(
    style_train_dataset, batch_size=1, shuffle=True)
style_validation_dataset = datasets.ImageFolder(
    "%s/val" % args.style_folder, image_transform)
style_validation_loader = DataLoader(style_validation_dataset, batch_size=1)

# Initialize models
enc = make_encoder(model_file=args.model_encoder)
adaIN = AdaInstanceNormalization()
dec = DecoderLayer()
perceptual_loss = PerceptualLoss(enc)

for param in dec.parameters():
    if param.dim() < 2:
        continue
    torch.nn.init.kaiming_normal(param.data)

optimizer = Adam(dec.parameters(), lr)

torch.manual_seed(TORCH_SEED)

if CUDA:
    enc.cuda()
    adaIN.cuda()
    dec.cuda()
    perceptual_loss.cuda()
    torch.cuda.manual_seed(TORCH_SEED)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # kwargs = {'num_workers': 4, 'pin_memory': True}

dec.train()
enc.eval()

scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)


def train(epoch):
    dec.train()

    loader = zip(content_train_loader, style_train_loader)

    avg_closs = avg_sloss = avg_loss = 0

    n_samples = 0

    for i, ((xx, _), (ss, _)) in enumerate(loader):
        scheduler.step()

        x = Variable(xx)
        s = Variable(ss)
        if CUDA:
            x = x.cuda()
            s = s.cuda()

        optimizer.zero_grad()

        fc, fs = enc(x), enc(s)
        t = adaIN(fc, fs)
        gt = dec(t)
        ft = enc(gt)

        content_loss = F.mse_loss(ft, t, size_average=False)
        style_loss = perceptual_loss(gt, s.expand_as(gt))
        loss = content_loss + 0.1 * style_loss

        batch_closs, batch_sloss, batch_loss = content_loss.data.sum(
        ), style_loss.data.sum(), loss.data.sum()

        avg_closs += batch_closs
        avg_sloss += batch_sloss
        avg_loss += batch_loss

        n_samples += len(xx)

        loss.backward()
        optimizer.step()

        if i % SAVE_DEBUG_IMAGE_PER_NBATCH == 0:
            logger.info("Train Epoch %d: Batch %d content: %.6f style: %.6f loss: %.6f" %
                        (epoch, i, batch_closs / len(xx), batch_sloss / len(xx), batch_loss / len(xx)))
            save_image(recover_from_ImageNet(x.data), recover_from_ImageNet(
                gt.data), recover_from_ImageNet(s.data), 'debug_train_%d_%d.png' % (epoch, i, ))

    avg_closs /= n_samples
    avg_sloss /= n_samples
    avg_loss /= n_samples

    logger.info("Train Epoch %d: Overall content: %.6f style: %.6f loss: %.6f" %
                (epoch, avg_closs, avg_sloss, avg_loss))

    # stacked = torch.stack(
    #     [x.data, s.data.expand_as(x.data), gt.data]).view(-1, 3, 256, 256)
    # save_image(recover(x.data), 'origin.png')
    # save_image(stacked, 'debug.png', nrow=8, range=(0.0, 1.0))
    # save_image(recover(s.data), 'style.png')


def validation():
    dec.eval()
    avg_closs = avg_sloss = avg_loss = 0
    n_samples = 0
    for i, (xx, _) in enumerate(content_validation_loader):
        for j, (ss, _) in enumerate(style_validation_loader):
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

            avg_closs += content_loss.data.sum()
            avg_sloss += style_loss.data.sum()
            avg_loss += loss.data.sum()

            n_samples += len(ss)

            if i % 5 == 0 and j == 0:
                save_image(recover_from_ImageNet(x.data), recover_from_ImageNet(
                    gt.data), recover_from_ImageNet(s.data), 'debug_val_%d_%d.png' % (epoch, i, ))

    avg_closs /= n_samples
    avg_sloss /= n_samples
    avg_loss /= n_samples

    logger.info('Validation - Average loss: Content: %.4f, Style: %.4f, Total: %.4f\n' % (
        avg_closs, avg_sloss, avg_loss))


if __name__ == '__main__':
    logger.info("Content folder:" + args.content_folder)
    logger.info("Style folder:" + args.style_folder)

    logger.debug("Decoder Layer:\n" + str(dec))
    logger.debug("Encoder Layer:\n" + str(enc))

    for epoch in range(EPOCH):
        train(epoch)
        # Save the trained decoder model
        if epoch % SAVE_MODEL_PER_NEPOCH == 0:
            torch.save(dec.state_dict(), "decoder_%d_%d.model" %
                       (job_id, epoch,))

    logger.info("Training finished.\n")
