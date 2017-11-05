import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms, datasets
from torchvision.utils import save_image

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from torch.autograd import Variable

from loss import PerceptualLoss
from transform_net import make_encoder, DecoderLayer, AdaInstanceNormalization


IMAGE_SIZE = 256
BATCH_SIZE = 4
# DATASET = "/data/jz2653/cv/c1"
DATASET = "./images"
STYLE_IMAGES = "./styles"
CONTENT_WEIGHT = 0
STYLE_WEIGHT = 1
MAX_ITER = 100000
LOG_INT = 10
lr = 1e-3
CUDA = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_dataset = datasets.ImageFolder(DATASET, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

style_dataset = datasets.ImageFolder(STYLE_IMAGES, transform)
style_loader = DataLoader(style_dataset, shuffle=True)

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

print(dec)
print(enc)

scheduler = StepLR(optimizer, step_size=100, gamma=0.99)


def train():
    for epoch in range(10000):
        for i, (x, _) in enumerate(train_loader):
            scheduler.step()

            s, _ = next(iter(style_loader))
            s = Variable(s)
            x = Variable(x)

            if CUDA:
                x, s = x.cuda(), s.cuda()

            optimizer.zero_grad()

            fc, fs = enc(x), enc(s)

            t = Variable(adaIN(fc, fs).data, requires_grad=False)

            # save_image(t.data.view(512, 1, 32, 32), 'debug.png')

            gt = dec(t)

            ft = enc(gt)

            content_loss = F.mse_loss(ft, t, size_average=False)
            style_loss = perceptual_loss(gt, s.expand_as(gt))

            loss = content_loss + 0.1 * style_loss

            loss.backward()
            optimizer.step()

            agg_closs = content_loss.data.sum() / len(x)
            agg_sloss = style_loss.data.sum() / len(x)
            agg_loss = loss.data.sum() / len(x)

            if i % LOG_INT == 0:
                print("ITER %d content: %.6f style: %.6f loss: %.6f" %
                      (i, agg_closs / LOG_INT, agg_sloss / LOG_INT, agg_loss / LOG_INT))

                dec.eval()

                def recover(img):
                    '''
                    recover from ImageNet normalized rep to real img rep [0, 1]
                    '''
                    img *= torch.Tensor([0.229, 0.224, 0.225]
                                        ).view(1, 3, 1, 1).expand_as(img)
                    img += torch.Tensor([0.485, 0.456, 0.406]
                                        ).view(1, 3, 1, 1).expand_as(img)

                    return img

                save_image(recover(gt.data), 'debug.png')
                save_image(recover(s.data), 'style.png')

                dec.train()


train()
