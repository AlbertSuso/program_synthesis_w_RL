import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

class DiscriminatorBlock(nn.Module):
    '''ResNet Block'''

    def __init__(self, in_size, out_size, stride=1):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = spectral_norm(nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = spectral_norm(nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn3 = nn.BatchNorm2d(out_size)
        self.conv_res = spectral_norm(nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn_res = nn.BatchNorm2d(out_size)

    def forward(self, x, maxpool=True):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.bn_res(self.conv_res(residual))
        if maxpool:
            x = F.max_pool2d(F.leaky_relu(x), 2)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_chanels=2, emb_size=512):
        super(Discriminator, self).__init__()

        self.block1 = DiscriminatorBlock(in_chanels, 64)
        self.block2 = DiscriminatorBlock(64, 128)
        self.block3 = DiscriminatorBlock(128, 256)
        self.block4 = DiscriminatorBlock(256, emb_size)
        self.linear1 = spectral_norm(nn.Linear(3*3*emb_size, 100))
        self.linear2 = spectral_norm(nn.Linear(100, 1))

    def forward(self, input):
        shape = list(input.shape)
        shape[1] = shape[1]//2
        mask = torch.randint(0, 2, shape)
        mask2 = torch.ones(shape)-mask
        mask = torch.cat((mask, mask2), dim=1)

        input = mask*input

        # Input 2x28x28
        x = self.block1(input)
        # Input 64x14x14
        x = self.block2(x)
        # Input 128x7x7
        x = self.block3(x)
        # Input 256x3x3
        x = self.block4(x, maxpool=False)

        x = x.view(-1)

        return F.sigmoid(self.linear2(F.leaky_relu(self.linear1(x))))
