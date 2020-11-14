import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''ResNet Block'''

    def __init__(self, in_size, out_size, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.conv_res = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_size)

    def forward(self, x, maxpool=True):
        residual = x
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.bn_res(self.conv_res(residual))
        if maxpool:
            x = F.max_pool2d(nn.ReLU()(x), 2)
        return x


class ResNet12(nn.Module):
    def __init__(self, in_chanels=2, emb_size=512):
        super(ResNet12, self).__init__()
        self.block1 = Block(in_chanels, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, emb_size)
        self.emb_size = emb_size

        self.pretraining = True
        self.fc = nn.Linear(emb_size*8*8, 10)

    def forward(self, x):
        # Input 64x64x2
        x = self.block1(x)
        # Input 32x32x64
        x = self.block2(x)
        # Input 16x16x128
        x = self.block3(x)
        # Input 8x8x256
        x = self.block4(x, maxpool=False)

        return self.fc(x.view(x.shape[0], -1)) if self.pretraining else x

