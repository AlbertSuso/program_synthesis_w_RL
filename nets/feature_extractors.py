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


class ResNet1(nn.Module):
    def __init__(self, in_chanels=1, emb_size=128):
        super(ResNet1, self).__init__()
        self.block1 = Block(in_chanels, 64)
        self.block2 = Block(64, emb_size)
        self.emb_size = emb_size
        self.clasificator = nn.Linear(emb_size*49, 10)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)

        return self.clasificator(x.view(-1, self.emb_size*49)) if self.training else x


class ResNet2(nn.Module):
    def __init__(self, in_chanels=256, emb_size=1024):
        super(ResNet2, self).__init__()
        self.block1 = Block(in_chanels, 512)
        self.block2 = Block(512, 1024)
        self.emb_size = emb_size

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x, maxpool=False)

        return x.view(-1, self.emb_size*9)
