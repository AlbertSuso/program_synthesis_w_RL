import torch.nn as nn
import torch.nn.functional as F

from nets.feature_extractors import Block


class Discriminator(nn.Module):
    def __init__(self, in_chanels=2, emb_size=512):
        super(Discriminator, self).__init__()

        self.block1 = Block(in_chanels, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, emb_size)
        self.linear1 = nn.Linear(3*3*emb_size, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, input):
        # Input 2x28x28
        x = self.block1(input)
        # Input 64x14x14
        x = self.block2(x)
        # Input 128x7x7
        x = self.block3(x)
        # Input 256x3x3
        x = self.block4(x, maxpool=False)

        x = x.view(-1)

        return self.linear2(F.relu(self.linear1(x)))
