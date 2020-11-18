import torch.nn as nn
import torch.nn.functional as F


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

