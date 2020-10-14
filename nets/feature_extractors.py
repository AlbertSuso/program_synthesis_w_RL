import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MnistFeatureExtractor(nn.Module):
    """INCLUIR DROPOUT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    def __init__(self):
        super(MnistFeatureExtractor, self).__init__()
        self._conv1 = nn.Conv2d(1, 5, 5, padding=2)
        self._conv2 = nn.Conv2d(5, 30, 3, padding=1)
        self._conv3 = nn.Conv2d(30, 90, 3, padding=1)
        self._pool = nn.MaxPool2d(2)
        self._fc1 = nn.Linear(7 * 7 * 90, 1500)
        self._fc2 = nn.Linear(1500, 500)
        self._fc3 = nn.Linear(500, 300)

    def forward(self, x):
        features = self._pool(F.relu(self._conv1(x)))
        features = self._pool(F.relu(self._conv2(features)))
        features = F.relu(self._conv3(features))
        return features.view(-1, 7 * 7 * 90)


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
    ''' In this network the input image is supposed to be 84x84x3 '''

    def __init__(self, in_chanels=1, emb_size=512):
        super(ResNet12, self).__init__()
        self.block1 = Block(in_chanels, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, emb_size)
        self.emb_size = emb_size
        self.clasificator = nn.Linear(emb_size*9, 10)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x, maxpool=False)
        x = x.view(-1, self.emb_size*9)
        return self.clasificator(x) if self.training else x



class PreTrainedFeatureExtractor(nn.Module):
    def __init__(self, pretrained_net=models.vgg11_bn(pretrained=True)):
        super(PreTrainedFeatureExtractor, self).__init__()
        self._extractor = pretrained_net
        for param in self._extractor.parameters():
            param.requires_grad = False
        self._extractor.classifier = nn.Flatten()

    def forward(self, x):
        return self._extractor(x)

    def requires_grad(self, flag):
        if flag:
            for param in self._extractor.parameters():
                param.requires_grad = True
        else:
            for param in self._extractor.parameters():
                param.requires_grad = False
