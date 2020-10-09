import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistPolicy(nn.Module):
    """Para adaptarla a CIFAR10 solo hace falta cambiar el valor de la variable features_size"""
    def __init__(self, output_sizes=(196, 196, 2, 10, 20, 20, 20, 20)):
        super(MnistPolicy, self).__init__()

        features_size = 2 * (512 * 9)

        self._fcControl1 = nn.Linear(features_size, int(features_size/(features_size/output_sizes[0])**(1/3)))
        self._fcControl2 = nn.Linear(int(features_size/(features_size/output_sizes[0])**(1/3)), int(features_size/(features_size/output_sizes[0])**(2/3)))
        self._outputControl = nn.Linear(int(features_size/(features_size/output_sizes[0])**(2/3)), output_sizes[0])

        self._fcEnd1 = nn.Linear(features_size, int(features_size/(features_size/output_sizes[1])**(1/3)))
        self._fcEnd2 = nn.Linear(int(features_size/(features_size/output_sizes[1])**(1/3)), int(features_size/(features_size/output_sizes[1])**(2/3)))
        self._outputEnd = nn.Linear(int(features_size/(features_size/output_sizes[1])**(2/3)), output_sizes[1])

        self._fcFlag = nn.Linear(features_size, int(features_size/(features_size/output_sizes[2])**(1/2)))
        self._outputFlag = nn.Linear(int(features_size/(features_size/output_sizes[2])**(1/2)), output_sizes[2])

        self._fcPressure = nn.Linear(features_size, int(features_size/(features_size/output_sizes[3])**(1/2)))
        self._outputPressure = nn.Linear(int(features_size/(features_size/output_sizes[3])**(1/2)), output_sizes[3])

        self._fcSize = nn.Linear(features_size, int(features_size/(features_size/output_sizes[4])**(1/2)))
        self._outputSize = nn.Linear(int(features_size/(features_size/output_sizes[4])**(1/2)), output_sizes[4])

        #self._fcRed = nn.Linear(features_size, int(features_size/(features_size/output_sizes[5])**(1/2)))
        #self._outputRed = nn.Linear(int(features_size/(features_size/output_sizes[5])**(1/2)), output_sizes[5])

        #self._fcGreen = nn.Linear(features_size, int(features_size/(features_size/output_sizes[6])**(1/2)))
        #self._outputGreen = nn.Linear(int(features_size/(features_size/output_sizes[6])**(1/2)), output_sizes[6])

        #self._fcBlue = nn.Linear(features_size, int(features_size/(features_size/output_sizes[7])**(1/2)))
        #self._outputBlue = nn.Linear(int(features_size/(features_size/output_sizes[7])**(1/2)), output_sizes[7])

    def forward(self, canvasFeatures, objectiveFeatures):
        features = torch.cat((canvasFeatures, objectiveFeatures), dim=1)

        outControl = F.softmax(self._outputControl(F.relu(self._fcControl2(F.relu(self._fcControl1(features))))), dim=1)
        outEnd = F.softmax(self._outputEnd(F.relu(self._fcEnd2(F.relu(self._fcEnd1(features))))), dim=1)
        outFlag = F.softmax(self._outputFlag(F.relu(self._fcFlag(features))), dim=1)
        outPressure = F.softmax(self._outputPressure(F.relu(self._fcPressure(features))), dim=1)
        outSize = F.softmax(self._outputSize(F.relu(self._fcSize(features))), dim=1)
        #outRed = F.softmax(self._outputRed(F.relu(self._fcRed(features))), dim=1)
        #outGreen = F.softmax(self._outputGreen(F.relu(self._fcGreen(features))), dim=1)
        #outBlue = F.softmax(self._outputBlue(F.relu(self._fcBlue(features))), dim=1)
        return [outControl, outEnd, outFlag, outPressure, outSize]
