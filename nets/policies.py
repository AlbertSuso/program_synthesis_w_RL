import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNPolicy(nn.Module):
    def __init__(self, action_space_shapes, input_sizes=(9*512, 9*512, 8, 1), lstm_size=512, batch_size=64):
        super(RNNPolicy, self).__init__()

        features_size = sum(input_sizes)

        # MLP inicial
        self._fc1 = nn.Sequential(
            nn.Linear(features_size, int(features_size / (features_size / lstm_size) ** (1 / 3))),
            nn.Dropout(0.5),
            nn.ReLU())
        self._fc2 = nn.Sequential(
            nn.Linear(int(features_size / (features_size / lstm_size) ** (1 / 3)),
                      int(features_size / (features_size / lstm_size) ** (2 / 3))),
            nn.Dropout(0.3),
            nn.ReLU())
        self._fc3 = nn.Sequential(
            nn.Linear(int(features_size/(features_size/lstm_size)**(2/3)), lstm_size),
            nn.Dropout(0.15),
            nn.ReLU())

        # LSTM
        self._LSTM = nn.LSTM(input_size=lstm_size, hidden_size=lstm_size, num_layers=1, dropout=0)

        # Hidden states actuales del LSTM
        self._h = torch.zeros((1, batch_size, lstm_size)).cuda()
        self._c = torch.zeros((1, batch_size, lstm_size)).cuda()

        # Autoregressive decoder
        self._autoregressiveDecoder = AutoregressiveDecoder(action_space_shapes, lstm_size, batch_size)

    def forward(self, actualCanvasFeatures, objectiveCanvasFeatures, last_action, episode_percentage):
        features = torch.cat((actualCanvasFeatures, objectiveCanvasFeatures, last_action, torch.tensor([episode_percentage]*self._h.shape[1]).view(-1, 1).cuda()), dim=1)
        out_MLP = self._fc3(self._fc2(self._fc1(features)))
        out_MLP = out_MLP.view(1, self._h.shape[1], self._h.shape[2])
        seed, (self._h, self._c) = self._LSTM(out_MLP, (self._h, self._c))
        if episode_percentage == 1:
            self._h = torch.zeros(self._h.shape)
            self._c = torch.zeros(self._c.shape)

        action, entropy, log_probabilities = self._autoregressiveDecoder(seed.view(self._h.shape[1], -1))
        return action, entropy, log_probabilities


class AutoregressiveDecoder(nn.Module):
    def __init__(self, action_spaces_shapes, input_size=512, batch_size=64):
        """
        :param input_size: tamaño de la seed
        :param action_spaces_shapes: tupla con el numero posible de acciones en cada subacción.
        """
        super(AutoregressiveDecoder, self).__init__()

        self._action_spaces_shapes = action_spaces_shapes
        self._batch_size = batch_size

        # Capa FC común
        self._fc1 = nn.Sequential(
            nn.Linear(input_size, int(input_size//2)),
            nn.Dropout(0.3),
            nn.ReLU())
        self._fc2 = [nn.Sequential(nn.Linear(input_size//2, out_size), nn.Softmax(dim=1)) for out_size in action_spaces_shapes]

        #Capa FC de output
        self._fcOutput = nn.Sequential(
            nn.Linear(input_size+1, input_size),
            nn.ReLU())

    def forward(self, z):
        action = torch.zeros((self._batch_size, 1), dtype=torch.float32)
        log_probabilities = torch.zeros(self._batch_size, dtype=torch.float32)
        entropy = 0
        for layer in self._fc2:
            # Sampleamos la sub-acción
            distribution = layer(self._fc1(z))
            act = torch.distributions.Categorical(distribution).sample()

            action = torch.cat((action, act.view(-1, 1)), dim=1)
            entropy = entropy + torch.sum(torch.distributions.Categorical(distribution).entropy())
            for i in range(self._batch_size):
                log_probabilities[i] = log_probabilities[i] + torch.log(distribution[i, act[i]])

            # Obtenemos la siguiente seed
            z = self._fcOutput(torch.cat((z, act.view(self._batch_size, 1).cuda()), dim=1))

        action = action[:, 1:]
        return action, entropy, log_probabilities


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
