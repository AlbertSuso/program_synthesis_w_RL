import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        features_size = 2*(512*9)
        features_size += 1
        self._fc1 = nn.Linear(features_size, int(features_size/features_size**0.25))
        self._fc2 = nn.Linear(int(features_size/features_size**0.25), int(features_size/features_size**0.5))
        self._fc3 = nn.Linear(int(features_size/features_size**0.5), int(features_size/features_size**0.75))
        self._output = nn.Linear(int(features_size/features_size**0.75)+1, 1)

    def forward(self, featuresActualState, featuresObjective, episode_percentage):
        x = torch.cat((featuresActualState, featuresObjective, torch.tensor([episode_percentage]*featuresActualState.shape[0], dtype=torch.float32, device="cuda").reshape(featuresActualState.shape[0], 1)), dim=1)
        out = F.relu(self._fc1(x))
        out = F.relu(self._fc2(out))
        out = F.relu(self._fc3(out))
        out = torch.cat((out, torch.tensor([episode_percentage]*out.shape[0], dtype=torch.float32, device="cuda").reshape(out.shape[0], 1)), dim=1)
        return F.sigmoid(self._output(out))