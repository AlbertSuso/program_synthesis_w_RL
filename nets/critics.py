import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, num_steps, feature_extractor):
        super(Critic, self).__init__()

        features_size = 8*8*512 + 8

        self._feature_extractor = feature_extractor
        self._episode_percentage_embedding = nn.Embedding(num_steps, 8)

        self._fc1 = nn.Sequential(
            nn.Linear(features_size, int(features_size/features_size**(1/3))),
            nn.Dropout(0.4),
            nn.ReLU())

        self._fc2 = nn.Sequential(
            nn.Linear(int(features_size/features_size**(1/3)), int(features_size/features_size**(2/3))),
            nn.Dropout(0.2),
            nn.ReLU())

        self._fc3 = nn.Sequential(
            nn.Linear(int(features_size/features_size**(2/3)), 1),
            nn.Sigmoid())

    def forward(self, actualCanvas, objectiveCanvas, episode_percentage):
        batch_size = actualCanvas.shape[0]

        features_maps = torch.cat((actualCanvas, objectiveCanvas), dim=1)
        features_vector = self._feature_extractor(features_maps).view(batch_size, -1)

        episode_percentage_embedding = self._episode_percentage_embedding(episode_percentage)

        x = torch.cat((features_vector, episode_percentage_embedding), dim=1)
        return self._fc3(self._fc2(self._fc1(x)))
