import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, num_steps, feature_extractor, canvas_length):
        super(Critic, self).__init__()

        features_size = 3*3*512 + 16 + 8

        self._feature_extractor = feature_extractor
        self._episode_percentage_embedding = nn.Embedding(num_steps, 8)
        self._brush_position_embedding = nn.Embedding(canvas_length, 16)

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

    def forward(self, actualCanvas, objectiveCanvas, brush_position, episode_percentage):
        batch_size = actualCanvas.shape[0]

        features_maps = torch.cat((actualCanvas, objectiveCanvas), dim=1)
        features_vector = self._feature_extractor(features_maps).view(batch_size, -1)

        episode_percentage_embedding = self._episode_percentage_embedding(episode_percentage)
        brush_position_embedding = self._brush_position_embedding(brush_position)

        x = torch.cat((features_vector, brush_position_embedding, episode_percentage_embedding.view(features_vector.shape[0], -1)), dim=1)
        return self._fc3(self._fc2(self._fc1(x))).view(-1)
