import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.feature_extractors import ResNet12


class RNNPolicy(nn.Module):
    def __init__(self, num_steps, action_space_shapes, feature_extractor, lstm_size=512, batch_size=64):
        super(RNNPolicy, self).__init__()

        self.num_steps = num_steps
        self._firstExecution = True

        self._feature_extractor = feature_extractor

        self._end_position_embedding = nn.Embedding(action_space_shapes[0], 16)
        self._episode_percentage_embedding = nn.Embedding(num_steps, 8)

        features_size = 3*3*512 + 16 + 8

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
        self._h = torch.zeros((1, batch_size, lstm_size))
        self._c = torch.zeros((1, batch_size, lstm_size))

        # Autoregressive decoder
        self._autoregressiveDecoder = AutoregressiveDecoder(action_space_shapes, lstm_size, batch_size)

    def forward(self, actualCanvas, objectiveCanvas, last_action_end_position, episode_percentage):
        if self._firstExecution:
            self._h = torch.zeros_like(self._h, device=actualCanvas.device)
            self._c = torch.zeros_like(self._c, device=actualCanvas.device)
            self._firstExecution = False

        features_maps = torch.cat((actualCanvas, objectiveCanvas), dim=1)
        features_vector = self._feature_extractor(features_maps).view(self._h.shape[1], -1)

        last_action_end_position_embedding = self._end_position_embedding(last_action_end_position)
        episode_percentage_embedding = self._episode_percentage_embedding(episode_percentage)

        features = torch.cat((features_vector, last_action_end_position_embedding, episode_percentage_embedding), dim=1)

        out_MLP = self._fc3(self._fc2(self._fc1(features))).view(1, self._h.shape[1], self._h.shape[2])

        seed, (self._h, self._c) = self._LSTM(out_MLP, (self._h, self._c))

        if episode_percentage[0] == self.num_steps-1:
            self._h = torch.zeros(self._h.shape, device=self._h.device)
            self._c = torch.zeros(self._c.shape, device=self._c.device)

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

        self._fc1 = nn.ModuleList([nn.Sequential(nn.Linear(input_size, int(input_size//2)), nn.Dropout(0.3), nn.ReLU(),
                                   nn.Linear(input_size//2, out_size), nn.Softmax(dim=1))
                                  for out_size in action_spaces_shapes])

        self._embeddings = nn.ModuleList([nn.Embedding(action_shape, 16) for action_shape in action_spaces_shapes])

        #Capa FC de output
        self._fcOutput = nn.Sequential(
            nn.Linear(input_size+16, input_size),
            nn.ReLU())

    def forward(self, z):
        action = torch.zeros((self._batch_size, 1), dtype=torch.long, device=z.device)
        log_probabilities = torch.zeros(self._batch_size, dtype=torch.float32, device=z.device)
        entropy = 0
        for mlp, embedding in zip(self._fc1, self._embeddings):
            # Sampleamos la sub-acción
            distribution = mlp(z)
            act = torch.distributions.Categorical(distribution).sample()

            action = torch.cat((action, torch.tensor(act, dtype=torch.long, device=z.device).view(-1, 1)), dim=1)
            entropy = entropy + torch.sum(torch.distributions.Categorical(distribution).entropy())

            for i in range(self._batch_size):
                log_probabilities[i] = log_probabilities[i] + torch.log(distribution[i, act[i]])

            embedding_act = embedding(act)

            # Obtenemos la siguiente seed
            z = self._fcOutput(torch.cat((z, embedding_act), dim=1))

        action = action[:, 1:]
        return action.cpu(), entropy, log_probabilities
