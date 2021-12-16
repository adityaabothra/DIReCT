import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseNetwork


class QNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        op_dim=1,
        hidden_layers=[32],
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.out = nn.Linear(hidden_layers[0], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # return q-value
        return x