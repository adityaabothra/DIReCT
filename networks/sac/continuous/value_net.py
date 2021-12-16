import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseNetwork


class ValueNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        op_dim=1,
        hidden_layers=[64],
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.out = nn.Linear(hidden_layers[0], 1)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # return state value
        return x