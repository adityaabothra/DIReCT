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
        hidden_layers=[32, 32],
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.out = nn.Linear(hidden_layers[0], action_dim)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(-3e-3, 3e-3)
#         self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # return q-value
        return x