import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from networks.base import BaseNetwork


class PolicyNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        eps,
        hidden_layers=[32, 32],
    ):
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.out = nn.Linear(hidden_layers[1], action_dim)

    def forward(self, state):
        x = state
        x.to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = F.softmax(x, dim=-1)

        return x

    def sample(self, state):
        probs = self.forward(state)
        action_dist = Categorical(probs)
        actions = action_dist.sample().to(self.device)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)

        return actions, probs, log_probs