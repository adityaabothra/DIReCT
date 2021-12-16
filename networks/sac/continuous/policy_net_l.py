import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from networks.base import BaseNetwork


class PolicyNetworkLSTM(BaseNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        eps,
        hidden_layers=[32],
        sigma_min=-20,
        sigma_max=2,
        hidden_dim=20,
        n_layers=2,
    ):
        super().__init__()
        self.eps = eps
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[0])
        self.out = nn.Linear(hidden_layers[0], 2*action_dim)

    def forward(self, x, l_prev_state):
        x = x.view(1, 1, self.state_dim)
        x, l_state = self.lstm(x, l_prev_state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        log_sigma = torch.clamp(
            log_sigma,
            min=self.sigma_min,
            max=self.sigma_max,
        )

        return mu.to(self.device), log_sigma.to(self.device), l_state

    def sample(self, state, l_prev_state, add_noise=True):
        mu, log_sigma, l_state = self.forward(state, l_prev_state)
        sigma = log_sigma.exp()
        probs = Normal(mu, sigma)

        if add_noise:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = torch.tanh(actions)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.eps)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs, l_state

    def init_states(self, size):
        hdn_st = torch.zeros(
            self.n_layers,
            size,
            self.hidden_dim,
        )
        cell_st = torch.zeros(
            self.n_layers,
            size,
            self.hidden_dim,
        )

        return hdn_st, cell_st