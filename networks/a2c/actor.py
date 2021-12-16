import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.fc1 = nn.Linear(state_dim, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, state):
        state.to(self.device)
        output = F.relu(self.fc1(state))
        policy = F.softmax(self.out(output), dim=-1)
        policy = Categorical(policy)

        return policy