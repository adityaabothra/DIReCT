import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.fc1 = nn.Linear(state_dim, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        state_val = self.out(output)
        return state_val