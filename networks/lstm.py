import torch
from torch import nn
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x, prev_state):
        x = x.view(x.shape[0], 1, self.input_dim)
        x, state = self.lstm(x, prev_state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out, state

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

        return (hdn_st, cell_st)