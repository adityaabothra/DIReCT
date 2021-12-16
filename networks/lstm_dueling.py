import torch
from torch import nn
from torch.nn import functional as F


class LSTMDueling(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=20,
        n_layers=4,
        dueling_type='mean',
    ):
        super().__init__()
        self.dueling_type = dueling_type
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_value = nn.Linear(32, 1)
        self.fc_action_adv = nn.Linear(32, output_dim)

    def forward(self, x, prev_state):
        x = x.view(x.shape[0], 1, self.input_dim)
        x, state = self.lstm(x, prev_state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_value(x)
        a = self.fc_action_adv(x)

        if self.dueling_type == 'max':
            q = v + a - a.max()
        else:
            q = v + a - a.mean()

        return q, state

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