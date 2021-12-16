import torch.nn as nn
import torch.nn.functional as F

class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim, dueling_type='mean'):
        super().__init__()
        self.dueling_type = dueling_type
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        
        self.fc_value = nn.Linear(64, 1)
        self.fc_action_adv = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        v = self.fc_value(x)
        a = self.fc_action_adv(x)
        
        if self.dueling_type == 'max':
            q = v + a - a.max()
        else:
            q = v + a - a.mean()
        
        return q
