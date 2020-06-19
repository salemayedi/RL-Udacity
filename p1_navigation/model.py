import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units = 32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        #self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc1_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units = 32):
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_common = nn.Linear(state_size, fc1_units)
        self.fc2_common = nn.Linear(fc1_units, fc3_units)
        self.fc_V = nn.Linear(fc3_units, action_size)
        self.fc_A = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Common part
        x = F.relu(self.fc1_common(state))
        x = F.relu(self.fc2_common(x))
        # V and A
        V = self.fc_V(x)
        Adv = self.fc_A(x)
        # Q
        Adv_mean = torch.mean(Adv, dim = 1).unsqueeze(1)
        Q = V+ Adv - Adv_mean
        return Q
