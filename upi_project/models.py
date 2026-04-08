import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudPolicy(nn.Module):
    def __init__(self):
        super(FraudPolicy, self).__init__()
        # Input: 3 features (Amount, Frequency, Location Risk)
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3) # Output: 3 actions (Allow, Flag, Block)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
        