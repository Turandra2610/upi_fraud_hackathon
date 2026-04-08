import torch.nn as nn
import torch.nn.functional as F

class FraudPolicyNet(nn.Module):
    def __init__(self):
        super(FraudPolicyNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3) # Output: Probabilities for Allow, Flag, Block

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
        