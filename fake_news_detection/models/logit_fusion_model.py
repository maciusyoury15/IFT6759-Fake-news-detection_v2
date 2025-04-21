import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitFusionModel(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2):
        super(LogitFusionModel, self).__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, logits1, logits2):
        combined = torch.cat((logits1, logits2), dim=1)
        x = F.relu(self.fc1(combined))
        out = self.fc2(x)
        return out
