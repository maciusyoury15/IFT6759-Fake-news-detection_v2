import torch
import torch.nn as nn

class MultiModalClassifier(nn.Module):
    def __init__(self, image_dim=320, text_dim=768, hidden_dim=512, num_classes=2):
        super(MultiModalClassifier, self).__init__()
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.classifier1 = nn.Linear(hidden_dim * 2, int(hidden_dim / 2))
        self.classifier2 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.classifier3 = nn.Linear(int(hidden_dim / 4), num_classes)

    def forward(self, image_embedding, text_embedding):
        img_feat = torch.relu(self.image_fc(image_embedding))
        text_feat = torch.relu(self.text_fc(text_embedding))
        combined = torch.cat((img_feat, text_feat), dim=1)
        x = torch.relu(self.classifier1(combined))
        x = torch.relu(self.classifier2(x))
        x = self.classifier3(x)
        return x
