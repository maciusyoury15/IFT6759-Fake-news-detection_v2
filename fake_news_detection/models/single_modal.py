import torch
import torch.nn as nn

class TextOnlyClassifier(nn.Module):
    def __init__(
            self,
            text_dim=768,
            hidden_dim=512,
            dropout_p=0.5,
            num_classes=2
    ):
        super(TextOnlyClassifier, self).__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_embedding, text_embedding):
        x = torch.relu(self.fc1(text_embedding))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class ImageOnlyClassifier(nn.Module):
    def __init__(
            self,
            image_dim=320,
            hidden_dim=512,
            dropout_p=0.5,
            num_classes=2
    ):
        super(ImageOnlyClassifier, self).__init__()
        self.fc1 = nn.Linear(image_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_embedding, text_embedding):
        x = torch.relu(self.fc1(image_embedding))
        x = self.dropout(x)
        out = self.fc2(x)
        return out