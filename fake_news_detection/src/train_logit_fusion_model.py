import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from fake_news_detection.models.logit_fusion_model import LogitFusionModel


class LogitsFusionDataset(Dataset):
    def __init__(self, logits1, logits2, labels):
        self.logits1 = torch.tensor(logits1, dtype=torch.float32)
        self.logits2 = torch.tensor(logits2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logits1[idx], self.logits2[idx], self.labels[idx]


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device,
    epochs=20, patience=3
):
    best_acc = 0.0
    best_model_state = None
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for logits1_batch, logits2_batch, labels_batch in train_loader:
            logits1_batch = logits1_batch.to(device)
            logits2_batch = logits2_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(logits1_batch, logits2_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for logits1_batch, logits2_batch, labels_batch in val_loader:
                logits1_batch = logits1_batch.to(device)
                logits2_batch = logits2_batch.to(device)
                outputs = model(logits1_batch, logits2_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels_batch.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Val Accuracy = {acc:.4f}")
        print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"\nEarly stopping triggered. Restoring best model (Val Accuracy: {best_acc:.4f})")
                model.load_state_dict(best_model_state)
                break

if __name__ == "__main__":
    train_logits = pd.read_csv(r'C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection_v2\fake_news_detection\logs\clip_concat_model_100k_sample_2_classes\train_logits.tsv', sep='\t')
    val_logits = pd.read_csv(r'C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection_v2\fake_news_detection\logs\clip_concat_model_100k_sample_2_classes\val_logits.tsv', sep='\t')

    # Split
    X1_train = train_logits[['title_image_logit_class_0', 'title_image_logit_class_1']].values
    X2_train = train_logits[['probit']].values
    y_train = train_logits['2_way_label'].values

    X1_val = val_logits[['title_image_logit_class_0', 'title_image_logit_class_1']].values
    X2_val = val_logits[['probit']].values
    y_val = val_logits['2_way_label'].values

    comment_model_preds = val_logits['predicted_label']
    title_image_model_preds = val_logits['title_image_predicted_label']

    print("Classification Report, Comments model:\n", classification_report(y_val, comment_model_preds, digits=4))
    print("Classification Report, Title+Image model:\n", classification_report(y_val, title_image_model_preds, digits=4))


    train_dataset = LogitsFusionDataset(X1_train, X2_train, y_train)
    val_dataset = LogitsFusionDataset(X1_val, X2_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogitFusionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
