
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from fake_news_detection.src.dataset_rtf import MultiModalDataset
from fake_news_detection.models.multi_modal import MultiModalClassifier
from fake_news_detection.models.ViT import VisionTransformerModel
from fake_news_detection.models.text_transformer import TextTransformerModel
from fake_news_detection.src.utils import setup_logging, load_config, save_conf_matrix 


def evaluate(model, vit_model, text_model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    batch_count = 0

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, text_inputs, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if images.numel() == 0 or not text_inputs:
                continue  

            image_embeddings = vit_model.get_image_embedding(images)
            text_embeddings = text_model.get_text_embedding(text_inputs)

            outputs = model(image_embeddings, text_embeddings)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            batch_count += 1

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    avg_val_loss = val_loss / batch_count if batch_count > 0 else float('inf')
    
    return avg_val_loss, accuracy, class_report, conf_matrix


def train(config_path):
    # Load configuration
    config = load_config(config_path)
    save_dir = config["save_dir"]
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info("Configuration loaded successfully.")
    logger.info(f"Configuration: {config}")
    
    # Paths
    train_tsv = config["train_data"]
    val_tsv = config["val_data"]
    image_folder = config["image_folder"]

    logger.info(f"Training data: {train_tsv}, Validation data: {val_tsv}")

    # Set up number of classes
    num_classes = config["num_classes"]
    logger.info(f"Training model using {num_classes}-way labels")

    # Load datasets
    train_dataset = MultiModalDataset(train_tsv, image_folder, num_classes)
    val_dataset = MultiModalDataset(val_tsv, image_folder, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize models
    vit_model = VisionTransformerModel(device=device)
    text_model = TextTransformerModel(device=device)
    multi_modal_model = MultiModalClassifier(num_classes=num_classes).to(device)

    logger.info("Models initialized successfully.")

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(multi_modal_model.parameters(), lr=config["learning_rate"])

    # Training parameters
    num_epochs = config["num_epochs"]
    patience = config["patience"]

    best_val_loss = float('inf')
    patience_counter = 0  # Early stopping counter

    logger.info("Starting training...")
    
    # Training loop
    for epoch in range(num_epochs):
        multi_modal_model.train()
        epoch_loss = 0.0
        batch_count = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tbar:
            for images, text_inputs, labels in tbar:
                images, labels = images.to(device), labels.to(device)

                if images.numel() == 0 or not text_inputs:
                    continue  

                image_embeddings = vit_model.get_image_embedding(images)
                text_embeddings = text_model.get_text_embedding(text_inputs)

                optimizer.zero_grad()
                outputs = multi_modal_model(image_embeddings, text_embeddings)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                tbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_val_loss, accuracy, class_report, conf_matrix = evaluate(multi_modal_model, vit_model, text_model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + class_report)

        # Save best model
        best_model_path = os.path.join(save_dir, "best_checkpoint.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': multi_modal_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, best_model_path)
            logger.info(f"Best model saved at {best_model_path} (val loss {best_val_loss:.4f})")
            save_conf_matrix(conf_matrix, save_dir)
            logger.info(f"Confusion matrix saved at {save_dir}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break

    logger.info("Training complete.")

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection_v2\fake_news_detection\config\model\concat_model.yaml"
    train(CONFIG_PATH)
