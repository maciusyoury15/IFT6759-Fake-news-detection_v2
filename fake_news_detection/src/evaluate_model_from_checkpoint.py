import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from fake_news_detection.src.dataset_rtf import MultiModalDataset
from fake_news_detection.models.multi_modal import MultiModalClassifier
from fake_news_detection.models.ViT import VisionTransformerModel
from fake_news_detection.models.text_transformer import TextTransformerModel
from fake_news_detection.src.train_model import setup_logging, load_config, save_conf_matrix
from fake_news_detection.models.clip_model import CLIPImageEncoder, CLIPTextEncoder

def evaluate(model, vit_model, text_model, val_loader, criterion, device):
    model.eval()
    val_loss, batch_count = 0.0, 0

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

def evaluate_model_from_checkpoint(config_path):
    config = load_config(config_path)
    save_dir = config["save_dir"]
    logger = setup_logging(save_dir, "evaluation_from_checkpoint.log")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load dataset
    # val_tsv = config["val_data"]
    val_tsv=r'C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection\data\output_comments_model.tsv'
    image_folder = config["image_folder"]
    num_classes = config["num_classes"]
    vit_model_name = config["vit_model_name"]
    text_model_name = config["text_model_name"]
    if text_model_name == "openai/clip-vit-base-patch32":
        max_text_length = 77
    else:
        max_text_length = 128
    
    val_dataset = MultiModalDataset(
                        val_tsv, 
                        image_folder, 
                        num_classes,
                        vit_model_name=vit_model_name,
                        text_model_name=text_model_name,
                        max_text_length=max_text_length
                        )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Load models
    if vit_model_name == "timm/tiny_vit_5m_224.dist_in22k":
        vit_model = VisionTransformerModel(device=device)
    elif vit_model_name == "openai/clip-vit-base-patch32":
        vit_model = CLIPImageEncoder(device)
    else:
        print(f'unsupported VIT mode type {vit_model_name}')
    
    if text_model_name == "distilbert/distilbert-base-uncased":
        text_model = TextTransformerModel(device=device)
    elif text_model_name == "openai/clip-vit-base-patch32":
        text_model = CLIPTextEncoder(device)
    else:
        print(f'unsupported text mode type {text_model_name}')
    
    fusion_method = config['fusion_method']

    multi_modal_model = MultiModalClassifier(
                                image_dim=vit_model.embedding_size, 
                                text_dim=text_model.embedding_size,
                                num_classes=num_classes, 
                                fusion_method=fusion_method).to(device)
    
    # Load trained model checkpoint
    model_checkpoint = os.path.join(save_dir, "best_checkpoint.pth")
    checkpoint = torch.load(model_checkpoint, map_location=device)
    multi_modal_model.load_state_dict(checkpoint["model_state_dict"])
    multi_modal_model.eval()
    
    logger.info("Model loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    
    _, accuracy, class_report, conf_matrix = evaluate(multi_modal_model, vit_model, text_model, val_loader, criterion, device)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:\n" + class_report)

    save_conf_matrix(conf_matrix, save_dir)
    logger.info(f"Confusion matrix saved at {save_dir}")
    logger.info("Evaluation complete.")
    

if __name__ == "__main__":
    CONFIG_PATH =  r"C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection_v2\fake_news_detection\config\model\clip_concat_model.yaml"
    evaluate_model_from_checkpoint(CONFIG_PATH)
