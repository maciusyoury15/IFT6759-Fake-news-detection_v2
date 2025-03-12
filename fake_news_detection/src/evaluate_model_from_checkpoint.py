import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader

from fake_news_detection.src.dataset_rtf import MultiModalDataset
from fake_news_detection.models.multi_modal import MultiModalClassifier
from fake_news_detection.models.ViT import VisionTransformerModel
from fake_news_detection.models.text_transformer import TextTransformerModel
from fake_news_detection.src.train_model import setup_logging, load_config, evaluate, save_conf_matrix


def evaluate_model_from_checkpoint(config_path):
    config = load_config(config_path)
    save_dir = config["save_dir"]
    logger = setup_logging(save_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load dataset
    val_tsv = config["val_data"]
    image_folder = config["image_folder"]
    num_classes = config["num_classes"]
    val_dataset = MultiModalDataset(val_tsv, image_folder, num_classes)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Load models
    vit_model = VisionTransformerModel(device=device)
    text_model = TextTransformerModel(device=device)
    multi_modal_model = MultiModalClassifier(num_classes=num_classes).to(device)
    
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
    CONFIG_PATH =  r"C:\Users\Claire\Documents\UdeM\IFT6759\IFT6759-Fake-news-detection_v2\fake_news_detection\config\model\concat_model.yaml"
    evaluate_model_from_checkpoint(CONFIG_PATH)
