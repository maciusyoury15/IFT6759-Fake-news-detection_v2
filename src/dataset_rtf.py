import numpy as np  # Import numpy to check for NaN values

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from transformers import AutoProcessor, AutoTokenizer

class MultiModalDataset(Dataset):
    def __init__(self, tsv_file, image_folder, vit_model_name="timm/tiny_vit_5m_224.dist_in22k", text_model_name="distilbert/distilbert-base-uncased", max_text_length=128):
        """
        Dataset class to load images from a folder and texts from a TSV file.
        
        Args:
            tsv_file (str): Path to the TSV dataset.
            image_folder (str): Path to the folder containing images.
            vit_model_name (str): Vision Transformer model name.
            text_model_name (str): Text Transformer model name.
            max_text_length (int): Maximum text length for tokenization.
        """
        # Load dataset
        self.data = pd.read_csv(tsv_file, sep="\t")

        # Ensure `text` column is a string and replace NaN with empty strings
        self.data["clean_title"] = self.data["clean_title"].fillna("").astype(str)

        # Store image directory path
        self.image_folder = image_folder

        self.processor = AutoProcessor.from_pretrained(vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_text_length = max_text_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess_image(self, image_id):
        """
        Loads and preprocesses an image from a local folder.
        
        Args:
            image_id (int or str): Unique ID for the image.
        
        Returns:
            torch.Tensor: Processed image tensor.
        """
        image_path = os.path.join(self.image_folder, f"{image_id}.jpg")

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            return torch.zeros((3, 224, 224))  # Return a blank image if missing

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)  # Remove batch dim
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros((3, 224, 224))  # Return blank tensor if error

    def preprocess_text(self, text):
        """Tokenizes text into tensor format."""
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_text_length, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx]["id"]  # Use ID to find the image
        text = self.data.iloc[idx]["clean_title"]
        label = torch.tensor(self.data.iloc[idx]["2_way_label"], dtype=torch.long)

        image_tensor = self.preprocess_image(image_id)
        text_inputs = self.preprocess_text(text)

        return image_tensor, text_inputs, label
