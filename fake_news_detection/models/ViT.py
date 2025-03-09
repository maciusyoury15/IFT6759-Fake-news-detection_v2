import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests




class VisionTransformerModel:
    def __init__(self, model_name="timm/tiny_vit_5m_224.dist_in22k", device=None):
        """
        Wrapper for Vision Transformer model.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def get_image_embedding(self, images):
        """
        Extracts embeddings from a batch of images.
        
        Args:
            images (torch.Tensor): Batch of images (batch_size, 3, 224, 224).
        
        Returns:
            torch.Tensor: Image embeddings (batch_size, embedding_dim).
        """
        images = images.to(self.device)  # Move images to the correct device
        with torch.no_grad():
            outputs = self.model(pixel_values=images)
            embedding = outputs.last_hidden_state.mean(dim=[2, 3])  # Global Average Pooling
        return embedding

