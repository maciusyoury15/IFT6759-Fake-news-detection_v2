

import torch
from transformers import AutoModel

class TextTransformerModel:
    def __init__(self, model_name="distilbert/distilbert-base-uncased", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def get_text_embedding(self, text_inputs):
        """
        Extracts text embeddings from a batch of tokenized inputs.
        
        Args:
            text_inputs (dict): Tokenized input dictionary (from tokenizer).
        
        Returns:
            torch.Tensor: Text embeddings (batch_size, embedding_dim).
        """
        with torch.no_grad():
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(self.device)
            outputs = self.model(**text_inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embedding
