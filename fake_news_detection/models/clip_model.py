import torch
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

class CLIPImageEncoder:
    def __init__(self, device):
        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.embedding_size = 512
        
    def get_image_embedding(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
        image_embeds = outputs.image_embeds
        return image_embeds

class CLIPTextEncoder:
    def __init__(self, device):
        self.device = device
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.embedding_size = 512
        
    def get_text_embedding(self, text_inputs):

        with torch.no_grad():
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(self.device)
        
            outputs = self.model(**text_inputs)

        text_embeds = outputs.text_embeds
    
        return text_embeds