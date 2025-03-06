import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_rtf import MultiModalDataset
from models.multi_modal import MultiModalClassifier
from models.ViT import VisionTransformerModel
from models.text_transformer import TextTransformerModel


# Paths
tsv_path = "C:\\old laptop phd\\PhD\\Advanced ML Projects\\IFT6759-Fake-news-detection_v2\\src\\data\\raw\\sample_1k\\sample\\multimodal_train_1k.tsv"
image_folder = "C:\\old laptop phd\\PhD\Advanced ML Projects\\IFT6759-Fake-news-detection_v2\\src\\data\\raw\\sample_1k\\sample\\Images\\"  # Folder where images are stored

# Load dataset with image folder
dataset = MultiModalDataset(tsv_path, image_folder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = VisionTransformerModel(device=device)
text_model = TextTransformerModel(device=device)
model = MultiModalClassifier().to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for images, text_inputs, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Handle missing images (zero vectors)
        image_embeddings = vit_model.get_image_embedding(images)
        text_embeddings = text_model.get_text_embedding(text_inputs)

        # Skip empty batches
        if image_embeddings.numel() == 0 or text_embeddings.numel() == 0:
            print("Skipping empty batch")
            continue

        # Forward pass
        optimizer.zero_grad()
        outputs = model(image_embeddings, text_embeddings)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

    checkpoint_path = f"models/checkpoint.pth"
    checkpoint = {
                'epoch': epoch + 1,
                'iteration': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
    torch.save(checkpoint, checkpoint_path)

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
