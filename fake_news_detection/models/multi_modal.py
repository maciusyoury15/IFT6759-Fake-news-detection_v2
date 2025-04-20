import torch
import torch.nn as nn

class MultiModalClassifier(nn.Module):
    def __init__(
            self, 
            image_dim=320, 
            text_dim=768, 
            fusion_output_size=512, 
            hidden_dim=512, 
            dropout_p=0.5, 
            num_classes=2, 
            fusion_method='concat',
            num_attention_heads=4
    ):
        super(MultiModalClassifier, self).__init__()
        
        self.fusion_method = fusion_method

        if self.fusion_method == 'concat':
            self.fusion = nn.Linear(in_features=(image_dim + text_dim), out_features=hidden_dim)
        elif self.fusion_method == 'cross_attention':
            self.image_proj = nn.Linear(image_dim, hidden_dim)
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=True)
            self.fusion = nn.Linear(hidden_dim, fusion_output_size)
        else:
            raise ValueError("Unsupported fusion method. Choose from ['concat', 'cross_attention']")
        
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(in_features=fusion_output_size, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
  

    def forward(self, image_embedding, text_embedding):
        if self.fusion_method == 'concat':
            img_feat = torch.relu(image_embedding)
            text_feat = torch.relu(text_embedding)
            combined = torch.cat((img_feat, text_feat), dim=1)
            fused = self.dropout(torch.relu(self.fusion(combined)))

        elif self.fusion_method == 'cross_attention':
            img_feat = self.image_proj(image_embedding).unsqueeze(1)  
            text_feat = self.text_proj(text_embedding).unsqueeze(1)  
            attended_text, _ = self.cross_attention(query=text_feat, key=img_feat, value=img_feat)
            attended_text = attended_text.squeeze(1) 
            fused = self.dropout(torch.relu(self.fusion(attended_text)))
        
        hidden = torch.relu(self.fc1(fused))
        out = self.fc2(hidden)
        
        return out
