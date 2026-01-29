import torch
import torch.nn.functional as F
    
class EmbedModel(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EmbedModel, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)        # [B, T, D]
        x = x.mean(dim=1)           # pooling
        x = F.normalize(x, dim=-1)  # L2
        return x