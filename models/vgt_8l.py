import torch
import torch.nn as nn
import torch.nn.functional as F


class VGT_ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        nx = self.norm(x)
        out, _ = self.gru(nx)
        return residual + self.dropout(out)


class VGT_8L_Engine(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            VGT_ResidualBlock(d_model) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.fc(x)
        return logits, x  # x used for Forge / Norm monitoring