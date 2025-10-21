'''
model_training.py
Created by jolikewe
Created on 2025-10-17 13:25:53

This is a python file. 
'''




# =================================================
# RNN model
# =================================================

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


## RNN architecture
class CustomNN(nn.Module):
    def __init__(self, chars_len=26, embed_dim=16, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(chars_len, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)    # bidirectional doubles hidden size
        self.fc = nn.Linear(hidden_dim * 2, chars_len)
        self.embed_dropout = nn.Dropout(dropout)    # Optional dropout after embedding

    def forward(self, x, lengths):
        x = self.embedding(x)               # [B, T, embed_dim]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.norm(out)                # normalize bidirectional output
        logits = self.fc(out)               # [B, T, chars_len]
        return logits


## Define model, optimizer, loss, learning rate
model7 = CustomNN(chars_len=26, embed_dim=16, hidden_dim=128, num_layers=2, dropout=0.2)
optimizer = torch.optim.Adam(model7.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()
# reduces LR by factor of 0.5 if val loss stagnant for 2 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)





