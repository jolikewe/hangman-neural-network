"""Neural network architecture used to predict the hidden letters of a masked word."""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NUM_CHARS = 26
EMBED_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_rnn.pth"


class CustomNN(nn.Module):
    """Bidirectional GRU that outputs letter logits for every position of a word.

    Input is a one-hot encoded masked word (masked positions are all zeros);
    output logits have shape [batch, time, chars_len].
    """

    def __init__(
        self,
        chars_len: int = NUM_CHARS,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Linear(chars_len, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)  # bidirectional doubles hidden size
        self.fc = nn.Linear(hidden_dim * 2, chars_len)
        self.embed_dropout = nn.Dropout(dropout)  # defined but not used in forward

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # [B, T, embed_dim]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.norm(out)  # normalize bidirectional output
        return self.fc(out)  # [B, T, chars_len]


def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> CustomNN:
    """Load trained weights from `path` onto the CPU and return the model in eval mode."""
    model = CustomNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
