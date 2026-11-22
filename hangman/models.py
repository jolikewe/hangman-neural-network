"""Neural network architecture used to predict the hidden letters of a masked word."""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NUM_CHARS = 26
EMBED_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "model_rnn.pth"  # v1
V2_MODEL_PATH = MODELS_DIR / "model_rnn_v2.pth"


class CustomNN(nn.Module):
    """Bidirectional GRU that outputs letter logits for every position of a word.

    Input is a one-hot encoded masked word (masked positions are all zeros);
    output logits have shape [batch, time, chars_len].

    With `use_missed=True` (v2) the model also receives a 26-dim multi-hot vector
    of the letters guessed so far that are NOT in the word. It is concatenated to
    the one-hot letter at every timestep, so the embedding layer takes 52 inputs.
    """

    def __init__(
        self,
        chars_len: int = NUM_CHARS,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        use_missed: bool = False,
    ):
        super().__init__()
        self.chars_len = chars_len
        self.use_missed = use_missed
        self.embedding = nn.Linear(chars_len * (2 if use_missed else 1), embed_dim)
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

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, missed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`missed` is a [B, chars_len] multi-hot of wrong guesses (required when use_missed)."""
        if self.use_missed:
            if missed is None:
                raise ValueError("this model needs the `missed` letters vector")
            x = torch.cat([x, missed[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        x = self.embedding(x)  # [B, T, embed_dim]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.norm(out)  # normalize bidirectional output
        return self.fc(out)  # [B, T, chars_len]


def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> CustomNN:
    """Load trained weights from `path` onto the CPU and return the model in eval mode.

    v1 files are a bare state dict. v2 files are a dict with "state_dict" and
    "config" (the CustomNN constructor arguments).
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        model = CustomNN(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = CustomNN()
        model.load_state_dict(checkpoint)
    model.eval()
    return model
