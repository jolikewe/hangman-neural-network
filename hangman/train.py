"""Train the v2 Hangman model.

v2 differs from v1 (notebooks/01_training.ipynb) in three ways:

1. Training states come from simulated games, not random masks. Each step a word
   is "played" with a noisy letter-frequency guesser; a state is sampled from the
   moments before the game ends (at most 5 wrong guesses, word not complete), so
   the model sees the revealed-letter patterns and wrong-guess sets of real games.
2. The model is told which guessed letters were wrong (`--variant missed`).
   `--variant nomissed` is the control: same data and loss, no missed-letter input.
3. The loss is a softmax cross-entropy over the letters that have not been guessed
   yet, at hidden positions only. This is exactly the distribution used when playing.

A fixed set of words from words_train.txt is held out for validation. They are never
trained on, and `data/words_test.txt` is never read by this script.

    python -m hangman.train --variant missed --epochs 14 --out models/model_rnn_v2.pth
    python -m hangman.train --variant missed --epochs 4 --lr 3e-4 --warm-start models/model_rnn.pth --out models/model_rnn_v2_ft.pth
    python -m hangman.train --variant nomissed --epochs 14 --out models/model_rnn_v2_nomissed.pth
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import re
import string
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hangman.engine import MAX_WRONG_GUESSES
from hangman.game import TRAIN_WORDS_PATH, HangmanRNN
from hangman.models import CustomNN

NUM_CHARS = 26
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VAL_WORDS = 5000
UNIFORM_ORDER_PROB = 0.25  # share of simulated games that guess letters in uniformly random order
ALPHA_RANGE = (0.5, 2.0)  # otherwise: letter weights = frequency ** alpha

# A sample is (letter indices [T], visible positions [T] bool, wrong-guess multi-hot [26] bool)
Sample = Tuple[np.ndarray, np.ndarray, np.ndarray]


def clean_word_list(words: List[str], min_len: int = 2, max_len: int = 30) -> List[str]:
    """Same cleaning as v1: lowercase alphabetic, length 2..30, not one repeated letter."""
    cleaned = set()
    for w in words:
        w = w.strip().lower()
        if re.fullmatch(r"[a-z]+", w) and min_len <= len(w) <= max_len and len(set(w)) > 1:
            cleaned.add(w)
    return sorted(cleaned)


def split_words(seed: int, val_words: int = VAL_WORDS) -> Tuple[List[str], List[str]]:
    """Deterministic (train, validation) split of the cleaned training dictionary."""
    words = clean_word_list(TRAIN_WORDS_PATH.read_text().split("\n"))
    rng = random.Random(seed)
    rng.shuffle(words)
    return sorted(words[val_words:]), sorted(words[:val_words])


def letter_frequencies(words: List[str]) -> np.ndarray:
    """Number of training words containing each letter (document frequency)."""
    counts = collections.Counter(c for w in words for c in set(w))
    return np.array([counts[c] for c in string.ascii_lowercase], dtype=np.float64)


def simulate_state(word: str, log_freq: np.ndarray, rng: np.random.Generator) -> Sample:
    """Sample one in-game state of `word` by playing it with a noisy frequency guesser."""
    idx = np.frombuffer(word.encode(), dtype=np.uint8) - 97
    letters = set(idx.tolist())
    alpha = 0.0 if rng.random() < UNIFORM_ORDER_PROB else rng.uniform(*ALPHA_RANGE)
    # Plackett-Luce order over all 26 letters via Gumbel keys
    order = np.argsort(-(alpha * log_freq + rng.gumbel(size=NUM_CHARS)))

    states = []  # (set of revealed letters, wrong letters) before each guess
    revealed, wrong = set(), []
    for letter in order.tolist():
        if len(revealed) == len(letters) or len(wrong) >= MAX_WRONG_GUESSES:
            break
        states.append((set(revealed), list(wrong)))
        if letter in letters:
            revealed.add(letter)
        else:
            wrong.append(letter)
    revealed, wrong = states[rng.integers(len(states))]
    visible = np.isin(idx, list(revealed)) if revealed else np.zeros(len(idx), dtype=bool)
    missed = np.zeros(NUM_CHARS, dtype=bool)
    missed[wrong] = True
    return idx, visible, missed


def collate(samples: List[Sample], use_missed: bool):
    """Pad a list of samples (sorted by descending length) into model inputs and targets."""
    lengths = torch.tensor([len(s[0]) for s in samples])
    B, T = len(samples), int(lengths.max())
    letters = torch.zeros(B, T, dtype=torch.long)
    visible = torch.zeros(B, T, dtype=torch.bool)
    missed = torch.zeros(B, NUM_CHARS)
    for b, (idx, vis, mis) in enumerate(samples):
        n = len(idx)
        letters[b, :n] = torch.from_numpy(idx.astype(np.int64))
        visible[b, :n] = torch.from_numpy(vis)
        missed[b] = torch.from_numpy(mis.astype(np.float32))
    valid = torch.arange(T)[None, :] < lengths[:, None]
    hidden = valid & ~visible  # positions the loss is computed on
    x = F.one_hot(letters, NUM_CHARS).float() * (valid & visible)[..., None]
    guessed = (x.sum(dim=1) > 0) | missed.bool()  # letters that can no longer be guessed
    return x, lengths, (missed if use_missed else None), letters, hidden, guessed


def make_batches(samples: List[Sample], rng: random.Random):
    """Length-bucketed batches of indices, in random order."""
    order = list(range(len(samples)))
    rng.shuffle(order)
    chunk = BATCH_SIZE * 50
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: -len(samples[i][0]))
        batches += [part[i:i + BATCH_SIZE] for i in range(0, len(part), BATCH_SIZE)]
    rng.shuffle(batches)
    return batches


def batch_loss(model: CustomNN, batch: List[Sample], use_missed: bool) -> Tuple[torch.Tensor, int]:
    """Summed cross-entropy over hidden positions, softmax restricted to unguessed letters."""
    x, lengths, missed, letters, hidden, guessed = collate(batch, use_missed)
    logits = model(x, lengths, missed) if use_missed else model(x, lengths)
    logits = logits.masked_fill(guessed[:, None, :], -1e9)
    loss = F.cross_entropy(logits[hidden], letters[hidden], reduction="sum")
    return loss, int(hidden.sum())


def evaluate_loss(model: CustomNN, samples: List[Sample], use_missed: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_idx in make_batches(samples, random.Random(0)):
            loss, n = batch_loss(model, [samples[i] for i in batch_idx], use_missed)
            total += loss.item()
            count += n
    return total / count


def game_win_rate(model: CustomNN, val_words: List[str], games: int, seed: int) -> float:
    """Win rate of the model (cell decoding, fixed opening) in games over held-out words."""
    from hangman.evaluate import evaluate

    player = HangmanRNN(timeout=None, model=model)
    return evaluate(player, games=games, seed=seed, progress=False, words=val_words)["success_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["missed", "nomissed"], default="missed")
    parser.add_argument("--epochs", type=int, default=14, help="each epoch simulates one state per training word")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--val-games", type=int, default=1000, help="games on held-out words after training")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="initialise from a v1 checkpoint (missed-letter input weights start at zero)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--out", type=Path, default=Path("models/model_rnn_v2.pth"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    use_missed = args.variant == "missed"
    train_words, val_words = split_words(args.seed)
    log_freq = np.log(letter_frequencies(train_words))
    print(f"{len(train_words):,} train words, {len(val_words):,} validation words, variant={args.variant}")

    val_rng = np.random.default_rng(args.seed + 1)
    val_samples = [simulate_state(w, log_freq, val_rng) for w in val_words for _ in range(4)]

    config = dict(use_missed=use_missed)
    model = CustomNN(**config)
    if args.warm_start is not None:
        init = torch.load(args.warm_start, map_location="cpu")
        init = init.get("state_dict", init)
        weight = init["embedding.weight"]
        if use_missed:  # [16, 26] -> [16, 52], new columns zero
            init["embedding.weight"] = torch.cat([weight, torch.zeros_like(weight)], dim=1)
        model.load_state_dict(init)
        print(f"warm start from {args.warm_start} (validation words were seen by that model)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    rng, batch_rng = np.random.default_rng(args.seed + 2), random.Random(args.seed + 3)
    best = math.inf
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        samples = [simulate_state(w, log_freq, rng) for w in train_words]
        model.train()
        total, count = 0.0, 0
        for batch_idx in make_batches(samples, batch_rng):
            loss, n = batch_loss(model, [samples[i] for i in batch_idx], use_missed)
            optimizer.zero_grad()
            (loss / n).backward()
            optimizer.step()
            total += loss.item()
            count += n
        scheduler.step()
        val_loss = evaluate_loss(model, val_samples, use_missed)
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "config": config}, args.out)
        print(f"epoch {epoch:2d}  train CE {total / count:.4f}  val CE {val_loss:.4f}  "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)

    best_model = CustomNN(**config)
    best_model.load_state_dict(torch.load(args.out)["state_dict"])
    best_model.eval()
    rate = game_win_rate(best_model, val_words, args.val_games, args.seed)
    print(f"best val CE {best:.4f}; win rate on {args.val_games} games over held-out validation words: {rate:.3f}")


if __name__ == "__main__":
    main()
