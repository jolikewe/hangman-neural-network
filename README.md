# Hangman Neural Network

A bidirectional GRU that plays Hangman by predicting the hidden letters of a partially revealed word, plus a self-hosted game engine to benchmark it.

**[▶ Live demo](https://jolikewe.github.io/hangman-neural-network/)**: play Hangman against the trained network in your browser.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/API-FastAPI%20(optional)-009688)

## Overview

In Hangman the player sees a word with its letters masked as `_` and may make six wrong guesses. The classic baseline filters a dictionary by the current pattern and guesses the most frequent remaining letter. It works poorly when the secret words are not in the dictionary, which is the setting here: the test words are disjoint from the training words.

This project replaces that baseline with a sequence model. Training words are randomly masked, with all occurrences of a chosen letter hidden together, and the network learns to reconstruct the full word. At play time it outputs a letter distribution for every masked position. Letters already guessed are removed, each position is renormalised, and the single most probable (position, letter) pair is guessed. The first move, when nothing is revealed, uses a fixed vowel-first order.

## Results

Benchmark on the bundled engine: 1,000 games per strategy on the same random unseen words (`python -m hangman.evaluate --strategy both --games 1000 --seed 0`, stored in `notebooks/03_evaluation.ipynb`).

| Strategy | Wins / 1,000 | Success rate |
| --- | --- | --- |
| Dictionary baseline (pattern filter + letter frequency) | 188 | 18.8% |
| **Bidirectional GRU** | **527** | **52.7%** |

On an earlier external evaluation of 1,000 recorded games on unseen words, the same model scored **51.7%**, consistent with the local benchmark.

| Training detail | Value |
| --- | --- |
| Final validation loss (BCE, 10 epochs) | 0.1013 |
| Final training loss | 0.1015 |
| Model size | 1.59 MB (416,202 parameters) |

Loss per epoch, from `notebooks/01_training.ipynb`:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 0.1241 | 0.1113 | 0.1079 | 0.1060 | 0.1047 | 0.1038 | 0.1031 | 0.1024 | 0.1019 | 0.1015 |
| Validation | 0.1129 | 0.1084 | 0.1059 | 0.1047 | 0.1036 | 0.1030 | 0.1025 | 0.1021 | 0.1016 | 0.1013 |

## Repository structure

```
hangman/
  engine.py            Self-hosted game engine: secret words, rules, win statistics
  evaluate.py          CLI benchmark of strategies against the engine
  server.py            Optional REST API (FastAPI) around the engine
  game.py              HangmanBasic (dictionary baseline) and HangmanRNN (model-driven) players
  models.py            CustomNN architecture and load_model helper
notebooks/
  01_training.ipynb    Word cleaning, masked training data, training loop, example predictions
  02_testing.ipynb     Inference examples with per-position probabilities
  03_evaluation.ipynb  Step-by-step game and 1,000-game benchmark on the engine
data/
  words_train.txt      Training dictionary (227,300 words)
  words_test.txt       Held-out secret words used by the engine (208,230 words)
models/
  model_rnn.pth        Trained weights
requirements.txt
```

## Getting started

Requires Python 3.9 or newer. Run commands from the repository root.

```bash
pip install -r requirements.txt
```

**Benchmark** the baseline and the model (add `--demo` to watch one game move by move):

```bash
python -m hangman.evaluate --strategy both --games 1000 --seed 0
```

**Use the model** from Python:

```python
from hangman import HangmanRNN, load_model

player = HangmanRNN(timeout=None, model=load_model())
player.guessed_letters = ["a", "p", "e"]
print(player.guess("a p p _ e"))   # -> "l"
```

**Play through the engine** directly:

```python
from hangman.engine import HangmanEngine

engine = HangmanEngine(seed=0)
game = engine.new_game()
game = engine.guess(game.game_id, "e")
print(game.masked_word, game.tries_remaining, game.status)
```

**Serve it over HTTP** so any client, in any language, can play:

```bash
pip install fastapi uvicorn
uvicorn hangman.server:app --port 8000      # interactive docs at http://localhost:8000/docs
```

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /games` | `{"recorded": false}` | new game: `game_id`, masked `word`, `tries_remains` |
| `POST /games/{game_id}/guess` | `{"letter": "e"}` | updated game; `answer` is revealed once it ends |
| `GET /status` | | games played, wins and success rate (practice and recorded) |

Notebooks are meant to be run from inside `notebooks/` (`pip install jupyter pandas`). To retrain, run `01_training.ipynb`; the cell that saves weights is disabled with `if False:` so that `models/model_rnn.pth` is not overwritten by accident.

## Method

**Data.** `words_train.txt` is cleaned to lowercase alphabetic words of length 2 to 30 that are not a single repeated letter (227,259 words remain, 42 removed). Each word is repeated 3 times with independent random masks and duplicates are dropped, giving 681,777 samples, split 90/10 into training and validation sets.

**Input and target.** A word is one-hot encoded per position (26 letters); masked positions are all zeros. The target is the full one-hot word, and the loss is computed only on masked positions.

**Model.** Linear embedding (26 to 16), 2-layer bidirectional GRU (hidden size 128, dropout 0.2), layer norm, and a linear layer back to 26 letter logits per position.

**Training.** Adam (learning rate 1e-3), `BCEWithLogitsLoss`, batch size 256, 10 epochs, with a `ReduceLROnPlateau` scheduler (factor 0.5, patience 2). Batches are padded dynamically and packed by length.

**Inference.** See Overview. The strategy lives in `HangmanRNN.guess` in `hangman/game.py`.

## Engine rules

- Secret words are drawn from `data/words_test.txt`, which shares no words with the training dictionary.
- Each guess reveals every occurrence of the letter; a wrong guess costs one of 6 lives.
- Guessing a letter twice ends the game as a loss, so a strategy cannot loop forever.
- `seed` makes the sequence of secret words reproducible, so strategies can be compared on identical games.
