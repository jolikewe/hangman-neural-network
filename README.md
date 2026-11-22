# Hangman Neural Network

A bidirectional GRU that plays Hangman by predicting the hidden letters of a partially revealed word, plus a self-hosted game engine to benchmark it. The best model wins 56.5% of games on unseen words, against 19.0% for a dictionary baseline.

**[▶ Live demo](https://jolikewe.github.io/hangman-neural-network/)**: play Hangman against the trained network in your browser.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/API-FastAPI%20(optional)-009688)

## Overview

In Hangman the player sees a word with its letters masked as `_` and may make six wrong guesses. The classic baseline filters a dictionary by the current pattern and guesses the most frequent remaining letter. It works poorly when the secret words are not in the dictionary, which is the setting here: the test words are disjoint from the training words.

This project replaces that baseline with a sequence model. A bidirectional GRU reads the masked word and outputs a letter distribution for every hidden position. Two versions are included:

- **v1**: trained on randomly masked words; plays the single most probable (position, letter) cell.
- **v2**: v1 fine-tuned on masked states taken from simulated games, with the wrong guesses given as an extra input; plays the letter with the largest probability summed over the hidden positions.

## Results

All numbers come from the bundled engine (`python -m hangman.evaluate --strategy all --games 2000 --seed 0 1`, reproduced in `notebooks/04_improvements.ipynb`): 2,000 games per seed on seeds 0 and 1, the same random unseen words for every variant. One win rate has a standard error of about 1.1 percentage points at 2,000 games, so differences under about 1.5 points between single rows are within noise.

| Variant | Seed 0 | Seed 1 | Mean | vs v1 |
| --- | --- | --- | --- | --- |
| Dictionary baseline (pattern filter + letter frequency) | 18.45% | 19.55% | 19.00% | -33.25 |
| **v1** original (single best cell, fixed vowel-first opening) | 52.15% | 52.35% | 52.25% | 0.00 |
| v1 + sum decoding | 54.30% | 54.20% | 54.25% | +2.00 |
| v1 + noisy-or decoding | 54.60% | 53.50% | 54.05% | +1.80 |
| v1 + sum decoding + length-conditioned opening | 53.95% | 54.10% | 54.02% | +1.77 |
| Scratch model, game-state training, no missed input, best cell | 46.45% | 46.70% | 46.58% | -5.67 |
| Scratch model, game-state training, no missed input, sum | 50.65% | 50.70% | 50.68% | -1.58 |
| Scratch model, game-state training, missed input, best cell | 46.70% | 47.05% | 46.88% | -5.38 |
| Scratch model, game-state training, missed input, sum | 50.25% | 51.10% | 50.68% | -1.58 |
| v2 (v1 fine-tuned, missed input), best cell | 53.55% | 52.95% | 53.25% | +1.00 |
| **v2 + sum decoding** (shipped) | **56.50%** | **56.55%** | **56.52%** | **+4.27** |

The same v1 model scored 52.7% on 1,000 games with seed 0 (the figure in earlier versions of this README) and 51.7% on an earlier external evaluation of 1,000 recorded games.

### What improved accuracy and why

- **Sum decoding (+2.0 points, no retraining).** v1 guessed the one brightest cell of the heatmap. That letter can be a confident guess for a single slot while a different letter appears in several slots. Ranking letters by the probability mass summed over all hidden positions is closer to what Hangman rewards (reveal many letters per guess). Noisy-or (`1 - prod(1 - p)`) gave the same gain within noise; sum is simpler and was marginally ahead on validation.
- **Fine-tuning on simulated game states (about +2.3 points over v1 + sum).** Random masks hide each letter with probability 0.5, which rarely resembles a real game (frequent letters revealed first, a few wrong guesses). v2 continues training v1 on states from simulated games and uses a cross-entropy loss over the letters not yet guessed, exactly the distribution used at play time.
- **Wrong-guess input: no measurable effect.** Among the two scratch models, which differ only in this input, the means are identical (50.68% each). v2 still takes the input, but its gain over v1 + sum cannot be attributed to it; a fine-tune without the input was not run.
- **Length-conditioned opening: no gain.** Choosing the opening letters from letter-presence frequencies of training words of the same length gave 54.02% against 54.25% for the fixed order on v1, and no difference on validation, so v2 keeps the fixed opening.
- **Training from scratch on simulated states was worse than v1** (50.7% with sum decoding). It saw about 2.7M states (12 epochs of one state per word) against roughly 6.1M samples for v1, and it held 5,000 words out of training. v2 therefore warm-starts from v1.

### Training and selection protocol

- Decoding and opening were chosen on 3,000 games over 5,000 validation words held out of `words_train.txt` (scratch models only, since v1 had seen those words): sum 49.5% vs single cell 45.9% vs noisy-or 49.2% for the missed-input model, and fixed vs length-conditioned opening within 0.2 points. `data/words_test.txt` was never used for training or for choosing these options.
- Caveat: the fine-tuning recipe (4 epochs, learning rate 3e-4, from v1) was tried once after the scratch runs had been scored on the test words, and was not tuned further. The first fine-tune is the one reported. Its validation loss is not an independent measure, because v1 had seen the validation words.
- v1 training detail: validation BCE 0.1013 after 10 epochs (416,202 parameters, 1.59 MB); per-epoch losses are in `notebooks/01_training.ipynb`. v2 has 416,618 parameters.

## Repository structure

```
hangman/
  engine.py            Self-hosted game engine: secret words, rules, win statistics (unchanged)
  evaluate.py          CLI benchmark: --strategy basic | rnn-v1 | rnn-v2 | rnn | both | all, several seeds
  train.py             CLI training of v2 (game-state sampling, missed-letter input, warm start)
  server.py            Optional REST API (FastAPI) around the engine
  game.py              HangmanBasic (dictionary baseline) and HangmanRNN (decoding and opening options)
  models.py            CustomNN architecture (use_missed switch) and load_model helper
notebooks/
  01_training.ipynb    v1: word cleaning, masked training data, training loop, example predictions
  02_testing.ipynb     Inference examples with per-position probabilities
  03_evaluation.ipynb  Step-by-step game and 1,000-game benchmark of v1
  04_improvements.ipynb  Executed ablation of every variant (2,000 games x seeds 0 and 1)
data/
  words_train.txt      Training dictionary (227,300 words)
  words_test.txt       Held-out secret words used by the engine (208,230 words)
models/
  model_rnn.pth        v1 weights
  model_rnn_v2.pth     v2 weights (shipped model)
  ablation/            Scratch-trained models used in the ablation (with and without the missed input)
requirements.txt
```

## Getting started

Requires Python 3.9 or newer. Run commands from the repository root.

```bash
pip install -r requirements.txt
```

**Benchmark** the baseline, v1 and v2 (add `--demo` to watch one game move by move; `--val` plays on the held-out validation words instead):

```bash
python -m hangman.evaluate --strategy all --games 2000 --seed 0 1
python -m hangman.evaluate --strategy rnn --model models/model_rnn.pth --decoding sum --opening fixed   # any combination
```

**Use the model** from Python:

```python
from hangman import HangmanRNN, load_model
from hangman.models import V2_MODEL_PATH

player = HangmanRNN(timeout=None, model=load_model(V2_MODEL_PATH), decoding="sum")
player.guessed_letters = ["a", "p", "e"]
print(player.guess("a p p _ e"))
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

Notebooks are meant to be run from inside `notebooks/` (`pip install jupyter pandas`).

**Train v2** (seeded; the validation split, simulated states and initialisation are all derived from `--seed`). The shipped model is the fine-tune from v1; the scratch commands reproduce the ablation models:

```bash
python -m hangman.train --variant missed --epochs 4 --lr 3e-4 --warm-start models/model_rnn.pth --out models/model_rnn_v2.pth
python -m hangman.train --variant missed   --epochs 12 --out models/ablation/v2_scratch_missed.pth
python -m hangman.train --variant nomissed --epochs 12 --out models/ablation/v2_scratch_nomissed.pth
```

Training v2 does not read `data/words_test.txt`. On 2 CPU cores a 12-epoch scratch run took about 51 minutes (two runs in parallel, one thread each) and the 4-epoch fine-tune about 6 minutes. To retrain v1, run `01_training.ipynb`; the cell that saves weights is disabled with `if False:` so that `models/model_rnn.pth` is not overwritten by accident.

## Method

**Data.** `words_train.txt` is cleaned to lowercase alphabetic words of length 2 to 30 that are not a single repeated letter (227,259 words remain, 42 removed). Each word is repeated 3 times with independent random masks and duplicates are dropped, giving 681,777 samples, split 90/10 into training and validation sets.

**Input and target.** A word is one-hot encoded per position (26 letters); masked positions are all zeros. The target is the full one-hot word, and the loss is computed only on masked positions.

**Model (v1 and v2).** Linear embedding (26 to 16; 52 to 16 for v2), 2-layer bidirectional GRU (hidden size 128, dropout 0.2), layer norm, and a linear layer back to 26 letter logits per position.

**Training (v1).** Adam (learning rate 1e-3), `BCEWithLogitsLoss`, batch size 256, 10 epochs, with a `ReduceLROnPlateau` scheduler (factor 0.5, patience 2). Batches are padded dynamically and packed by length.

**Inference (v1).** The softmax output at each hidden position is renormalised after zeroing letters already guessed, and the single most probable (position, letter) pair is guessed. The first move, when nothing is revealed, uses a fixed vowel-first order. The strategy lives in `HangmanRNN.guess` in `hangman/game.py`.

**v2 changes.**

- *Input:* a 26-dim multi-hot vector of the guessed letters that are not in the word is concatenated to the one-hot letter at every timestep (embedding input 52 instead of 26; v1 weights are copied in and the new columns start at zero).
- *Training states:* for each word, letters are guessed in a noisy letter-frequency order (Plackett-Luce sampling with weights frequency^alpha; a quarter of games use a uniformly random order), and one state before the game ends (at most 5 wrong guesses, word not complete) is sampled per word per epoch.
- *Loss:* softmax cross-entropy over the letters not yet guessed, at hidden positions only.
- *Optimisation:* Adam, batch size 256, cosine learning-rate decay; 5,000 words of `words_train.txt` are held out for validation.
- *Decoding:* `sum` (see Results). `HangmanRNN` also implements `noisy_or` and a length-conditioned opening, which did not help and are off by default.

## Engine rules

- Secret words are drawn from `data/words_test.txt`, which shares no words with the training dictionary.
- Each guess reveals every occurrence of the letter; a wrong guess costs one of 6 lives.
- Guessing a letter twice ends the game as a loss, so a strategy cannot loop forever.
- `seed` makes the sequence of secret words reproducible, so strategies can be compared on identical games.
