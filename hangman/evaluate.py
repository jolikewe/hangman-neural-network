"""Benchmark guessing strategies against the self-hosted engine.

    python -m hangman.evaluate --strategy all --games 2000 --seed 0 1
    python -m hangman.evaluate --strategy rnn --model models/model_rnn.pth --decoding noisy_or --opening freq
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from hangman.engine import GameState, HangmanEngine
from hangman.game import DECODINGS, OPENINGS, V2_DECODING, V2_OPENING, HangmanBasic, HangmanRNN
from hangman.models import DEFAULT_MODEL_PATH

torch.set_num_threads(1)  # batch-1 inference is fastest single-threaded


def play_game(player: HangmanBasic, engine: HangmanEngine, word: Optional[str] = None,
              recorded: bool = True, verbose: bool = False) -> GameState:
    """Play one game: the player only sees the masked word returned by the engine."""
    player.guessed_letters = []
    player.current_dictionary = player.full_dictionary
    state = engine.new_game(word=word, recorded=recorded)
    if verbose:
        print(f"Game {state.game_id} | tries: {state.tries_remaining} | word: {state.masked_word}")

    while state.status == "ongoing":
        letter = player.guess(state.masked_word)
        state = engine.guess(state.game_id, letter)
        player.guessed_letters.append(letter)
        if verbose:
            print(f"  guess {letter} | tries: {state.tries_remaining} | word: {state.masked_word}")

    if verbose:
        print(f"  -> {state.status} ({state.reason}) answer: {state.word}\n")
    return state


def evaluate(player: HangmanBasic, games: int = 1000, seed: Optional[int] = 0,
             desc: str = "games", progress: bool = True,
             words: Optional[List[str]] = None) -> Dict[str, float]:
    """Play `games` recorded games on fresh random words (default: the test words)."""
    engine = HangmanEngine(words=words, seed=seed)
    start = time.perf_counter()
    for _ in tqdm(range(games), desc=desc, leave=False, disable=not progress):
        play_game(player, engine)
    stats = engine.status()
    return {
        "games": games,
        "wins": stats["recorded_wins"],
        "success_rate": stats["recorded_success_rate"],
        "seconds": time.perf_counter() - start,
    }


STRATEGIES = ("basic", "rnn-v1", "rnn-v2", "rnn", "both", "all")


def build_players(args: argparse.Namespace) -> Dict[str, HangmanBasic]:
    """Instantiate the players selected by --strategy."""
    from hangman.models import DEFAULT_MODEL_PATH, V2_MODEL_PATH, load_model

    wanted = {
        "basic": ["basic"], "rnn-v1": ["rnn-v1"], "rnn-v2": ["rnn-v2"], "rnn": ["rnn"],
        "both": ["basic", "rnn-v1"], "all": ["basic", "rnn-v1", "rnn-v2"],
    }[args.strategy]
    players: Dict[str, HangmanBasic] = {}
    for name in wanted:
        if name == "basic":
            players["Dictionary baseline"] = HangmanBasic()
        elif name == "rnn-v1":
            players["GRU v1"] = HangmanRNN(timeout=None, model=load_model(DEFAULT_MODEL_PATH))
        elif name == "rnn-v2":
            players["GRU v2"] = HangmanRNN(timeout=None, model=load_model(V2_MODEL_PATH),
                                           decoding=V2_DECODING, opening=V2_OPENING)
        else:
            label = f"{args.model.name} [{args.decoding}/{args.opening}]"
            players[label] = HangmanRNN(timeout=None, model=load_model(args.model),
                                        decoding=args.decoding, opening=args.opening)
    return players


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=STRATEGIES, default="rnn-v2")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, nargs="+", default=[0],
                        help="seed(s) for word selection; the same words are used for every strategy")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="weights for --strategy rnn")
    parser.add_argument("--decoding", choices=DECODINGS, default="cell", help="for --strategy rnn")
    parser.add_argument("--opening", choices=OPENINGS, default="fixed", help="for --strategy rnn")
    parser.add_argument("--val", action="store_true",
                        help="play on the held-out validation words of words_train.txt instead of the test words")
    parser.add_argument("--demo", action="store_true", help="print one verbose game first")
    args = parser.parse_args()

    words = None
    if args.val:
        from hangman.train import split_words
        words = split_words(seed=0)[1]
    players = build_players(args)

    print(f"{'Strategy':<46}{'Seed':>5}{'Games':>7}{'Wins':>7}{'Success':>10}{'Time (s)':>10}")
    for name, player in players.items():
        if args.demo:
            play_game(player, HangmanEngine(words=words, seed=args.seed[0]), recorded=False, verbose=True)
        rates = []
        for seed in args.seed:
            r = evaluate(player, games=args.games, seed=seed, desc=name, words=words)
            rates.append(r["success_rate"])
            print(f"{name:<46}{seed:>5}{r['games']:>7}{r['wins']:>7}{r['success_rate']:>10.3f}{r['seconds']:>10.1f}")
        if len(rates) > 1:
            print(f"{name:<46}{'mean':>5}{'':>14}{sum(rates) / len(rates):>10.3f}")


if __name__ == "__main__":
    main()
