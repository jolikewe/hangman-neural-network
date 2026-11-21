"""Benchmark guessing strategies against the self-hosted engine.

    python -m hangman.evaluate --strategy rnn --games 1000 --seed 0
    python -m hangman.evaluate --strategy both --games 500
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional

from tqdm import tqdm

from hangman.engine import GameState, HangmanEngine
from hangman.game import HangmanBasic, HangmanRNN


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
             desc: str = "games", progress: bool = True) -> Dict[str, float]:
    """Play `games` recorded games on fresh random test words and return summary stats."""
    engine = HangmanEngine(seed=seed)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["rnn", "basic", "both"], default="rnn")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed for word selection (same words for every strategy)")
    parser.add_argument("--demo", action="store_true", help="print one verbose game first")
    args = parser.parse_args()

    players = {}
    if args.strategy in ("basic", "both"):
        players["Dictionary baseline"] = HangmanBasic()
    if args.strategy in ("rnn", "both"):
        from hangman.models import load_model
        players["Bidirectional GRU"] = HangmanRNN(timeout=None, model=load_model())

    print(f"{'Strategy':<22}{'Games':>7}{'Wins':>7}{'Success':>10}{'Time (s)':>10}")
    for name, player in players.items():
        if args.demo:
            play_game(player, HangmanEngine(seed=args.seed), recorded=False, verbose=True)
        r = evaluate(player, games=args.games, seed=args.seed, desc=name)
        print(f"{name:<22}{r['games']:>7}{r['wins']:>7}{r['success_rate']:>10.3f}{r['seconds']:>10.1f}")


if __name__ == "__main__":
    main()
