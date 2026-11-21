"""Self-hosted Hangman game engine.

Replaces the external game server the project was originally evaluated on.
The engine owns the secret words and the rules; players only ever see the
masked word, exactly as they would through a remote API.

Rules: the secret word is drawn from a word list the player has never seen
(``data/words_test.txt`` is disjoint from the training dictionary). Each guess
reveals every occurrence of the letter. A wrong guess costs one of
``MAX_WRONG_GUESSES`` lives. Repeating a letter ends the game, which stops
strategies from looping forever.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_WORDS_PATH = DATA_DIR / "words_test.txt"
MAX_WRONG_GUESSES = 6


@dataclass
class GameState:
    """Public view of one game. The secret word is only exposed once the game ends."""

    game_id: str
    masked_word: str
    tries_remaining: int
    guessed_letters: List[str] = field(default_factory=list)
    status: str = "ongoing"  # "ongoing" | "success" | "failed"
    reason: str = ""
    word: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "word": self.masked_word,
            "tries_remains": self.tries_remaining,
            "guessed_letters": list(self.guessed_letters),
            "status": self.status,
            "reason": self.reason,
            "answer": self.word,
        }


@dataclass
class _Game:
    secret: str
    guessed: List[str] = field(default_factory=list)
    wrong: int = 0
    status: str = "ongoing"
    reason: str = ""
    recorded: bool = False


class HangmanEngine:
    """In-memory Hangman server.

    Typical use::

        engine = HangmanEngine(seed=0)
        state = engine.new_game()
        state = engine.guess(state.game_id, "e")
    """

    def __init__(
        self,
        words: Union[str, Path, List[str], None] = None,
        max_wrong_guesses: int = MAX_WRONG_GUESSES,
        seed: Optional[int] = None,
    ):
        if words is None:
            words = DEFAULT_WORDS_PATH
        if isinstance(words, (str, Path)):
            words = Path(words).read_text().split()
        self.words = [w.strip().lower() for w in words if w.strip().isalpha()]
        if not self.words:
            raise ValueError("Word list is empty.")
        self.max_wrong_guesses = max_wrong_guesses
        self._rng = random.Random(seed)
        self._games: Dict[str, _Game] = {}
        self.stats = {"practice_games": 0, "practice_wins": 0, "recorded_games": 0, "recorded_wins": 0}

    # ------------------------------------------------------------------
    def new_game(self, word: Optional[str] = None, recorded: bool = False) -> GameState:
        """Start a game on a random word (or on `word`, for testing)."""
        secret = (word or self._rng.choice(self.words)).lower()
        game_id = uuid.uuid4().hex[:12]
        self._games[game_id] = _Game(secret=secret, recorded=recorded)
        return self._state(game_id)

    def guess(self, game_id: str, letter: str) -> GameState:
        """Guess one letter in an ongoing game."""
        game = self._get(game_id)
        if game.status != "ongoing":
            raise ValueError(f"Game {game_id} is already finished ({game.status}).")
        letter = letter.lower()
        if len(letter) != 1 or letter not in string.ascii_lowercase:
            raise ValueError(f"Invalid guess {letter!r}: expected a single letter a-z.")

        if letter in game.guessed:
            self._finish(game, "failed", "Guessed repeated letter.")
            return self._state(game_id)

        game.guessed.append(letter)
        if letter not in game.secret:
            game.wrong += 1

        if all(ch in game.guessed for ch in game.secret):
            self._finish(game, "success", "Word solved.")
        elif game.wrong >= self.max_wrong_guesses:
            self._finish(game, "failed", "# of tries exceeded.")
        return self._state(game_id)

    def status(self) -> Dict[str, float]:
        """Aggregate results, mirroring a game server's account status."""
        s = dict(self.stats)
        for kind in ("practice", "recorded"):
            n = s[f"{kind}_games"]
            s[f"{kind}_success_rate"] = s[f"{kind}_wins"] / n if n else 0.0
        return s

    # ------------------------------------------------------------------
    def _get(self, game_id: str) -> _Game:
        try:
            return self._games[game_id]
        except KeyError:
            raise KeyError(f"Unknown game_id {game_id!r}.") from None

    def _finish(self, game: _Game, status: str, reason: str) -> None:
        game.status, game.reason = status, reason
        kind = "recorded" if game.recorded else "practice"
        self.stats[f"{kind}_games"] += 1
        self.stats[f"{kind}_wins"] += status == "success"

    def _state(self, game_id: str) -> GameState:
        game = self._games[game_id]
        masked = " ".join(ch if ch in game.guessed else "_" for ch in game.secret)
        return GameState(
            game_id=game_id,
            masked_word=masked,
            tries_remaining=self.max_wrong_guesses - game.wrong,
            guessed_letters=list(game.guessed),
            status=game.status,
            reason=game.reason,
            word=game.secret if game.status != "ongoing" else None,
        )
