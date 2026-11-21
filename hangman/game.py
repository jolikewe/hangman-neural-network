"""Hangman game loop and guessing strategies.

HangmanBasic guesses the most frequent letter among dictionary words that match
the current pattern. HangmanRNN asks a trained network for the most likely
letter among the masked positions.
"""

import collections
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_WORDS_PATH = DATA_DIR / "words_train.txt"
TEST_WORDS_PATH = DATA_DIR / "words_test.txt"

MAX_TRIES = 6

# Opening guesses used by HangmanRNN when no letter is revealed yet.
OPENING_VOWELS = ["e", "a", "o", "i", "u"]
OPENING_CONSONANTS = [
    "t", "n", "s", "h", "r", "d", "l", "b", "c", "f", "g",
    "j", "k", "m", "p", "q", "v", "w", "x", "y", "z",
]


class HangmanBasic:
    """Hangman simulator with a dictionary-filtering letter-frequency strategy."""

    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout
        self.guessed_letters: List[str] = []

        self.test_words = self.build_wordlist(TEST_WORDS_PATH)
        self.full_dictionary = self.build_wordlist(TRAIN_WORDS_PATH)
        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.full_dictionary)
        ).most_common()
        self.current_dictionary: List[str] = []

    # ------------------------------------------------------------------
    # Guessing
    # ------------------------------------------------------------------
    def guess(self, word: str) -> str:
        """Return the next letter for a masked word such as "_ p p _ e "."""
        # strip the space characters; replace "_" with "." (regex wildcard)
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)

        # narrow the candidate dictionary to words matching the pattern
        self.current_dictionary = [
            dict_word
            for dict_word in self.current_dictionary
            if len(dict_word) == len_word and re.match(clean_word, dict_word)
        ]

        # most frequent unguessed letter among the remaining candidates
        letter_counts = collections.Counter("".join(self.current_dictionary)).most_common()
        guess_letter = self._first_unguessed(letter_counts)

        # no candidate left: fall back to letter frequencies of the full dictionary
        if guess_letter == "!":
            guess_letter = self._first_unguessed(self.full_dictionary_common_letter_sorted)
        return guess_letter

    def _first_unguessed(self, sorted_letter_counts: List[Tuple[str, int]]) -> str:
        for letter, _ in sorted_letter_counts:
            if letter not in self.guessed_letters:
                return letter
        return "!"

    # ------------------------------------------------------------------
    # Game mechanics
    # ------------------------------------------------------------------
    @staticmethod
    def build_wordlist(file_path: Union[str, Path]) -> List[str]:
        with open(file_path, "r") as file:
            return file.read().split("\n")

    def get_test_word(self) -> Tuple[int, str]:
        idx = random.randint(0, len(self.test_words) - 1)
        return idx, self.test_words[idx]

    def mask_word(self, test_word: str) -> str:
        masked = ["_" if ch not in self.guessed_letters else ch for ch in test_word]
        return " ".join(masked)

    def check_guesses(self, test_word: str) -> bool:
        """True when every letter of `test_word` has been guessed."""
        return "_" not in self.mask_word(test_word)

    def start_game(self, verbose: bool = True, show_test_word: bool = False) -> bool:
        """Play one game on a random test word. Returns True if the word was solved."""
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        game_id, test_word = self.get_test_word()
        masked_word = self.mask_word(test_word)
        tries_remaining = MAX_TRIES
        status = "fail"
        reason = ""

        if verbose:
            print("\nNew game starting.")
            print(f"Game ID: {game_id} | Total tries: {tries_remaining} | Word: {masked_word}\n")

        while tries_remaining > 0:
            guess_letter = self.guess(masked_word)

            # end the game on a repeated guess, to prevent an infinite loop
            if guess_letter in self.guessed_letters:
                reason = "Guessed repeated letter."
                break
            # only wrong guesses cost a try
            if guess_letter not in test_word:
                tries_remaining -= 1

            self.guessed_letters.append(guess_letter)
            masked_word = self.mask_word(test_word)
            if verbose:
                print(
                    f"Guessing letter: {guess_letter} | "
                    f"Tries remaining: {tries_remaining} | Word: {masked_word}"
                )

            if self.check_guesses(test_word):
                status = "pass"
                break
        else:
            reason = "# of tries exceeded"

        if status == "pass":
            if verbose:
                print(f"\nSuccessfully finished game: {game_id}.\n")
            return True

        if verbose:
            message = f"\nFailed game: {game_id} | Reason: {reason}"
            if show_test_word:
                message += f" | Correct word: {test_word}"
            print(message + "\n")
        return False

    def measure_performance(self, num_runs: int = 10) -> None:
        """Play `num_runs` games and print the success rate."""
        results = [
            self.start_game(verbose=False, show_test_word=False)
            for _ in tqdm(range(num_runs), desc="Running...")
        ]
        print(f"Total games ran: {num_runs} \nSuccess rate: {np.mean(results):.2f}\n")


class HangmanRNN(HangmanBasic):
    """Hangman simulator whose guesses come from a trained sequence model."""

    def __init__(self, timeout: Optional[int], model: torch.nn.Module):
        super().__init__(timeout)
        self.model = model

        all_chars = list(string.ascii_lowercase)
        self.all_chars_len = len(all_chars)
        self.char_to_idx = {c: i for i, c in enumerate(all_chars)}
        self.idx_to_char = {i: c for i, c in enumerate(all_chars)}

    def encode_word(self, word: str, all_chars_len: int = 26) -> torch.Tensor:
        """One-hot encode a masked word; "_" positions stay all-zero. Shape [len, 26]."""
        encoded_word = torch.zeros((len(word), all_chars_len))
        for i, char in enumerate(word):
            if char == "_":
                continue
            encoded_word[i, self.char_to_idx[char]] = 1
        return encoded_word

    def _opening_guess(self) -> Optional[str]:
        """First-move heuristic: unguessed vowels first, then consonants by frequency."""
        for letter in OPENING_VOWELS + OPENING_CONSONANTS:
            if letter not in self.guessed_letters:
                return letter
        return None

    def guess(self, masked_word: str, return_probs: bool = False) -> Union[str, Dict]:
        """Pick the next letter for a masked word such as "_ _ _ a _".

        With `return_probs=True`, return a dict with the letter ('char'), the
        position(s) of the best prediction ('pos') and the per-position
        probability matrix ('probs_norm'); the latter two are None for opening
        guesses.
        """
        stripped = masked_word.replace(" ", "")

        if all(c == "_" for c in stripped):
            opening = self._opening_guess()
            if opening is not None:
                if return_probs:
                    return {"char": opening, "pos": None, "probs_norm": None}
                return opening

        word_len = len(stripped)
        x_input = self.encode_word(stripped)

        self.model.eval()
        with torch.no_grad():
            lengths = torch.tensor([word_len])  # batch of 1
            logits = self.model(x_input.unsqueeze(0), lengths)
            probs = torch.softmax(logits[0], dim=-1)  # [word_len, all_chars_len]

            # 1. zero out positions that are already revealed
            known_positions_mask = x_input.sum(dim=1) > 0
            probs[known_positions_mask] = 0.0

            # 2. zero out previously guessed letters
            if self.guessed_letters:
                guessed_char_idx = torch.tensor([self.char_to_idx.get(x) for x in self.guessed_letters])
                unknown_positions = torch.tensor(
                    [i for i in range(word_len) if not known_positions_mask[i]]
                )
                probs[unknown_positions[:, None], guessed_char_idx] = 0

            # 3. normalize probabilities within each position
            row_sums = probs.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
            probs_normalized = probs / row_sums

            # 4. pick the highest probability over all positions and letters
            pos, char = torch.where(probs_normalized == probs_normalized.max())
            guessed_char = self.idx_to_char[char[0].item()]

        if return_probs:
            return {"char": guessed_char, "pos": pos, "probs_norm": probs_normalized}
        return guessed_char


def main() -> None:
    """Benchmark the baseline and the RNN strategy on the held-out test words."""
    from hangman.models import load_model

    game_basic = HangmanBasic(timeout=2000)
    game_basic.start_game(verbose=True, show_test_word=True)
    game_basic.measure_performance(num_runs=100)

    game_rnn = HangmanRNN(timeout=2000, model=load_model())
    game_rnn.start_game(verbose=True, show_test_word=True)
    game_rnn.measure_performance(num_runs=1000)


if __name__ == "__main__":
    main()
