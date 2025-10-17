


# source venv3.9/bin/activate





file_path = "words_train_raw.txt"

with open(file_path, 'r') as file:
    words = file.read().split('\n')



def clean_word_list(words, min_len=2, max_len=30):
    import re

    cleaned = set()
    for w in words:
        w = w.strip().lower()
        # rule 1: alphabetic only
        if not re.fullmatch(r'[a-z]+', w):
            continue
        # rule 2: length constraints
        if not (min_len <= len(w) <= max_len):
            continue
        # rule 3: skip words of all identical letters
        if len(set(w)) == 1:
            continue
        cleaned.add(w)

    print(f"Number of words removed: {len(words) - len(cleaned):,}")
    print(f"Remaining words: {len(cleaned):,}")
    return sorted(cleaned)


max_word_len = 30

clean_words = clean_word_list(words)




import re
import time
import string
import random
import collections
import numpy as np
from tqdm import tqdm


class HangmanBasic(object):
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.guessed_letters = []

        words_test_path = "words_test.txt"
        self.test_words = self.build_wordlist(words_test_path)
        
        words_train_path = "words_train_raw.txt"
        self.full_dictionary = self.build_wordlist(words_train_path)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

    ## Default guess function
    def guess(self, word): # word input example: "_ p p _ e "

        # strip away the space characters; replace "_" with "." as "." for regex
        clean_word = word[::2].replace("_",".")
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter

    def build_wordlist(self, file_path):
        with open(file_path, 'r') as file:
            words = file.read().split('\n')
        return words
    
    def get_test_word(self):
        id = random.randint(0, len(self.test_words)-1)
        return id, self.test_words[id]

    def mask_word(self, test_word):
        masked = ['_' if ch not in self.guessed_letters else ch for ch in test_word]
        return ' '.join(masked)

    def check_guesses(self, test_word):
        updated = self.mask_word(test_word)
        if '_' not in updated:
            return True
        return False

    def start_game(self, verbose=True, show_test_word=False):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        game_id, test_word = self.get_test_word()
        masked_word = self.mask_word(test_word)
        tries_remaining = 6
        status = 'fail'

        if verbose:
            print("\nNew game starting.")
            print(f"Game ID: {game_id} | Total tries: {tries_remaining} | Word: {masked_word}\n")
        while tries_remaining>0:
            # get guessed letter from user code
            guess_letter = self.guess(masked_word)

            # end game if repeated guess, to prevent infinite loop
            if guess_letter in self.guessed_letters:
                status = 'fail'
                reason = "Guessed repeated letter."
                break
            # only deduct a try if wrong guess
            elif guess_letter not in test_word:
                tries_remaining -= 1
            
            # append to guessed_letters
            self.guessed_letters.append(guess_letter)

            # Update word masking and num of tries
            masked_word = self.mask_word(test_word)
            if verbose:
                print(f"Guessing letter: {guess_letter} | Tries remaining: {tries_remaining} | Word: {masked_word}")

            # all letters correctly guessed
            if self.check_guesses(test_word):
                status = 'pass'
                break
        else:
            reason = '# of tries exceeded'


        if status=="pass":
            if verbose:
                print(f"\nSuccessfully finished game: {game_id}.\n")
            return True
        elif status=="fail":
            if verbose:
                if show_test_word:
                    print(f"\nFailed game: {game_id} | Reason: {reason} | Correct word: {test_word}\n")
                else:
                    print(f"\nFailed game: {game_id} | Reason: {reason}\n")
            return False
    
    def measure_performance(self, num_runs=10):
        perf = []
        for _ in tqdm(range(num_runs), desc="Running..."):
            perf.append(self.start_game(verbose=False, show_test_word=False))
        print(f"Total games ran: {num_runs} \nSuccess rate: {np.mean(perf):.2f}\n")


game = HangmanBasic(timeout=2000)

status = game.start_game(verbose=True, show_test_word=True)

game.measure_performance(num_runs=100)







import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class customRNN(nn.Module):
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
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, chars_len)
        self.embed_dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.norm(out)
        logits = self.fc(out)
        return logits


model = customRNN(chars_len=26, embed_dim=16, hidden_dim=128, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load("model7.pth", map_location=torch.device("cpu")))
model.eval()


import os
print(f"{os.path.getsize('model7.pth') / (1024**2):.2f} MB")





class HangmanRNN(HangmanBasic):
    def __init__(self, timeout, model):
        super().__init__(timeout)
        self.model = model

        all_chars = list(string.ascii_lowercase)
        self.all_chars_len = len(all_chars)
        self.char_to_idx = {c: i for i, c in enumerate(all_chars)}
        self.idx_to_char = {i: c for i, c in enumerate(all_chars)}
    

    def encode_word(self, word, all_chars_len=26):
        encoded_word = torch.zeros((len(word), all_chars_len))

        for i, char in enumerate(word):
            if char == '_':
                continue  # Skip if masked character
            encoded_word[i, self.char_to_idx[char]] = 1
        return encoded_word


    def guess(self, masked_word):

        stripped = masked_word.replace(" ", "")
        if all(c == '_' for c in stripped):
            # First guess: prioritize vowels not yet guessed
            vowels = ['e', 'a', 'o', 'i', 'u']
            for v in vowels:
                if v not in self.guessed_letters:
                    return v
            # If all vowels are wrong, fall back to consonants (ETAOIN SHRDLU consonants)
            sorted_consonants = ['t', 'n', 's', 'h', 'r', 'd', 'l', 'b', 'c', 'f', 'g', 'j', 'k', 'm', 'p', 'q', 'v', 'w', 'x', 'y', 'z']
            for c in sorted_consonants:
                if c not in self.guessed_letters:
                    return c

        word_len = len(stripped)
        x_input = self.encode_word(stripped)

        self.model.eval()
        with torch.no_grad():
            lengths = torch.tensor([word_len])  # batch of 1
            logits = self.model(x_input.unsqueeze(0), lengths)
            probs = torch.softmax(logits[0], dim=-1)    # max_word_len x all_chars_len
            
            # 1. Zero out positions already known (non-zero in input)
            known_positions_mask = x_input.sum(dim=1) > 0  # [T]
            probs[known_positions_mask] = 0.0

            # 2. Zero out previously guessed letters
            if self.guessed_letters:
                guessed_char_idx = torch.tensor([self.char_to_idx.get(x) for x in self.guessed_letters])
                unknown_positions_mask = torch.tensor([i for i in range(word_len) if not known_positions_mask[i]])
                probs[unknown_positions_mask[:, None], guessed_char_idx] = 0

            # 3. normalize probabilities within each position
            row_sums = probs.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
            probs_normalized = probs / row_sums

            # 4. pick max probability among all positions and characters
            pos, char = torch.where(probs_normalized == probs_normalized.max())
            guessed_char = self.idx_to_char[char[0].item()]

        return guessed_char



game = HangmanRNN(timeout=2000, model=model)

status = game.start_game(verbose=True, show_test_word=True)

game.measure_performance(num_runs=1000)