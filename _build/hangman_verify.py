"""Generate PyTorch reference cases for each shipped model; check with node hangman_verify.js.

    python _build/hangman_verify.py /path/to/repo && node _build/hangman_verify.js
"""
import sys, json, random, torch
repo = sys.argv[1] if len(sys.argv) > 1 else '../hangman-neural-network-main'
sys.path.insert(0, repo)
from hangman.models import load_model
from hangman.game import HangmanRNN

CONFIGS = {  # name -> (weights file, decoding)
    'v1': ('model_rnn.pth', 'cell'),
    'v1_sum': ('model_rnn.pth', 'sum'),
    'v2': ('model_rnn_v2.pth', 'sum'),
}
words = [w.strip() for w in open(repo + '/data/words_test.txt') if w.strip().isalpha()]
out = {}
for name, (file, decoding) in CONFIGS.items():
    random.seed(3)
    m = load_model(repo + '/models/' + file)
    p = HangmanRNN(timeout=None, model=m, decoding=decoding, opening='fixed')
    cases = []
    while len(cases) < 60:
        w = random.choice(words)
        if not 2 <= len(w) <= 20: continue
        k = random.choice([0, 1, 2, 3, 4, 6, 8, 10])
        g = random.sample('abcdefghijklmnopqrstuvwxyz', k) if k else []
        if random.random() < .5: g = [c for c in 'eaoitnsr' if random.random() < .6]
        masked = ' '.join(c if c in g else '_' for c in w)
        if '_' not in masked: continue
        p.guessed_letters = list(g)
        stripped = masked.replace(' ', '')
        with torch.no_grad():
            x = p.encode_word(stripped)
            if m.use_missed:
                lg = m(x.unsqueeze(0), torch.tensor([len(stripped)]), p._missed_vector(stripped))[0]
            else:
                lg = m(x.unsqueeze(0), torch.tensor([len(stripped)]))[0]
        r = p.guess(masked, return_probs=True)
        pr = r['probs_norm']
        cases.append({'masked': masked, 'guessed': g, 'logits': lg.tolist(), 'char': r['char'],
                      'probs': pr.tolist() if pr is not None else None, 'opening': pr is None})
    out[name] = cases
    print(name, len(cases), 'cases,', sum(c['opening'] for c in cases), 'opening')
json.dump(out, open('/tmp/hangman_cases.json', 'w'))
