"""Export model weights (and a sample of test words) for the page.

Run from the page folder:  python _build/hangman_export.py /path/to/hangman-neural-network
Writes weights.bin/json (v1) and weights_v2.bin/json (v2), float32 little-endian.
"""
import sys, json, random
repo = sys.argv[1] if len(sys.argv) > 1 else '../hangman-neural-network-main'
sys.path.insert(0, repo)
from hangman.models import load_model

for name, path in [('weights', repo + '/models/model_rnn.pth'), ('weights_v2', repo + '/models/model_rnn_v2.pth')]:
    sd = load_model(path).state_dict()
    man, buf, off = [], bytearray(), 0
    for k, v in sd.items():
        a = v.detach().numpy().astype('<f4')
        man.append({'name': k, 'shape': list(a.shape), 'offset': off, 'length': a.size})
        buf += a.tobytes(); off += a.size
    open(name + '.bin', 'wb').write(buf)
    json.dump({'dtype': 'float32', 'endian': 'little', 'tensors': man, 'params': off}, open(name + '.json', 'w'))
    print(name, off, 'params')

random.seed(7)
w = [x.strip() for x in open(repo + '/data/words_test.txt') if x.strip().isalpha() and 2 <= len(x.strip()) <= 20]
json.dump(random.sample(w, 3000), open('words.json', 'w'), separators=(',', ':'))
