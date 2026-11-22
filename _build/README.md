# Page build scripts

Scripts that export the data shown on this page from the project's main branch and verify the JavaScript port.
With the repository cloned next to this folder (or pass its path as the argument):

```bash
git clone https://github.com/jolikewe/hangman-neural-network.git ../hangman-neural-network-main
python _build/hangman_export.py ../hangman-neural-network-main   # weights.bin/json (v1), weights_v2.bin/json (v2), words.json
python _build/hangman_verify.py ../hangman-neural-network-main   # PyTorch reference cases for v1, v1 + sum decoding, v2
node _build/hangman_verify.js                                    # needs 60/60 letters matched and logit error < 1e-4 per model
```
