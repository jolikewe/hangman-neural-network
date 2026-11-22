// Compare the JS port with the PyTorch reference cases for every shipped model.
const fs = require('fs'), path = require('path');
const M = require('../hangman-nn.js');
const root = path.join(__dirname, '..');
const cases = JSON.parse(fs.readFileSync('/tmp/hangman_cases.json'));
const SETUP = { v1: ['weights', 'cell'], v1_sum: ['weights', 'sum'], v2: ['weights_v2', 'sum'] };
let ok = true;
for (const name of Object.keys(SETUP)) {
  const [file, decoding] = SETUP[name];
  const man = JSON.parse(fs.readFileSync(path.join(root, file + '.json')));
  const b = fs.readFileSync(path.join(root, file + '.bin'));
  const m = M.fromBuffer(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), man, decoding);
  let maxL = 0, maxP = 0, match = 0, n = 0;
  for (const c of cases[name]) {
    const r = m.guess(c.masked, c.guessed); n++;
    if (r.char === c.char) match++; else console.log('MISMATCH', name, c.masked, c.guessed.join(''), c.char, r.char);
    if (r.logits) {
      c.logits.forEach((row, i) => row.forEach((v, j) => maxL = Math.max(maxL, Math.abs(v - r.logits[i][j]))));
      c.probs.forEach((row, i) => row.forEach((v, j) => maxP = Math.max(maxP, Math.abs(v - r.probs[i][j]))));
    }
  }
  console.log(`${name}: cases=${n} letters matched=${match}/${n} max|logit err|=${maxL.toExponential(3)} max|prob err|=${maxP.toExponential(3)}`);
  if (!(maxL < 1e-4 && match === n)) ok = false;
}
process.exit(ok ? 0 : 1);
