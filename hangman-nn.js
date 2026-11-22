/* Vanilla-JS forward pass of CustomNN (hangman/models.py) and HangmanRNN.guess (hangman/game.py).
   Embedding Linear (26 or 52 -> 16), 2-layer bidirectional GRU (hidden 128), LayerNorm(256), Linear 256->26.
   v2 weights take a 52-dim input: the one-hot letter plus a 26-dim multi-hot of wrong guesses.
   decoding: 'cell' (single best position/letter cell, v1) or 'sum' (letter with the largest summed
   probability over hidden slots, v2). Opening guesses (nothing revealed) use the fixed vowel-first order. */
(function (root) {
  'use strict';
  var H = 128, V = 26, E = 16;
  var OPEN = 'eaoiu'.split('').concat('tnshrdlbcfgjkmpqvwxyz'.split(''));

  function Model(tensors, decoding) {
    this.t = tensors;
    this.inDim = tensors['embedding.weight'].length / E;
    this.useMissed = this.inDim === 2 * V;
    this.decoding = decoding || 'cell';
  }
  Model.prototype.withDecoding = function (d) { return new Model(this.t, d); };

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  // one GRU direction over seq (array of Float64Array/Float32Array), returns array of hidden states by time index
  function gruDir(t, layer, rev, seq, inDim) {
    var sfx = '_l' + layer + (rev ? '_reverse' : '');
    var Wih = t['rnn.weight_ih' + sfx], Whh = t['rnn.weight_hh' + sfx];
    var bih = t['rnn.bias_ih' + sfx], bhh = t['rnn.bias_hh' + sfx];
    var T = seq.length, out = new Array(T), h = new Float64Array(H);
    var gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (var s = 0; s < T; s++) {
      var i = rev ? T - 1 - s : s, x = seq[i], k, j;
      for (j = 0; j < 3 * H; j++) {
        var a = bih[j], b = bhh[j], ro = j * inDim, rh = j * H;
        for (k = 0; k < inDim; k++) a += Wih[ro + k] * x[k];
        for (k = 0; k < H; k++) b += Whh[rh + k] * h[k];
        gi[j] = a; gh[j] = b;
      }
      var nh = new Float64Array(H);
      for (j = 0; j < H; j++) {
        var r = sigmoid(gi[j] + gh[j]);
        var z = sigmoid(gi[H + j] + gh[H + j]);
        var n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        nh[j] = (1 - z) * n + z * h[j];
      }
      h = nh; out[i] = h;
    }
    return out;
  }

  // masked: string of a-z and '_' (spaces ignored). Returns T arrays of 26 logits.
  Model.prototype.logits = function (masked, missed) {
    var t = this.t, w = masked.replace(/ /g, ''), T = w.length, p, k, j;
    var emb = new Array(T);
    for (p = 0; p < T; p++) {
      var e = new Float64Array(E), c = w.charCodeAt(p) - 97, hot = w[p] !== '_';
      for (j = 0; j < E; j++) {
        var ew = t['embedding.weight'], base = j * this.inDim, acc = t['embedding.bias'][j];
        if (hot) acc += ew[base + c];
        if (this.useMissed && missed) for (k = 0; k < missed.length; k++) acc += ew[base + V + missed[k]];
        e[j] = acc;
      }
      emb[p] = e;
    }
    var f0 = gruDir(t, 0, false, emb, E), b0 = gruDir(t, 0, true, emb, E), l1 = new Array(T);
    for (p = 0; p < T; p++) { var cat = new Float64Array(2 * H); cat.set(f0[p], 0); cat.set(b0[p], H); l1[p] = cat; }
    var f1 = gruDir(t, 1, false, l1, 2 * H), b1 = gruDir(t, 1, true, l1, 2 * H);
    var out = new Array(T), g = t['norm.weight'], be = t['norm.bias'], W = t['fc.weight'], fb = t['fc.bias'];
    for (p = 0; p < T; p++) {
      var y = new Float64Array(2 * H); y.set(f1[p], 0); y.set(b1[p], H);
      var mu = 0, va = 0;
      for (k = 0; k < 2 * H; k++) mu += y[k]; mu /= 2 * H;
      for (k = 0; k < 2 * H; k++) va += (y[k] - mu) * (y[k] - mu); va /= 2 * H;
      var inv = 1 / Math.sqrt(va + 1e-5);
      for (k = 0; k < 2 * H; k++) y[k] = (y[k] - mu) * inv * g[k] + be[k];
      var lg = new Float64Array(V);
      for (j = 0; j < V; j++) { var s = fb[j]; for (k = 0; k < 2 * H; k++) s += W[j * 2 * H + k] * y[k]; lg[j] = s; }
      out[p] = lg;
    }
    return out;
  };

  // Mirrors HangmanRNN.guess. guessed: array of letters already guessed.
  // Returns {char, pos, opening, probs (T x 26 renormalised, zero on revealed rows), raw (softmax), logits}
  Model.prototype.guess = function (masked, guessed) {
    var w = masked.replace(/ /g, ''), T = w.length, i, p, j;
    var allMasked = true;
    for (i = 0; i < T; i++) if (w[i] !== '_') { allMasked = false; break; }
    if (allMasked) {
      for (i = 0; i < OPEN.length; i++) if (guessed.indexOf(OPEN[i]) < 0)
        return { char: OPEN[i], pos: null, opening: true, probs: null, raw: null, logits: null };
    }
    var missed = [];
    for (j = 0; j < guessed.length; j++) if (w.indexOf(guessed[j]) < 0) missed.push(guessed[j].charCodeAt(0) - 97);
    var lg = this.logits(w, missed), raw = [], probs = [];
    for (p = 0; p < T; p++) {
      var m = -Infinity, s = 0, pr = new Float64Array(V);
      for (j = 0; j < V; j++) if (lg[p][j] > m) m = lg[p][j];
      for (j = 0; j < V; j++) { pr[j] = Math.exp(lg[p][j] - m); s += pr[j]; }
      for (j = 0; j < V; j++) pr[j] /= s;
      raw.push(Float64Array.from(pr));
      if (w[p] !== '_') pr.fill(0);
      else {
        for (j = 0; j < guessed.length; j++) pr[guessed[j].charCodeAt(0) - 97] = 0;
        var rs = 0; for (j = 0; j < V; j++) rs += pr[j]; rs += 1e-8;
        for (j = 0; j < V; j++) pr[j] /= rs;
      }
      probs.push(pr);
    }
    var best = -1, bp = 0, bc = 0, score = null;
    if (this.decoding === 'sum') {
      score = new Float64Array(V);
      for (p = 0; p < T; p++) for (j = 0; j < V; j++) score[j] += probs[p][j];
      for (j = 0; j < V; j++) if (score[j] > best) { best = score[j]; bc = j; }
      best = -1;
      for (p = 0; p < T; p++) if (probs[p][bc] > best) { best = probs[p][bc]; bp = p; }
    } else {
      for (p = 0; p < T; p++) for (j = 0; j < V; j++) if (probs[p][j] > best) { best = probs[p][j]; bp = p; bc = j; }
    }
    return { char: String.fromCharCode(97 + bc), pos: bp, opening: false, probs: probs, raw: raw, logits: lg, score: score, decoding: this.decoding };
  };

  Model.fromBuffer = function (buf, manifest, decoding) {
    var t = {};
    manifest.tensors.forEach(function (d) { t[d.name] = new Float32Array(buf, d.offset * 4, d.length); });
    return new Model(t, decoding);
  };
  Model.OPENING = OPEN;

  root.HangmanNN = Model;
  if (typeof module !== 'undefined') module.exports = Model;
})(typeof window !== 'undefined' ? window : globalThis);
