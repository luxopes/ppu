// BitLM inference v JavaScriptu (dopredny pruchod bitove shodny s bitlm_ref.c pro
// z[] a argmax; softmax ve float s volitelnou "teplotou" pro vzorkovani).
// Pouziva se ve webovem demu a testuje se v Node proti C modelu (test_bitlm_js.mjs).
"use strict";
const BitLM = (() => {
  const VOCAB = " abcdefghijklmnopqrstuvwxyz.,?!-";
  const H = 512, D = 256, V = 32, E = 32, N = 8, S2 = 4;

  function b64ToBytes(b64) {
    const bin = (typeof atob === "function") ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // snimek: W1 bity (H*D/8) | W2 bity (V*H/8) | x int8 (V*E)
  function decode(b64) {
    const b = b64ToBytes(b64);
    const n1 = H * D / 8, n2 = V * H / 8;
    const w1 = new Int8Array(H * D), w2 = new Int8Array(V * H);
    for (let i = 0; i < H * D; i++) w1[i] = (b[i >> 3] >> (i & 7)) & 1 ? 1 : -1;
    for (let i = 0; i < V * H; i++) w2[i] = (b[n1 + (i >> 3)] >> (i & 7)) & 1 ? 1 : -1;
    const xt = new Int8Array(b.buffer.slice(n1 + n2, n1 + n2 + V * E));
    return { w1, w2, xt };
  }

  function forward(m, ctx) {
    const x = new Int32Array(D);
    for (let n = 0; n < N; n++)
      for (let e = 0; e < E; e++) x[n * E + e] = m.xt[ctx[n] * E + e];
    const h = new Int32Array(H);
    let hmax = 0;
    for (let j = 0; j < H; j++) {
      let acc = 0;
      const row = j * D;
      for (let k = 0; k < D; k++) acc += m.w1[row + k] * x[k];
      h[j] = acc;
      if (acc > hmax) hmax = acc;
    }
    let s = 0;
    while ((hmax >> s) > 127) s++;
    const a = new Int32Array(H);
    for (let j = 0; j < H; j++) {
      const t = h[j] >> s;                 // aritmeticky posun = floor
      a[j] = t < 0 ? 0 : (t > 127 ? 127 : t);
    }
    const z = new Int32Array(V);
    let argmax = 0;
    for (let i = 0; i < V; i++) {
      let acc = 0;
      const row = i * H;
      for (let j = 0; j < H; j++) acc += m.w2[row + j] * a[j];
      z[i] = acc >> S2;
      if (i === 0 || z[i] > z[argmax]) argmax = i;
    }
    return { z, argmax };
  }

  // softmax jako v HW: exp(-(zmax - z)/16), teplota T meni "kreativitu"
  function probs(z, T) {
    let zmax = -Infinity;
    for (const v of z) if (v > zmax) zmax = v;
    const p = new Float64Array(V);
    let sum = 0;
    for (let i = 0; i < V; i++) { p[i] = Math.exp(-(zmax - z[i]) / 16 / T); sum += p[i]; }
    for (let i = 0; i < V; i++) p[i] /= sum;
    return p;
  }

  function sample(p, rnd) {
    let u = rnd(), c = 0;
    for (let i = 0; i < V; i++) { c += p[i]; if (u < c) return i; }
    return V - 1;
  }

  // text -> tokeny stejne jako tools/bitlm/prepare_corpus.py --colon-dot
  function encode(text) {
    const t = text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
    const out = [];
    let prevSpace = true;
    for (const ch of t) {
      let c = ch;
      if (/\s/.test(c)) { if (!prevSpace) { out.push(0); prevSpace = true; } continue; }
      if (c === ":") c = ".";
      if (c === ";") c = ",";
      const k = VOCAB.indexOf(c);
      if (k < 0) continue;
      out.push(k);
      prevSpace = k === 0;
    }
    return out;
  }

  function decodeText(tokens) { return tokens.map(t => VOCAB[t]).join(""); }

  return { VOCAB, N, decode, forward, probs, sample, encode, decodeText };
})();
if (typeof module !== "undefined") module.exports = BitLM;
