// Test: JS dopredny pruchod == C referencni model (z[] a argmax) pro vsechny snimky.
// Pouziti: node test_bitlm_js.cjs demo_weights.json vectors_<step>.json ...
const fs = require("fs");
const BitLM = require("./bitlm.js");
const W = JSON.parse(fs.readFileSync(process.argv[2]));
let fails = 0, checks = 0;
for (const vf of process.argv.slice(3)) {
  const step = vf.match(/(\d+)\.json$/)[1];
  const m = BitLM.decode(W.snaps[step]);
  for (const t of JSON.parse(fs.readFileSync(vf))) {
    const r = BitLM.forward(m, t.ctx);
    const same = r.argmax === t.argmax && t.z.every((v, i) => v === r.z[i]);
    checks++;
    if (!same) { fails++; console.log("NESHODA krok", step, "ctx", t.ctx, "C", t.z.slice(0, 6), "JS", Array.from(r.z.slice(0, 6))); }
  }
}
// kodovani textu stejne jako prepare_corpus.py
const enc = BitLM.decodeText(BitLM.encode("Helena: Žluťoučký kůň;  úpěl!"));
if (enc !== "helena. zlutoucky kun, upel!") { fails++; console.log("encode:", JSON.stringify(enc)); }
console.log(fails ? `JS TEST SELHAL (${fails} z ${checks + 1})` : `JS TEST OK (${checks} vektoru + kodovani textu)`);
process.exit(fails ? 1 : 0);
