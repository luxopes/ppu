#!/usr/bin/env python3
"""Sestaveni webove demonstrace BitLM (jeden HTML soubor) z namerenych dat.

Vstupy (vychozi cesty v repozitari):
  results/cz/snap_*.bin            snimky vah BitLM z treninku (bitlm_train -S)
  results/cz/snapshot_eval.txt     val CE/presnost snimku (build/eval_snapshot)
  results/cz/golden_train.txt      baseline n-gramu a ukazka textu BitLM
  results/cz/transformer_ctx*.json srovnavaci transformery (transformer_baseline.py)
  tools/bitlm/demo/verdict_cz.html shrnuti srovnani (psane rucne podle cisel)
Vystup: HTML pro Artifact (bez <html>/<head>, viz demo_template.html).

Pouziti: build_demo.py vystup.html
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/cz")
DEMO = os.path.join(ROOT, "tools/bitlm/demo")
SNAPS = [0, 2000, 10000, 50000, 200000, 1000000, 2000000]


def main():
    out = sys.argv[1]
    tmp_w = out + ".weights.json"
    subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools/bitlm/export_demo.py"), tmp_w] +
                   [os.path.join(R, f"snap_{k}.bin") for k in SNAPS], check=True)
    weights = open(tmp_w).read()
    os.remove(tmp_w)

    ev = {}
    for line in open(os.path.join(R, "snapshot_eval.txt")):
        k, ce, acc = line.split()
        ev[k] = {"ce": float(ce), "acc": float(acc)}

    g = open(os.path.join(R, "golden_train.txt")).read()
    base = dict(re.findall(r"(unigram|bigram|trigram) ([0-9.]+)", g))
    nval = int(re.search(r"validace (\d+)", g).group(1))
    bit_sample = g.split("--- generovano:\n")[1].split("\n---")[0].strip()
    bit_prompt = re.search(r'prompt \(z validace\): "([^"]*)"', g).group(1)

    t8 = json.load(open(os.path.join(R, "transformer_ctx8.json")))
    t64 = json.load(open(os.path.join(R, "transformer_ctx64.json")))
    final = ev[str(SNAPS[-1])]
    bitlm_bytes = 512 * 256 // 8 + 32 * 512 // 8 + 32 * 32   # 1-bit W1, W2 + int8 embedding
    rows = [
        {"name": "BitLM (náš čip)", "short": "Náš čip, 1 bit", "ours": True, "params": 148480,
         "bits": "1 bit", "ctx": 8, "ce": final["ce"], "acc": final["acc"], "bytes": bitlm_bytes,
         "train": "přímo v čipu"},
        {"name": "Transformer, 4 bity", "short": "Transformer 4 bity", "ours": False, "params": t8["params"],
         "bits": "4 bity", "ctx": 8, "ce": t8["int4"]["val_ce_bits"], "acc": t8["int4"]["val_acc"],
         "bytes": t8["int4"]["bytes"], "train": "procesor, float32"},
        {"name": "Transformer, 4 bity, dlouhý kontext", "short": "Transformer 4 bity, 64 znaků", "ours": False,
         "params": t64["params"], "bits": "4 bity", "ctx": 64, "ce": t64["int4"]["val_ce_bits"],
         "acc": t64["int4"]["val_acc"], "bytes": t64["int4"]["bytes"], "train": "procesor, float32"},
        {"name": "Transformer, plná přesnost (před zmenšením)", "short": "Transformer 32 bitů", "ours": False,
         "params": t8["params"], "bits": "32 bitů", "ctx": 8, "ce": t8["float32"]["val_ce_bits"],
         "acc": t8["float32"]["val_acc"], "bytes": t8["float32"]["bytes"], "train": "procesor, float32"},
    ]
    compare = {
        "rows": rows,
        "baselines": [{"name": "Trigram (statistika 3 znaků)", "ctx": "2 znaky", "ce": float(base["trigram"])},
                      {"name": "Bigram (statistika 2 znaků)", "ctx": "1 znak", "ce": float(base["bigram"])}],
        "bitlm_log_by_step": ev,
        "samples": [
            {"title": f"Náš čip (1 bit), začátek „{bit_prompt}“", "text": bit_sample[:300]},
            {"title": f"Transformer 4 bity, začátek „{t8['prompt']}“", "text": t8["int4"]["sample"][:300]},
        ],
    }

    html = open(os.path.join(DEMO, "demo_template.html")).read()
    js = open(os.path.join(DEMO, "bitlm.js")).read()
    verdict = open(os.path.join(DEMO, "verdict_cz.html")).read().strip()
    html = (html.replace("/*BITLM_JS*/", js)
                .replace("/*WEIGHTS_JSON*/", weights)
                .replace("/*COMPARE_JSON*/", json.dumps(compare, ensure_ascii=False))
                .replace("%%VERDICT%%", verdict)
                .replace("%%NVAL%%", f"{nval:,}".replace(",", " ")))
    open(out, "w").write(html)
    print(out, os.path.getsize(out), "B")
    for r in rows:
        print(f"  {r['name']:45s} CE {r['ce']:.3f}  acc {100 * r['acc']:.1f} %  {r['bytes'] / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
