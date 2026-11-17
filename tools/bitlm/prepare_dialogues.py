#!/usr/bin/env python3
"""Korpus pro SFT na dialozich: TinyDialogues (MIT, Feng et al. 2024) + cast TinyStories (proti zapominani).

Dialog se prevede na prosty text "Mluvci: replika" po radcich (bez Markdownu), dialogy oddeluje <eos>.
Slovnik a spojeni se prevezmou z predtrenovaneho modelu (<ts>.vocab.json), aby tokeny sedely.

Vystup: <out>.tok = [TinyStories (prvnich --ts-tokens z <ts>.tok)][TinyDialogues train][validace TinyDialogues]
        <out>.vocab.json (stejny slovnik, val_tokens = delka validace)

Pouziti: prepare_dialogues.py data/ts/ts td_train.txt td_val.txt -o data/td/td --ts-tokens 30000000
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_tinystories as pt  # noqa: E402


def dialogues(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f.read().split("<|endoftext|>"):
            s = raw.replace("\\n", "\n")
            s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)        # **Mom**: -> Mom:
            s = re.sub(r"[ \t]*\n[ \t\n]*", "\n", s)            # jedna replika = jeden radek
            s = pt.clean(s).strip()
            if s.count("\n") >= 1:
                yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts", help="predtrenovany korpus bez pripony (data/ts/ts)")
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ts-tokens", type=int, default=30_000_000)
    ap.add_argument("--val-tokens", type=int, default=40_000, help="priblizna velikost validace")
    a = ap.parse_args()

    voc = json.load(open(a.ts + ".vocab.json"))
    rank = {tuple(p): i for i, p in enumerate(voc["merges"])}
    cache = {}

    def enc(s, out):
        for w in pt.WORD.findall(s):
            t = cache.get(w)
            if t is None:
                t = cache[w] = bytes(pt.encode_word(w, rank))
            out.extend(t)
        out.append(0)

    out = bytearray()
    with open(a.ts + ".tok", "rb") as f:
        ts = f.read(a.ts_tokens)
    cut = ts.rfind(b"\x00") + 1                     # konec posledni cele povidky
    out.extend(ts[:cut])
    n_ts = len(out)
    n_tr = 0
    for s in dialogues(a.train):
        enc(s, out)
        n_tr += 1
    tr_tok = len(out)
    n_va = 0
    for s in dialogues(a.val):
        enc(s, out)
        n_va += 1
        if len(out) - tr_tok >= a.val_tokens:
            break
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".tok", "wb").write(out)
    json.dump({"V": voc["V"], "tokens": voc["tokens"], "merges": voc["merges"], "train_tokens": tr_tok,
               "val_tokens": len(out) - tr_tok, "ts_tokens": n_ts}, open(a.out + ".vocab.json", "w"))
    print(f"TinyStories {n_ts} tokenu, dialogu {n_tr} ({tr_tok - n_ts} tokenu), val {n_va} dialogu "
          f"({len(out) - tr_tok} tokenu)")


if __name__ == "__main__":
    main()
