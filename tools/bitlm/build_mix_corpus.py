#!/usr/bin/env python3
"""Smiseny cesky korpus: ostatni ceske knihy (Project Gutenberg) + R.U.R.

Vystup: tokeny [knihy ...][R.U.R. cely]; validace = poslednich 10 % R.U.R.
(stejna validacni cast jako u modelu trenovanych jen na R.U.R.).
Delka validace se vypise; treninkove nastroje ji dostanou parametrem
(bitlm_train -v, transformer_baseline.py --val-len).

Pouziti: build_mix_corpus.py vystup.tok rur.txt kniha1.txt kniha2.txt ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_corpus as pc  # noqa: E402

pc.MAP[":"] = pc.MAP["."]   # stejne jako --colon-dot


def tok(path):
    with open(path, encoding="utf-8") as f:
        return pc.tokenize(pc.strip_gutenberg(f.read()))


def main():
    out, rur, books = sys.argv[1], sys.argv[2], sys.argv[3:]
    parts = []
    for b in books:
        t = tok(b)
        print(f"{os.path.basename(b)}: {len(t)} tokenu")
        parts.append(t + b"\x00")
    r = tok(rur)
    val_len = len(r) - len(r) * 9 // 10
    data = b"".join(parts) + r
    open(out, "wb").write(data)
    print(f"celkem {len(data)} tokenu, trenink {len(data) - val_len}, validace {val_len} (konec R.U.R.)")


if __name__ == "__main__":
    main()
