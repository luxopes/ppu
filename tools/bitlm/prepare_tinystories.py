#!/usr/bin/env python3
"""Anglicky korpus TinyStories (velka i mala pismena) -> tokeny po 1 bajtu, slovnik V=256.

Slovnik: 0 = konec povidky, 1 = novy radek, 2..96 = tisknutelne ASCII (mezera..~), zbytek = nejcastejsi
spojeni (BPE) naucena na vzorku korpusu. Slova se deli predem (mezera patri k nasledujicimu slovu),
spojeni nikdy nepresahuji hranici slova. Token se vejde do bajtu, takze se na cipu nemeni rozlozeni
pameti (korpus 1 bajt na token).

Vystup: <out>.tok (tokeny), <out>.vocab.json ({"V": 256, "tokens": [...], "merges": [[a, b], ...]}).
Validace: posledni povidky korpusu (train_big.py bere konec korpusu jako validaci), --val pripoji
soubor TinyStories valid na konec.

Pouziti: prepare_tinystories.py data/ts/train.txt data/ts/valid.txt -o data/ts/ts --max-chars 600000000
"""
import argparse
import collections
import json
import re
import sys

EOS = "<|endoftext|>"
V = 256
NBASE = 97
FIX = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-",
       "…": "...", " ": " ", "\t": " "}
WORD = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9 \n]+|\n| +")


def clean(text):
    for a, b in FIX.items():
        text = text.replace(a, b)
    text = text.replace("\r", "")
    return "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 127)


def stories(path, max_chars):
    n = 0
    buf = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == EOS:
                s = clean("".join(buf)).strip()
                buf = []
                if s:
                    yield s
                    n += len(s)
                    if max_chars and n >= max_chars:
                        return
            else:
                buf.append(line)
    s = clean("".join(buf)).strip()
    if s:
        yield s


def base_id(ch):
    return 1 if ch == "\n" else ord(ch) - 32 + 2


def train_merges(word_counts, n_merges):
    words = [(list(base_id(c) for c in w), c) for w, c in word_counts.items()]
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for seq, c in words:
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        new = NBASE + len(merges)
        merges.append((a, b))
        for i, (seq, c) in enumerate(words):
            if len(seq) < 2:
                continue
            out, j = [], 0
            while j < len(seq):
                if j + 1 < len(seq) and seq[j] == a and seq[j + 1] == b:
                    out.append(new)
                    j += 2
                else:
                    out.append(seq[j])
                    j += 1
            words[i] = (out, c)
    return merges


def encode_word(w, rank):
    seq = [base_id(c) for c in w]
    while len(seq) > 1:
        best, bi = None, -1
        for i in range(len(seq) - 1):
            r = rank.get((seq[i], seq[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        seq = seq[:bi] + [NBASE + best] + seq[bi + 2:]
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("val", nargs="?")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max-chars", type=int, default=0, help="0 = cely soubor")
    ap.add_argument("--val-chars", type=int, default=200_000, help="velikost validace (znaku)")
    ap.add_argument("--bpe-chars", type=int, default=50_000_000, help="vzorek pro uceni spojeni")
    a = ap.parse_args()

    # 1) spojeni z vzorku
    wc = collections.Counter()
    for s in stories(a.train, a.bpe_chars):
        wc.update(WORD.findall(s))
    merges = train_merges(wc, V - NBASE)
    rank = {p: i for i, p in enumerate(merges)}
    tokens = ["<eos>", "\n"] + [chr(c) for c in range(32, 127)]
    for x, y in merges:
        tokens.append(tokens[x] + tokens[y])

    # 2) kodovani (cache po slovech)
    cache = {}
    out = bytearray()
    nch = 0

    def put(s):
        nonlocal nch
        for w in WORD.findall(s):
            t = cache.get(w)
            if t is None:
                t = cache[w] = bytes(encode_word(w, rank))
            out.extend(t)
        out.append(0)
        nch += len(s) + 1

    n_tr = 0
    for s in stories(a.train, a.max_chars):
        put(s)
        n_tr += 1
    tr_tok = len(out)
    n_va = 0
    if a.val:
        for s in stories(a.val, a.val_chars):
            put(s)
            n_va += 1
    open(a.out + ".tok", "wb").write(out)
    json.dump({"V": V, "tokens": tokens, "merges": merges, "train_tokens": tr_tok,
               "val_tokens": len(out) - tr_tok, "chars": nch}, open(a.out + ".vocab.json", "w"))
    print(f"povidek {n_tr} + val {n_va}, znaku {nch}, tokenu {len(out)} (val {len(out) - tr_tok}), "
          f"{nch / len(out):.2f} znaku/token", file=sys.stderr)
    print("nejdelsi tokeny:", sorted(tokens[NBASE:], key=len)[-12:], file=sys.stderr)


if __name__ == "__main__":
    main()
