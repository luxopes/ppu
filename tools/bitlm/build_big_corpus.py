#!/usr/bin/env python3
"""Velky cesky korpus: ceska Wikipedie (proza) + ceske knihy (Gutenberg) + R.U.R. na konci.

Vystup: tokeny [wikipedie][knihy x opakovani][R.U.R. cely]; validace = poslednich 10 % R.U.R.
(11 152 tokenu, stejna validace jako u vsech predchozich mereni).

Z Wikipedie bereme jen odstavce, ktere jsou souvislou prozou: bez cislic (cislice slovnik
nema a jejich vypusteni by kazilo vety), alespon 200 znaku a s malym podilem znaku mimo slovnik.

Pouziti: build_big_corpus.py vystup.tok --wiki cs0.parquet [cs1.parquet ...] --wiki-chars 60000000 \
             --book-repeat 4 rur.txt kniha1.txt ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_corpus as pc  # noqa: E402

pc.MAP[":"] = pc.MAP["."]   # stejne jako build_mix_corpus.py


def wiki_paragraphs(paths):
    import pyarrow.parquet as pq
    for p in paths:
        f = pq.ParquetFile(p)
        for batch in f.iter_batches(batch_size=2000, columns=["text"]):
            for text in batch.column(0).to_pylist():
                for par in text.split("\n"):
                    par = par.strip()
                    if len(par) < 200 or any(ch.isdigit() for ch in par):
                        continue
                    yield par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("rur")
    ap.add_argument("books", nargs="*")
    ap.add_argument("--wiki", nargs="*", default=[])
    ap.add_argument("--wiki-chars", type=int, default=60_000_000)
    ap.add_argument("--book-repeat", type=int, default=4)
    a = ap.parse_args()

    wiki = bytearray()
    kept = dropped = 0
    for par in wiki_paragraphs(a.wiki):
        t = pc.tokenize(par)
        if len(t) < 0.9 * len(par):      # hodne znaku mimo slovnik (cizi pismo, vzorce)
            dropped += 1
            continue
        wiki += t + b"\x00"
        kept += 1
        if len(wiki) >= a.wiki_chars:
            break
    print(f"wikipedie: {len(wiki)} tokenu ({kept} odstavcu, {dropped} vyrazeno)")

    books = bytearray()
    for b in a.books:
        with open(b, encoding="utf-8") as f:
            t = pc.tokenize(pc.strip_gutenberg(f.read()))
        print(f"{os.path.basename(b)}: {len(t)} tokenu")
        books += t + b"\x00"
    with open(a.rur, encoding="utf-8") as f:
        r = pc.tokenize(pc.strip_gutenberg(f.read()))
    val_len = len(r) - len(r) * 9 // 10
    # R.U.R. bez validacni casti se opakuje s knihami; cele R.U.R. je na konci (validace = jeho konec)
    head = r[:len(r) - val_len] + b"\x00"
    data = bytes(wiki) + (bytes(books) + head) * (a.book_repeat - 1) + bytes(books) + r
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"celkem {len(data)} tokenu, trenink {len(data) - val_len}, validace {val_len} (konec R.U.R.)")
    print("ukazka wiki:", "".join(pc.VOCAB[t] for t in wiki[100000:100300]))


if __name__ == "__main__":
    main()
