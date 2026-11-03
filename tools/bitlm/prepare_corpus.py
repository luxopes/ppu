#!/usr/bin/env python3
"""Priprava korpusu pro BitLM-nano.

Prevede text (Project Gutenberg) na tokeny 0..31:
  0 = mezera (i konce radku, opakovane mezery se slucuji)
  1..26 = a..z (diakritika se odstrani, velka pismena se zmensi)
  27 '.', 28 ',', 29 '?', 30 '!', 31 '-'
Ostatni znaky se zahodi (':' a ';' se mapuji na ',').

Vystup: binarni soubor, 1 bajt na token.
Pouziti: prepare_corpus.py vstup.txt vystup.tok [--colon-dot]
  --colon-dot  ':' -> '.' (ceske drama "Helena: ..." -> "helena. ...", stejne jako
               v anglickem vydani, kde jmena postav konci teckou)
"""
import sys
import unicodedata

VOCAB = " abcdefghijklmnopqrstuvwxyz.,?!-"
assert len(VOCAB) == 32
MAP = {c: i for i, c in enumerate(VOCAB)}
MAP.update({":": MAP[","], ";": MAP[","]})


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


def tokenize(text: str) -> bytes:
    text = unicodedata.normalize("NFKD", text)
    out = bytearray()
    prev_space = True
    for ch in text:
        if unicodedata.combining(ch):
            continue
        c = ch.lower()
        if c.isspace():
            if not prev_space:
                out.append(0)
                prev_space = True
            continue
        t = MAP.get(c)
        if t is None:
            continue
        out.append(t)
        prev_space = t == 0
    return bytes(out)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    if "--colon-dot" in sys.argv:
        MAP[":"] = MAP["."]
    sys.argv[1:3] = args
    with open(sys.argv[1], encoding="utf-8") as f:
        text = strip_gutenberg(f.read())
    toks = tokenize(text)
    with open(sys.argv[2], "wb") as f:
        f.write(toks)
    print(f"{len(toks)} tokenu -> {sys.argv[2]}")
    print("ukazka:", "".join(VOCAB[t] for t in toks[2000:2200]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
