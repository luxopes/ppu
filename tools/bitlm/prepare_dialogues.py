#!/usr/bin/env python3
"""Korpus pro SFT na dialozich: TinyDialogues (MIT, Feng et al. 2024) + cast TinyStories (proti zapominani).

Dialog se prevede na prosty text "Mluvci: replika" po radcich (bez Markdownu), dialogy oddeluje <eos>.
Slovnik a spojeni se prevezmou z predtrenovaneho modelu (<ts>.vocab.json), aby tokeny sedely.

Vystup: <out>.tok = [TinyStories (prvnich --ts-tokens z <ts>.tok)][TinyDialogues train][validace TinyDialogues]
        <out>.vocab.json (stejny slovnik, val_tokens = delka validace)

--user-assistant: jen dialogy dvou mluvcich (Child + jeden dalsi), Child -> User, druhy -> Assistant,
bez scenickych poznamek "(...)" u jmena a bez dialogu oslovujicich rodice (Mom/Dad...).

Pouziti: prepare_dialogues.py data/ts/ts td_train.txt td_val.txt -o data/td/td --ts-tokens 30000000
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_tinystories as pt  # noqa: E402


PARENT = re.compile(r"\b(Mom|Mommy|Mama|Mum|Dad|Daddy|Papa)\b")


def to_user_assistant(raw):
    sp = set(re.findall(r"\*\*([^*\n]+)\*\*", raw))
    if len(sp) != 2 or "Child" not in sp:
        return None
    other = (sp - {"Child"}).pop()
    if PARENT.search(re.sub(r"\*\*[^*\n]+\*\*", "", raw)):
        return None
    raw = re.sub(r"\*\*([^*\n]+)\*\*\s*\([^)\n]*\)", r"**\1**", raw)   # scenicke poznamky u jmena
    return raw.replace("**Child**", "**User**").replace(f"**{other}**", "**Assistant**")


def dialogues(path, user_assistant=False):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f.read().split("<|endoftext|>"):
            if user_assistant:
                raw = to_user_assistant(raw)
                if raw is None:
                    continue
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
    ap.add_argument("--user-assistant", action="store_true")
    ap.add_argument("--extra", default=None, help="dalsi trenovaci text (napr. QA z gen_qa.py), bez prevodu")
    ap.add_argument("--extra-val", default=None, help="validace z --extra misto validace dialogu")
    ap.add_argument("--extra-repeat", type=int, default=1, help="kolikrat --extra zopakovat")
    ap.add_argument("--max-train-tokens", type=int, default=0, help="omezit tokeny z hlavnich dialogu (0 = vse)")
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
    for s in dialogues(a.train, a.user_assistant):
        enc(s, out)
        n_tr += 1
        if a.max_train_tokens and len(out) - n_ts >= a.max_train_tokens:
            break
    n_main = len(out) - n_ts
    if a.extra:
        ex = list(dialogues(a.extra))
        for _ in range(a.extra_repeat):
            for s in ex:
                enc(s, out)
        print(f"extra {len(ex)} rozhovoru x{a.extra_repeat} ({len(out) - n_ts - n_main} tokenu)")
    tr_tok = len(out)
    n_va = 0
    val_src = a.extra_val or a.val
    for s in dialogues(val_src, a.user_assistant and not a.extra_val):
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
