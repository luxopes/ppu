#!/usr/bin/env python3
"""Interaktivni generovani textu z natrenovaneho 1-bit modelu (TinyStories, V=256).

Rozmery modelu (d, vrstvy, ff, kontext) se zjisti z vah; pocet hlav je parametr (vychozi 8).
Prazdny vstup = nova povidka od zacatku, jinak model pokracuje v zadanem textu.

Chat (model po SFT na dialozich): ts_play.py --chat --model models/td25_model.pt [--me Child --bot Mom]
  uzivatel pise repliky za --me, model odpovida jednou replikou za --bot (do konce radku).

Pouziti: ts_play.py [--model models/ts25_model.pt] [--vocab models/ts_vocab.json] [--temp 0.7]
         ts_play.py --prompt "Once upon a time"      (jednorazove, bez interakce)
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concept_lab as cl  # noqa: E402
import prepare_tinystories as pt  # noqa: E402
import transformer_baseline as tb  # noqa: E402


def load(model_path, vocab_path, heads):
    voc = json.load(open(vocab_path))
    tb.VOCAB, tb.V = voc["tokens"], len(voc["tokens"])
    rank = {tuple(p): i for i, p in enumerate(voc["merges"])}
    cl.CHIP, cl.GQ = True, 8
    sd = torch.load(model_path, weights_only=False, map_location="cpu")
    ctx, d = sd["pos.weight"].shape
    layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    ff = sd["blocks.0.fc1.weight"].shape[0]
    model = cl.BitGPT(ctx, d, layers, heads, ff, "binary")
    model.load_state_dict(sd)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"model {os.path.basename(model_path)}: d={d} vrstev={layers} ff={ff} kontext={ctx}, {n / 1e6:.1f} M parametru",
          file=sys.stderr)
    return model, rank, ctx


def generate(model, rank, ctx, prompt, n, temp, gen):
    seq = [0] + [t for w in pt.WORD.findall(prompt) for t in pt.encode_word(w, rank)]
    sys.stdout.write(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = model(torch.tensor([seq[-ctx:]]))[0, -1].float()
            t = int(torch.multinomial(F.softmax(logits / temp, 0), 1, generator=gen))
            if t == 0:            # <eos>: konec povidky
                break
            seq.append(t)
            sys.stdout.write(tb.VOCAB[t])
            sys.stdout.flush()
    sys.stdout.write("\n")


def chat(model, rank, ctx, me, bot, n, temp, gen):
    """Rozhovor: historie se drzi jako text, do modelu jde poslednich ctx tokenu."""
    hist = [0]
    enc = lambda s: [t for w in pt.WORD.findall(s) for t in pt.encode_word(w, rank)]  # noqa: E731
    nl = tb.VOCAB.index("\n")
    print(f"Pises za {me}, model odpovida za {bot}. Ctrl+D = konec, /new = novy rozhovor.", file=sys.stderr)
    while True:
        try:
            u = input(f"\n{me}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if u == "/new":
            hist = [0]
            continue
        hist += enc(f"{me}: {u}\n{bot}:")
        sys.stdout.write(f"{bot}:")
        with torch.no_grad():
            for _ in range(n):
                logits = model(torch.tensor([hist[-ctx:]]))[0, -1].float()
                t = int(torch.multinomial(F.softmax(logits / temp, 0), 1, generator=gen))
                if t == 0:
                    t = nl
                hist.append(t)
                if t == nl or "\n" in tb.VOCAB[t]:
                    break
                sys.stdout.write(tb.VOCAB[t])
                sys.stdout.flush()
        sys.stdout.write("\n")


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(root, "ts25_model.pt"))
    ap.add_argument("--vocab", default=os.path.join(root, "ts_vocab.json"))
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7, help="teplota (nizsi = opatrnejsi, vyssi = kreativnejsi)")
    ap.add_argument("--len", type=int, default=500, help="max. pocet tokenu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true", help="rozhovor (model po SFT na dialozich)")
    ap.add_argument("--me", default=None, help="za koho pise uzivatel (vychozi Child, u modelu ua* User)")
    ap.add_argument("--bot", default=None, help="za koho odpovida model (vychozi Mom, u modelu ua* Assistant)")
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    model, rank, ctx = load(a.model, a.vocab, a.heads)
    gen = torch.Generator()
    gen.manual_seed(a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little"))
    if a.chat:
        ua = os.path.basename(a.model).startswith("ua")
        a.me = a.me or ("User" if ua else "Child")
        a.bot = a.bot or ("Assistant" if ua else "Mom")
        chat(model, rank, ctx, a.me, a.bot, a.len, a.temp, gen)
        return
    if a.prompt is not None:
        generate(model, rank, ctx, a.prompt, a.len, a.temp, gen)
        return
    print("Napis zacatek povidky anglicky (Enter = nova povidka, Ctrl+D = konec).", file=sys.stderr)
    while True:
        try:
            p = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        generate(model, rank, ctx, p.strip(), a.len, a.temp, gen)


if __name__ == "__main__":
    main()
