#!/usr/bin/env python3
"""Dlouhy trenink 1-bit transformeru presne v rezimu cipu (vysledek kapitoly 10 v docs/bitlm_poc.md):

  1-bit vahy sign(w) * posun, int8 aktivace, int8 zpetny pruchod (stochasticke zaokrouhleni),
  vsechny parametry celociselne (int16 stin), optimalizator Lion s int8 momentem, krok 2^k.

Trenink je rozdeleny na useky (GitHub Actions: max. 6 h na ulohu): kazdy usek nacte checkpoint,
trenuje do --time-limit sekund a checkpoint ulozi. Po poslednim kroku udela uplnou evaluaci
nejlepsiho modelu, ukazku textu a zapise vysledek do --out.

Pouziti: train_big.py data/cz_big.tok --d 256 --layers 6 --heads 8 --ff 1024 --steps 60000 \
             --ckpt out/ckpt.pt --time-limit 19000 --out out/big.json
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concept_lab as cl  # noqa: E402
from transformer_baseline import V, evaluate, generate  # noqa: E402

CTX = 128


def opt_state(opt):
    return {"lr_shift": opt.lr_shift,
            "groups": [{k: v for k, v in g.items() if k != "p"} for g in opt.groups]}


def load_opt_state(opt, st):
    opt.lr_shift = st["lr_shift"]
    assert len(st["groups"]) == len(opt.groups)
    for g, s in zip(opt.groups, st["groups"]):
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                g[k].copy_(v)
            else:
                g[k] = v
        g["p"].data.copy_(g["q"] * g["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-shift", type=int, default=4)
    ap.add_argument("--mbits", type=int, default=8)
    ap.add_argument("--gq", type=int, default=8)
    ap.add_argument("--val-len", type=int, default=11152)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--time-limit", type=float, default=1e12, help="sekund, pak ulozit checkpoint a skoncit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cl.CHIP, cl.GQ = True, a.gq
    torch.manual_seed(a.seed)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    raw = open(a.corpus, "rb").read()
    data = (torch.frombuffer(bytearray(raw), dtype=torch.uint8) & 31).long()
    split = len(data) - a.val_len
    model = cl.BitGPT(CTX, a.d, a.layers, a.heads, a.ff, "binary")
    n_params = sum(p.numel() for p in model.parameters())
    opt = cl.ChipOpt(model, "lion", a.lr_shift, mbits=a.mbits)
    print(f"model d={a.d} vrstev={a.layers} hlav={a.heads} ff={a.ff}: {n_params} parametru; "
          f"korpus {len(data)} tokenu (trenink {split}); lion lr_shift {a.lr_shift} mbits {a.mbits} gq {a.gq}",
          flush=True)

    start, log, best, train_s = 1, [], None, 0.0
    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, weights_only=False)
        model.load_state_dict(ck["model"])
        load_opt_state(opt, ck["opt"])
        torch.set_rng_state(ck["rng"])
        start, log, best, train_s = ck["step"] + 1, ck["log"], ck["best"], ck["train_s"]
        print(f"pokracuji od kroku {start} (dosud {train_s / 3600:.2f} h, nejlepsi {best[0] if best else None})",
              flush=True)

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt_state(opt), "rng": torch.get_rng_state(),
                    "step": step, "log": log, "best": best, "train_s": train_s, "args": vars(a)}, a.ckpt + ".tmp")
        os.replace(a.ckpt + ".tmp", a.ckpt)

    t0 = time.time()
    t_seg = t0
    step = start - 1
    for step in range(start, a.steps + 1):
        if step in (a.steps // 2, a.steps * 3 // 4, a.steps * 7 // 8):
            opt.lr_shift -= 1
        ix = torch.randint(CTX, split - 1, (a.batch,)).tolist()
        x = torch.stack([data[i - CTX:i] for i in ix])
        y = torch.stack([data[i - CTX + 1:i + 1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            now = time.time()
            train_s += now - t_seg
            t_seg = now
            ce, acc = cl.evaluate_stride(model, data, split, CTX, 4)
            tok = step * a.batch * CTX
            log.append({"step": step, "tokens": tok, "train_loss_bits": loss.item() / math.log(2),
                        "val_ce_bits": ce, "val_acc": acc, "hours": train_s / 3600})
            if best is None or ce < best[0]:
                best = (ce, step, copy.deepcopy(model.state_dict()))
            print(f"{step:7d}  tokenu {tok / 1e6:8.1f} M  train {loss.item() / math.log(2):.3f}  "
                  f"val CE(1/4) {ce:.4f}  acc {acc:.4f}  lr_shift {opt.lr_shift}  ({train_s / 3600:.2f} h)",
                  flush=True)
            save(step)
            if time.time() - t0 > a.time_limit and step < a.steps:
                print(f"casovy limit useku, checkpoint v kroku {step}", flush=True)
                return

    if step < a.steps:
        return
    model.load_state_dict(best[2])
    ce, acc = evaluate(model, data, split, CTX, bs=256)
    g = torch.Generator().manual_seed(7)
    prompt = data[split:split + CTX].tolist()
    sample = generate(model, prompt, 600, CTX, g)
    torch.save(best[2], os.path.splitext(a.out)[0] + "_model.pt")
    res = {"model": "bitnet1-chip-lion", "params": n_params, "d": a.d, "layers": a.layers, "heads": a.heads,
           "ff": a.ff, "ctx": CTX, "steps": a.steps, "batch": a.batch, "corpus_tokens": len(data),
           "lr_shift": a.lr_shift, "mbits": a.mbits, "gq": a.gq, "best_step": best[1],
           "val_ce_bits": ce, "val_acc": acc, "train_hours": train_s / 3600, "sample": sample, "log": log}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"VYSLEDEK: CE {ce:.4f} acc {acc:.4f} (nejlepsi krok {best[1]}, {n_params} parametru)")
    print("ukazka:", sample[:300])


if __name__ == "__main__":
    main()
