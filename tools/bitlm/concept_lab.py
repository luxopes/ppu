#!/usr/bin/env python3
"""Diagnostika konceptu BitLM: co stoji za rozdilem proti transformeru?

Vsechny modely ~1,18M parametru, kontext 128, korpus cz_mix, stejna validace
(konec R.U.R., kazda pozice, kontext predchozich 128 znaku) jako
transformer_baseline.py. Trenink ve float (PyTorch) s kvantizaci ve forward
pruchodu a straight-through estimatorem (QAT, jako BitNet).

Varianty (--model):
  mlp1     architektura BitLM (viceurovnovy kontext 16 slotu x E=16, H=4096,
           1-bit vahy W1/W2, 7-bit aktivace), ale optimalizator Adam + mini-batch.
           -> kolik stoji nas zpusob uceni (online SGD, kroky jako posuny).
  bitnet1  transformer, vahy 1 bit {-1,+1} (sign(w - mean) * mean|.|), aktivace 8 bit.
  bitnet158 transformer, vahy ternarni {-1,0,+1} (BitNet b1.58, absmean), aktivace 8 bit.
Embedding (a s nim svazany vystup) je u vsech int8, LayerNorm a vnitrek attention float.

Pouziti: concept_lab.py data/cz_mix.tok --model bitnet158 --out results/cz_mix/concept_bitnet158.json
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_baseline import VOCAB, V, evaluate, generate  # noqa: E402

BITLM_PARAMS = 4096 * 256 + 32 * 4096 + 32 * 16   # E=16, H=4096, 16 slotu: 1 180 160


def ste(x, q):
    """Straight-through: dopredu q, zpet gradient jako identita."""
    return x + (q - x).detach()


def act_quant(x, bits=8):
    """Aktivace: per-token absmax na int8 (BitNet)."""
    qmax = 2 ** (bits - 1) - 1
    s = qmax / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return ste(x, torch.clamp(torch.round(x * s), -qmax - 1, qmax) / s)


def w_ternary(w):
    s = w.abs().mean().clamp_min(1e-5)
    return ste(w, torch.clamp(torch.round(w / s), -1, 1) * s)


def w_binary(w):
    c = w - w.mean()
    a = c.abs().mean()
    return ste(w, torch.sign(c) * a)


def emb_int8(w):
    s = 127.0 / w.abs().amax(dim=1, keepdim=True).clamp_min(1e-5)
    return ste(w, torch.round(w * s) / s)


class BitLinear(nn.Linear):
    def __init__(self, i, o, mode):
        super().__init__(i, o, bias=False)
        self.mode = mode

    def forward(self, x):
        w = w_ternary(self.weight) if self.mode == "ternary" else w_binary(self.weight)
        return F.linear(act_quant(x), w)


class BitBlock(nn.Module):
    def __init__(self, d, heads, ff, mode):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = BitLinear(d, 3 * d, mode)
        self.proj = BitLinear(d, d, mode)
        self.fc1 = BitLinear(d, ff, mode)
        self.fc2 = BitLinear(ff, d, mode)
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class BitGPT(nn.Module):
    def __init__(self, ctx, d, layers, heads, ff, mode):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(BitBlock(d, heads, ff, mode) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.ctx = ctx

    def forward(self, idx):
        t = idx.shape[1]
        te = emb_int8(self.tok.weight)
        x = te[idx] + emb_int8(self.pos.weight)[:t]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ te.T


# okna viceurovnoveho kontextu (stejne jako bitlm_ref.h): (zacatek, velikost)
WINDOWS = [(0, 32), (32, 32), (64, 16), (80, 16), (96, 8), (104, 8), (112, 4), (116, 4)] + \
          [(120 + i, 1) for i in range(8)]


class MLP1(nn.Module):
    """Architektura BitLM (1-bit W1, W2; viceurovnovy kontext) pro trenink Adamem."""

    def __init__(self, e=16, h=4096):
        super().__init__()
        self.emb = nn.Embedding(V, e)
        self.w1 = nn.Parameter(torch.randn(h, 16 * e) * 0.02)
        self.w2 = nn.Parameter(torch.randn(V, h) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(0.05))  # odpovida posunu s2 v HW
        self.ctx = 128

    def forward(self, idx):
        # vraci logity pro posledni pozici okna ve tvaru (B, 1, V) kvuli kompatibilite s evaluate()
        x = emb_int8(self.emb.weight)[idx[:, -self.ctx:]]               # B,128,E
        x = torch.cat([x[:, s:s + n].mean(dim=1) for s, n in WINDOWS], dim=1)   # B,16E
        h = F.linear(act_quant(x), torch.sign(self.w1) + (self.w1 - self.w1.detach()))
        a = F.relu(h)
        a = act_quant(a / a.amax(dim=-1, keepdim=True).clamp_min(1e-5), bits=8)  # 7 bit kladne
        z = F.linear(a, torch.sign(self.w2) + (self.w2 - self.w2.detach()))
        return (z * self.logit_scale).unsqueeze(1)


@torch.no_grad()
def evaluate_stride(model, data, split, ctx, stride, bs=512):
    """Prubezna evaluace na kazde stride-te pozici validace (zrychleni); finalni je uplna."""
    model.eval()
    pos = list(range(split, len(data), stride))
    ce, ok = 0.0, 0
    for i in range(0, len(pos), bs):
        p = pos[i:i + bs]
        idx = torch.stack([data[q - ctx:q] for q in p])
        logits = model(idx)[:, -1, :]
        tgt = data[p]
        ce += F.cross_entropy(logits, tgt, reduction="sum").item()
        ok += (logits.argmax(1) == tgt).sum().item()
    model.train()
    return ce / len(pos) / math.log(2), ok / len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--model", required=True, choices=["mlp1", "bitnet1", "bitnet158"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-len", type=int, default=11152)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    data = torch.tensor(list(open(a.corpus, "rb").read()), dtype=torch.long) & 31
    split = len(data) - a.val_len
    ctx = 128
    if a.model == "mlp1":
        model = MLP1()
    else:
        # d=128, 5 vrstev, ff=648 -> 1 180 416 parametru (BitLM 1 180 160)
        model = BitGPT(ctx, 128, 5, 4, 648, "ternary" if a.model == "bitnet158" else "binary")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.model}: {n_params} parametru (BitLM {BITLM_PARAMS}, {100 * (n_params - BITLM_PARAMS) / BITLM_PARAMS:+.2f} %)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log, best, t0 = [], None, time.time()
    for step in range(1, a.steps + 1):
        ix = torch.randint(ctx, split - 1, (a.batch,))
        if a.model == "mlp1":
            # 1 predikce na vzorek (jako BitLM)
            x = torch.stack([data[i - ctx:i] for i in ix.tolist()])
            loss = F.cross_entropy(model(x)[:, -1, :], data[ix])
            preds = step * a.batch
        else:
            x = torch.stack([data[i - ctx:i] for i in ix.tolist()])
            y = torch.stack([data[i - ctx + 1:i + 1] for i in ix.tolist()])
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            preds = step * a.batch * ctx
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc = evaluate_stride(model, data, split, ctx, 4)
            log.append({"step": step, "predictions": preds, "val_ce_bits": ce, "val_acc": acc})
            if best is None or ce < best[0]:
                best = (ce, step, copy.deepcopy(model.state_dict()))
            print(f"{step:6d}  predikci {preds:10d}  val CE(1/4) {ce:.4f}  acc {acc:.4f}  ({time.time() - t0:.0f} s)",
                  flush=True)
    model.load_state_dict(best[2])
    ce, acc = evaluate(model, data, split, ctx, bs=512)
    g = torch.Generator().manual_seed(7)
    prompt = data[split:split + ctx].tolist()
    sample = generate(model, prompt, 400, ctx, g)
    res = {"model": a.model, "params": n_params, "bitlm_params": BITLM_PARAMS, "ctx": ctx,
           "steps": a.steps, "batch": a.batch, "best_step": best[1], "val_ce_bits": ce, "val_acc": acc,
           "train_seconds_cpu": round(time.time() - t0, 1), "sample": sample, "log": log}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"VYSLEDEK {a.model}: CE {ce:.4f} acc {acc:.4f} (nejlepsi krok {best[1]})")
    print("ukazka:", sample[:200])


if __name__ == "__main__":
    main()
