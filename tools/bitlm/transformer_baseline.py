#!/usr/bin/env python3
"""Srovnavaci model: znakovy transformer (GPT) se stejnym poctem parametru jako BitLM.

Trenink ve float32 (AdamW) na CPU, pak inference s vahami kvantizovanymi na int4
(symetricky, po radcich, round-to-nearest; LayerNorm zustava float).
Vyhodnoceni stejne jako BitLM: validacni cast = poslednich 10 % korpusu,
krizova entropie [bit/znak] a presnost top-1 predikce dalsiho znaku
na kazde pozici validace s kontextem predchozich `ctx` znaku.
Korpus je maly a transformer se preucuje, proto se pouzije nejlepsi checkpoint
podle validace (early stopping; vybira se na validacni casti, coz transformeru
spis pomaha - pro srovnani konzervativni volba).

Pouziti (venv s torch):
  python transformer_baseline.py data/rur_cz.tok --ctx 8 --out results/cz/transformer_ctx8.json
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = " abcdefghijklmnopqrstuvwxyz.,?!-"
V = 32


def load_vocab(corpus):
    """Pokud vedle korpusu lezi <korpus bez .tok>.vocab.json (napr. prepare_tinystories.py), nastavi V a
    VOCAB (seznam retezcu tokenu). Jinak zustava puvodni slovnik 32 znaku. Vraci V."""
    import json
    global V, VOCAB
    path = corpus[:-4] + ".vocab.json" if corpus.endswith(".tok") else corpus + ".vocab.json"
    if os.path.exists(path):
        v = json.load(open(path))
        V, VOCAB = v["V"], v["tokens"]
    return V


def load_tokens(corpus):
    """Korpus jako tensor tokenu (1 bajt na token); stary slovnik 32 znaku maskuje & 31."""
    raw = open(corpus, "rb").read()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    return data & 31 if load_vocab(corpus) == 32 else data


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, ctx, d=64, layers=3, heads=4, ff=250):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.ctx = ctx

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok.weight.T   # svazane vahy vystupu s embeddingem


def quantize_int4_(model):
    """Vahy Linear a Embedding -> int4 (-8..7) se skalou po radcich; LayerNorm float."""
    n_q = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() != 2:
                continue  # LayerNorm (vaha, bias) zustava float
            s = p.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
            p.copy_(torch.clamp(torch.round(p / s), -8, 7) * s)
            n_q += p.numel()
    return n_q


@torch.no_grad()
def evaluate(model, data, split, ctx, bs=2048):
    """CE a presnost na kazde pozici validace, kontext = predchozich ctx znaku."""
    model.eval()
    positions = torch.arange(split, len(data))
    ce, ok = 0.0, 0
    for i in range(0, len(positions), bs):
        pos = positions[i:i + bs]
        idx = torch.stack([data[p - ctx:p] for p in pos.tolist()])
        logits = model(idx)[:, -1, :]
        tgt = data[pos]
        ce += F.cross_entropy(logits, tgt, reduction="sum").item()
        ok += (logits.argmax(1) == tgt).sum().item()
    model.train()
    n = len(positions)
    return ce / n / math.log(2), ok / n


@torch.no_grad()
def generate(model, prompt, n, ctx, gen):
    model.eval()
    seq = list(prompt)
    for _ in range(n):
        idx = torch.tensor([seq[-ctx:]])
        p = F.softmax(model(idx)[0, -1], dim=0)
        seq.append(int(torch.multinomial(p, 1, generator=gen)))
    model.train()
    return "".join(VOCAB[t] for t in seq[len(prompt):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--ff", type=int, default=0, help="0 = dorovnat pocet parametru BitLM (maly model)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--val-len", type=int, default=0, help="0 = poslednich 10 %% korpusu")
    ap.add_argument("--bitlm-params", type=int, default=512 * 256 + 32 * 512 + 32 * 32,
                    help="pocet parametru srovnavaneho BitLM")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(int(__import__('os').environ.get('TORCH_THREADS', '6')))
    data = load_tokens(a.corpus)
    split = len(data) - a.val_len if a.val_len else len(data) * 9 // 10
    bitlm_params = a.bitlm_params

    ff = a.ff or (250 if a.ctx == 8 else 241)
    model = GPT(a.ctx, d=a.d, layers=a.layers, heads=a.heads, ff=ff)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"transformer ctx={a.ctx} ff={ff}: {n_params} parametru (BitLM {bitlm_params}, "
          f"rozdil {100 * (n_params - bitlm_params) / bitlm_params:+.2f} %)")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log, best = [], None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ix = torch.randint(a.ctx, split - 1, (a.batch,))
        x = torch.stack([data[i - a.ctx:i] for i in ix.tolist()])
        y = torch.stack([data[i - a.ctx + 1:i + 1] for i in ix.tolist()])
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc = evaluate(model, data, split, a.ctx)
            preds = step * a.batch * a.ctx
            log.append({"step": step, "predictions": preds, "val_ce_bits": ce, "val_acc": acc})
            if best is None or ce < best[0]:
                best = (ce, step, copy.deepcopy(model.state_dict()))
            print(f"{step:6d}  predikci {preds:10d}  val CE {ce:.4f}  acc {acc:.4f}  ({time.time() - t0:.0f} s)",
                  flush=True)
    train_s = time.time() - t0

    model.load_state_dict(best[2])
    print(f"nejlepsi checkpoint: krok {best[1]} (val CE {best[0]:.4f})")
    ce_f, acc_f = evaluate(model, data, split, a.ctx)
    g = torch.Generator().manual_seed(7)
    prompt = data[split:split + a.ctx].tolist()
    sample_f = generate(model, prompt, 400, a.ctx, g)
    n_q = quantize_int4_(model)
    ce_q, acc_q = evaluate(model, data, split, a.ctx)
    g = torch.Generator().manual_seed(7)
    sample_q = generate(model, prompt, 400, a.ctx, g)
    n_float = n_params - n_q
    # int4 vahy + fp16 skala na radek + LayerNorm ve fp16
    rows = sum(p.shape[0] for p in model.parameters() if p.dim() == 2)
    bytes_int4 = n_q // 2 + rows * 2 + n_float * 2

    res = {
        "model": "transformer", "ctx": a.ctx, "d": a.d, "layers": a.layers, "heads": a.heads, "ff": ff,
        "train_tokens": len(data) - (len(data) - split), "val_len": len(data) - split,
        "params": n_params, "bitlm_params": bitlm_params,
        "train_steps": a.steps, "best_step": best[1], "batch": a.batch,
        "train_seconds_cpu": round(train_s, 1),
        "float32": {"val_ce_bits": ce_f, "val_acc": acc_f, "bytes": n_params * 4, "sample": sample_f},
        "int4": {"val_ce_bits": ce_q, "val_acc": acc_q, "bytes": bytes_int4,
                 "quantized_params": n_q, "float_params": n_float, "sample": sample_q},
        "prompt": "".join(VOCAB[t] for t in prompt),
        "log": log,
    }
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"float32: CE {ce_f:.4f} acc {acc_f:.4f} | int4: CE {ce_q:.4f} acc {acc_q:.4f} "
          f"| int4 velikost {bytes_int4} B | trenink {train_s:.0f} s")
    print("int4 ukazka:", sample_q[:200])


if __name__ == "__main__":
    main()
