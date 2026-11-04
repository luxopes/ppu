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

Uceni jako v cipu (--opt sgd|nsgd|lion): vsechny parametry jsou celociselne (int16 stin,
pevny krok = mocnina dvou, pocatecni max|p| ~ 2^12 LSB;
embedding s --chip inicializovan N(0, 0,02) jako GPT-2), update = gradient posunuty o mocninu
dvou se stochastickym zaokrouhlenim (jako BitLM v RTL), bez Adamu:
  sgd   q -= SR(g * 2^lr_shift / krok)                (pevny posun pro vsechny tenzory)
  nsgd  q -= SR(g / 2^ceil(log2 max|g|) * 2^lr_shift)  (max po tenzoru nebo radku, --norm)
  lion  q -= sign(b1*m + (1-b1)*g) * 2^lr_shift; m = b2*m + (1-b2)*g   (+1 B/vahu na moment)
  bop   1-bit vahy bez stinu: m = (1-g)m + g*grad, preklopit bit kdyz m/rms(m) * w > tau (ostatni lion)
  afac  lion s faktorizovanou adaptivitou: krok = clip(m / sqrt(v_r v_c / mean v_r)) (radky+sloupce, ne vahy)
--chip: 1-bit vahy dopredu jako sign(w) * 2^-round(log2 sqrt(fan_in)) (posun misto
prumeru |w|, bez centrovani) - presne to, co umi cip; ternarni prah 0,5*mean|w|.
LR plan (--sched): posun +1 (polovicni krok) v 50 %, 75 % a 88 % treninku.

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
    if HW["pow2"]:
        return q_tok(x, bits)
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


CHIP = False   # nastavuje --chip
GQ = 0         # --gq: bitu gradientu ve zpetnem pruchodu (0 = float)
ABITS = 8      # --abits: bitu aktivaci na vstupu BitLinear (sirka scitacek v poli)
DWL = 8        # --dw-levels: pocet urovni exponentu u --dwq pow2
DWQ = "same"   # --dwq: kvantizace gradientu jen pro vypocet dW (same = jako dx, GQ bitu)


def q_sr(g, bits, per_token=False):
    """int na `bits` bitu, mocnina dvou absmax (po tenzoru / po tokenu), stochasticke zaokrouhleni."""
    qmax = 2 ** (bits - 1) - 1
    m = g.abs().amax(dim=-1, keepdim=True) if per_token else g.abs().amax()
    s = torch.exp2(torch.ceil(torch.log2(m.clamp_min(1e-30) / qmax)))
    return torch.clamp(torch.floor(g / s + torch.rand_like(g)), -qmax - 1, qmax) * s


def q_pow2(g, levels=8):
    """znamenko * 2^e (stochasticky mezi sousednimi mocninami, nestranne), e v `levels` urovnich pod max."""
    a = g.abs()
    top = torch.ceil(torch.log2(a.amax().clamp_min(1e-30)))
    lo = torch.exp2(top - levels + 1)
    e = torch.floor(torch.log2(a.clamp_min(1e-30)))
    b = torch.exp2(e)
    v = torch.where(torch.rand_like(a) < (a - b) / b, 2 * b, b)
    # pod nejnizsi urovni: stochasticky 0 nebo lo
    v = torch.where(a < lo, torch.where(torch.rand_like(a) < a / lo, lo, torch.zeros_like(a)), v)
    return torch.sign(g) * v


def q_dw(g):
    if DWQ == "int4":
        return q_sr(g, 4)
    if DWQ == "int4t":
        return q_sr(g, 4, per_token=True)
    if DWQ == "tern":
        return q_sr(g, 2)
    if DWQ == "ternt":
        return q_sr(g, 2, per_token=True)
    if DWQ == "pow2":
        return q_pow2(g, DWL)
    return q_sr(g, GQ)


class ChipLinearFn(torch.autograd.Function):
    """y = x W^T; zpet: dx z gradientu na GQ bitu, dW z gradientu kvantovaneho podle DWQ
    (v cipu: dW = sum_b 2^b x^T g_b po bitovych rovinach -> pocet pruchodu pole scitacek = bitu g)."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gx = q_sr(g, GQ)
        gw = gx if DWQ == "same" else q_dw(g)
        dw = gw.reshape(-1, gw.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        return gx @ w, dw


class GradQuant(torch.autograd.Function):
    """Dopredu identita; zpet gradient na GQ bitu, mocnina dvou absmax po tenzoru, stochasticke zaokrouhleni
    (tak by pocital zpetny pruchod v cipu: delta int8 s dynamickym posunem)."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        qmax = 2 ** (GQ - 1) - 1
        e = torch.ceil(torch.log2(g.abs().amax().clamp_min(1e-30) / qmax))
        s = torch.exp2(e)
        return torch.clamp(torch.floor(g / s + torch.rand_like(g)), -qmax - 1, qmax) * s


class BitLinear(nn.Linear):
    def __init__(self, i, o, mode):
        super().__init__(i, o, bias=False)
        self.mode = mode
        self.shift = 2.0 ** -round(math.log2(math.sqrt(i)))

    def forward(self, x):
        w = self.weight
        if CHIP:
            if self.mode == "ternary":
                t = 0.5 * w.abs().mean().detach()
                q = torch.sign(w) * (w.abs() > t)
            else:
                q = torch.sign(w)
            w = ste(w, q * self.shift)
        else:
            w = w_ternary(w) if self.mode == "ternary" else w_binary(w)
        xq = act_quant(x, ABITS)
        if CHIP and GQ and DWQ != "same":
            return ChipLinearFn.apply(xq, w)
        y = F.linear(xq, w)
        return GradQuant.apply(y) if GQ else y


class ChipOpt:
    """Celociselne uceni jako v cipu: int16 stin + posun o mocninu dvou + stochasticke zaokrouhleni."""

    def __init__(self, model, kind, lr_shift, norm="tensor", b1=0.9, b2=0.99, mbits=0):
        self.kind, self.lr_shift, self.norm, self.b1, self.b2 = kind, lr_shift, norm, b1, b2
        self.mbits = mbits
        self.bop_g, self.bop_tau = 1e-2, 3.0
        self.groups = []
        for mod in model.modules():
            for name, p in mod.named_parameters(recurse=False):
                # krok stinu: mocnina dvou, aby pocatecni max|p| ~ 2^12 LSB (8x rezerva do int16)
                step = 2.0 ** math.floor(math.log2(p.data.abs().max().clamp_min(1e-6).item() / 4096))
                q = torch.clamp(torch.round(p.data / step), -32767, 32767)
                p.data.copy_(q * step)
                st = {"p": p, "q": q, "step": step}
                st["bit"] = isinstance(mod, BitLinear)
                if kind in ("lion", "bop", "afac"):
                    st["m"] = torch.zeros_like(p.data)
                if kind == "afac" and p.dim() == 2:
                    st["vr"] = torch.zeros(p.shape[0], 1)
                    st["vc"] = torch.zeros(1, p.shape[1])
                if kind == "bop" and st["bit"]:
                    # bez stinu: vaha je jen znamenko (+-1 krok)
                    q.copy_(torch.where(q >= 0, 1.0, -1.0))
                    p.data.copy_(q * step)
                self.groups.append(st)

    def zero_grad(self, set_to_none=True):
        for g in self.groups:
            g["p"].grad = None

    @torch.no_grad()
    def step(self):
        k = 2.0 ** self.lr_shift
        for g in self.groups:
            p, q, step = g["p"], g["q"], g["step"]
            if p.grad is None:
                continue
            gr = p.grad
            if self.kind == "sgd":
                u = gr * (k / step)
            elif self.kind == "nsgd":
                if self.norm == "row" and gr.dim() == 2:
                    m = gr.abs().amax(dim=1, keepdim=True)
                else:
                    m = gr.abs().amax()
                e = torch.ceil(torch.log2(m.clamp_min(1e-30)))
                u = gr * torch.exp2(-e) * k
            elif self.kind == "bop" and g["bit"]:
                # Bop: moment gradientu, preklopeni bitu, kdyz normovany moment tlaci proti znamenku vahy
                mom = g["m"]
                mom.mul_(1 - self.bop_g).add_(gr, alpha=self.bop_g)
                mn = mom / mom.pow(2).mean().sqrt().clamp_min(1e-30)
                flip = (mn * q) > self.bop_tau
                q.copy_(torch.where(flip, -q, q))
                mom.masked_fill_(flip, 0.0)
                p.data.copy_(q * step)
                continue
            elif self.kind == "afac" and "vr" in g:
                # Lion + faktorizovana adaptivita (Adafactor): krok ~ moment / sqrt(v_radek * v_sloupec / mean)
                mom, vr, vc = g["m"], g["vr"], g["vc"]
                g2 = gr * gr
                vr.mul_(0.99).add_(g2.mean(dim=1, keepdim=True), alpha=0.01)
                vc.mul_(0.99).add_(g2.mean(dim=0, keepdim=True), alpha=0.01)
                mom.mul_(self.b2).add_(gr, alpha=1 - self.b2)
                den = torch.sqrt(vr * vc / vr.mean().clamp_min(1e-30)).clamp_min(1e-30)
                u = torch.clamp((self.b1 * mom + (1 - self.b1) * gr) / den, -1, 1) * k
            else:  # lion (u bop/afac pro embedding a LayerNorm)
                mom = g["m"]
                u = torch.sign(self.b1 * mom + (1 - self.b1) * gr) * k
                mom.mul_(self.b2).add_(gr, alpha=1 - self.b2)
                if self.mbits:
                    # moment ulozeny na mbits bitu: mocnina dvou absmax po tenzoru, stochasticke zaokrouhleni
                    qm = 2 ** (self.mbits - 1) - 1
                    sc = torch.exp2(torch.ceil(torch.log2(mom.abs().amax().clamp_min(1e-30) / qm)))
                    mom.copy_(torch.clamp(torch.floor(mom / sc + torch.rand_like(mom)), -qm - 1, qm) * sc)
            u = torch.floor(u + torch.rand_like(u))          # stochasticke zaokrouhleni
            q.sub_(u).clamp_(-32767, 32767)
            p.data.copy_(q * step)


HW = {"norm": "ln", "act": "gelu", "attn": "float", "pow2": False, "gain": False}   # --hw-*: zjednoduseni pro cip


def pow2_scale(m):
    """Mocnina dvou >= m (cip: posun misto nasobeni)."""
    return torch.exp2(torch.ceil(torch.log2(m.clamp_min(1e-30))))


def q_tok(x, bits=8):
    """Per-token kvantizace na int s mocninou dvou (STE), vraci dekvantovane hodnoty."""
    qmax = 2 ** (bits - 1) - 1
    s = pow2_scale(x.abs().amax(dim=-1, keepdim=True) / qmax)
    return ste(x, torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s)


class Norm(nn.Module):
    """ln = LayerNorm (float, afinni); rms = RMSNorm bez parametru; amax = deleni per-token absmax
    zaokrouhlenym na mocninu dvou (v cipu zdarma: je to posun, ktery uz dela kvantizace aktivaci)."""

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d) if HW["norm"] == "ln" else None
        # --hw-gain: uceny zisk po kanalech (v cipu jedno int nasobeni na prvek)
        self.g = nn.Parameter(torch.ones(d)) if HW["gain"] and self.ln is None else None

    def forward(self, x):
        if self.ln is not None:
            return self.ln(x)
        if HW["norm"] == "rms":
            y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        else:
            # absmax -> max 1; x4 (posun) vrati RMS zhruba na 1 jako u LayerNorm
            m = x.abs().amax(dim=-1, keepdim=True).detach()
            y = 4.0 * x / (pow2_scale(m) if HW["pow2"] else m.clamp_min(1e-6))
        return y * self.g if self.g is not None else y


def act_fn(x):
    if HW["act"] == "relu":
        return F.relu(x)
    if HW["act"] == "relu2":
        return F.relu(x).pow(2)
    return F.gelu(x)


def int_attention(q, k, v):
    """Attention jako v cipu: q, k, v int8 (per-token mocnina dvou), skore int (q.k) s posunem,
    softmax pres 2^x s 8bitovou tabulkou, pravdepodobnosti uint8, p.v int8 x uint8."""
    q, k, v = q_tok(q), q_tok(k), q_tok(v)
    t, dh = q.shape[-2], q.shape[-1]
    sc = (q @ k.transpose(-1, -2)) * (2.0 ** -round(math.log2(math.sqrt(dh))))
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    sc = sc.masked_fill(~mask, float("-inf"))
    e = torch.exp(sc - sc.amax(dim=-1, keepdim=True))
    e = ste(e, torch.round(e * 255) / 255)                      # tabulka exp: 8 bitu
    p = e / e.sum(dim=-1, keepdim=True)
    p = ste(p, torch.round(p * 255) / 255)                      # pravdepodobnosti uint8
    return p @ v


class BitBlock(nn.Module):
    def __init__(self, d, heads, ff, mode):
        super().__init__()
        self.ln1, self.ln2 = Norm(d), Norm(d)
        self.qkv = BitLinear(d, 3 * d, mode)
        self.proj = BitLinear(d, d, mode)
        self.fc1 = BitLinear(d, ff, mode)
        self.fc2 = BitLinear(ff, d, mode)
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        if HW["attn"] == "int":
            y = int_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(act_fn(self.fc1(self.ln2(x))))


class BitGPT(nn.Module):
    def __init__(self, ctx, d, layers, heads, ff, mode):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(BitBlock(d, heads, ff, mode) for _ in range(layers))
        self.ln_f = Norm(d)
        self.ctx = ctx
        if CHIP:
            nn.init.normal_(self.tok.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)

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
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd", "nsgd", "lion", "bop", "afac"])
    ap.add_argument("--lr-shift", type=int, default=0, help="krok updatu 2^lr_shift (v LSB stinu, u sgd v jednotkach g/krok)")
    ap.add_argument("--norm", default="tensor", choices=["tensor", "row"])
    ap.add_argument("--chip", action="store_true", help="1-bit vahy s pevnym posunem misto prumeru |w|")
    ap.add_argument("--b1", type=float, default=0.9, help="lion: vaha momentu pro znamenko (0 = signSGD)")
    ap.add_argument("--b2", type=float, default=0.99, help="lion: zapominani momentu")
    ap.add_argument("--mbits", type=int, default=0, help="lion: bitu ulozeneho momentu (0 = float)")
    ap.add_argument("--bop-g", type=float, default=1e-2, help="bop: rychlost momentu")
    ap.add_argument("--bop-tau", type=float, default=3.0, help="bop: prah preklopeni (v RMS momentu)")
    ap.add_argument("--gq", type=int, default=0, help="bitu gradientu ve zpetnem pruchodu BitLinear (0 = float)")
    ap.add_argument("--dw-levels", type=int, default=8, help="--dwq pow2: urovni exponentu (posuvnik)")
    ap.add_argument("--abits", type=int, default=8, help="bitu aktivaci na vstupu BitLinear")
    ap.add_argument("--dwq", default="same", choices=["same", "int4", "int4t", "tern", "ternt", "pow2"],
                    help="kvantizace gradientu pro dW (pocet pruchodu pole scitacek)")
    ap.add_argument("--sched", action="store_true", help="u sgd/nsgd/lion pulit krok v 50/75/88 %%")
    ap.add_argument("--hw-norm", default="ln", choices=["ln", "rms", "amax"])
    ap.add_argument("--hw-act", default="gelu", choices=["gelu", "relu", "relu2"])
    ap.add_argument("--hw-attn", default="float", choices=["float", "int"])
    ap.add_argument("--hw-pow2", action="store_true", help="skaly aktivaci jako mocniny dvou")
    ap.add_argument("--hw-gain", action="store_true", help="rms/amax norma s ucenym ziskem po kanalech")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    HW.update(norm=a.hw_norm, act=a.hw_act, attn=a.hw_attn, pow2=a.hw_pow2, gain=a.hw_gain)

    global CHIP, GQ, DWQ, ABITS, DWL
    CHIP, GQ, DWQ, ABITS, DWL = a.chip, a.gq, a.dwq, a.abits, a.dw_levels
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

    if a.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    else:
        opt = ChipOpt(model, a.opt, a.lr_shift, a.norm, a.b1, a.b2, a.mbits)
        opt.bop_g, opt.bop_tau = a.bop_g, a.bop_tau
        sched = None
    print(f"optimalizator {a.opt} lr_shift {a.lr_shift} norm {a.norm} chip {a.chip} sched {a.sched} gq {a.gq} b1 {a.b1} b2 {a.b2} mbits {a.mbits}", flush=True)
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
        if sched is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        else:
            if a.sched and step in (a.steps // 2, a.steps * 3 // 4, a.steps * 7 // 8):
                opt.lr_shift -= 1
            opt.step()
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
    res = {"model": a.model, "opt": a.opt, "lr_shift": a.lr_shift, "norm": a.norm, "chip": a.chip, "hw": dict(HW), "gq": a.gq, "dwq": a.dwq, "abits": a.abits, "dw_levels": a.dw_levels, "b1": a.b1, "b2": a.b2, "mbits": a.mbits,
           "sched": a.sched, "params": n_params, "bitlm_params": BITLM_PARAMS, "ctx": ctx,
           "steps": a.steps, "batch": a.batch, "best_step": best[1], "val_ce_bits": ce, "val_acc": acc,
           "train_seconds_cpu": round(time.time() - t0, 1), "sample": sample, "log": log}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"VYSLEDEK {a.model}: CE {ce:.4f} acc {acc:.4f} (nejlepsi krok {best[1]})")
    print("ukazka:", sample[:200])


if __name__ == "__main__":
    main()
