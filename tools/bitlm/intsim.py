#!/usr/bin/env python3
"""BitTF intsim - celociselna referencni simulace treninku 1-bit transformeru presne podle cipu
(specifikace docs/bitlm_poc.md kap. 11, aritmetika docs/bittf_golden.md).

Vse jsou cela cisla (torch.int64); float64 slouzi jen jako rychla presna nasobicka celych cisel
(vsechny soucty < 2^53, kontroluje se) a pro mereni CE pri evaluaci. Nahodna cisla pro stochasticke
zaokrouhleni jsou z hashe (fmix32) citace volani a indexu prvku -> stejny vysledek v C (sw/bittf/golden)
a pozdeji v RTL.

Tenzory nesou mantisu (int) a exponent (mocnina dvou, po tenzoru nebo po tokenu): hodnota = m * 2^e.

Pouziti: intsim.py data/cz_mix.tok --steps 8000 --out results/.../intsim.json [--ckpt c.pt --time-limit s]
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_baseline import VOCAB  # noqa: E402

I64 = torch.int64
M32 = 0xFFFFFFFF
V = 32

# ---------------------------------------------------------------- tabulky (ROM v cipu)
# 1/sqrt: index = horni bity normalizovane mantisy sumy ctvercu (64..255)
RSQ = torch.tensor([0] * 64 + [int(math.floor(2 ** 22 / math.sqrt((i + 0.5) * 256) + 0.5)) for i in range(64, 256)],
                   dtype=I64)
# exp(-i/32): 8 bitu pro attention, 16 bitu pro softmax vystupu
EXP8 = torch.tensor([int(math.floor(255 * math.exp(-i / 32) + 0.5)) for i in range(256)], dtype=I64)
EXP16 = torch.tensor([int(math.floor(65535 * math.exp(-i / 32) + 0.5)) for i in range(512)], dtype=I64)


# ---------------------------------------------------------------- RNG (hash, jako RTL BitLM)
def mul32_safe(a, c):
    """(a * c) mod 2^32 pro a v [0, 2^32) bez preteceni int64."""
    return (a * (c & 0xFFFF) + (((a * (c >> 16)) & 0xFFFF) << 16)) & M32


def mul32_wrap(a, c):
    """Totez pres nasobeni int64 s pretecenim (na CPU se dvojkovym doplnkem; overeno pri importu)."""
    return (a * c) & M32


_t = torch.randint(0, 2 ** 32, (4096,), dtype=I64)
mul32 = mul32_wrap if all(torch.equal(mul32_wrap(_t, c), mul32_safe(_t, c))
                          for c in (0x85EBCA6B, 0xC2B2AE35, 0x9E3779B1)) else mul32_safe
_IDX = {}


def fmix32(h):
    h = h ^ (h >> 16)
    h = mul32(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = mul32(h, 0xC2B2AE35)
    return h ^ (h >> 16)


def fmix32_int(h):
    return int(fmix32(torch.tensor([h & M32], dtype=I64))[0])


class RNG:
    """Kazde volani draw() = novy proud (citac ctr); prvek i dostane fmix32(i*φ + key)."""

    def __init__(self, seed):
        self.seed, self.ctr = seed & M32, 0

    def draw(self, shape):
        key = fmix32_int(self.seed ^ fmix32_int(self.ctr))
        self.ctr += 1
        n = 1
        for s in shape:
            n *= s
        if n not in _IDX:
            _IDX[n] = mul32(torch.arange(n, dtype=I64), 0x9E3779B1)
        return fmix32((_IDX[n] + key) & M32).reshape(shape)


# ---------------------------------------------------------------- celociselna primitiva
def bitlen(x):
    """Pocet bitu nezaporneho int64 (0 -> 0), po prvcich."""
    if x.numel() and int(x.max()) < 2 ** 53:      # presny prevod do float64 -> exponent z frexp
        return torch.frexp(x.double()).exponent.to(I64)
    n = torch.zeros_like(x)
    for k in (32, 16, 8, 4, 2, 1):
        n = n + k * ((x >> (n + k)) != 0)
    return n + (x != 0)


def _sh(s, like):
    return s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=I64)


def rsr(a, s):
    """a / 2^s zaokrouhlene (pul nahoru); s <= 0 = presny posun vlevo. s int nebo tenzor."""
    s = _sh(s, a)
    sp, sn = s.clamp(min=0, max=62), (-s).clamp(min=0)
    return ((a + ((torch.ones_like(sp) << sp) >> 1)) >> sp) << sn


def sr(a, s, r):
    """floor(a / 2^s + U), U = r / 2^s (stochasticke zaokrouhleni, nestranne); s <= 0 presne vlevo."""
    s = _sh(s, a)
    sp, sn = s.clamp(min=0, max=32), (-s).clamp(min=0)
    return ((a + (r & ((torch.ones_like(sp) << sp) - 1))) >> sp) << sn


def amax_int(m):
    return int(m.abs().max()) if m.numel() else 0


def normalize(m, e, bits):
    """BFP po tenzoru: posun mantisy tak, aby max|m| melo `bits` bitu (zaokrouhleni)."""
    a = amax_int(m)
    if a == 0:
        return m, e
    s = a.bit_length() - bits
    return rsr(m, s), e + s


def quant_tok(y, bits):
    """Kvantizace po tokenu (posledni osa) na int `bits` se skalou 2^s; vraci (mantisa, s)."""
    q = 2 ** (bits - 1) - 1
    s = bitlen(y.abs().amax(dim=-1, keepdim=True)) - (bits - 1)
    return rsr(y, s).clamp(-q - 1, q), s


def quant_sr(m, e, bits, rng):
    """Kvantizace po tenzoru na int `bits` se stochastickym zaokrouhlenim; vraci (mantisa, exponent)."""
    q = 2 ** (bits - 1) - 1
    a = amax_int(m)
    s = a.bit_length() - (bits - 1)
    r = rng.draw(m.shape)
    return sr(m, s, r).clamp(-q - 1, q), e + s


def pow2q(m, lo_e, rng):
    """Gradient pro dW: znamenko * 2^k (k = 0..7 nad 2^lo_e), stochasticky nestranne; vraci sgn*2^k."""
    a, sgn = m.abs(), torch.sign(m)
    lo = 1 << lo_e
    r = rng.draw(m.shape)
    e = (bitlen(a) - 1).clamp(min=0)
    b = torch.ones_like(e) << e
    vbig = torch.where((r & (b - 1)) < a - b, b << 1, b)
    vsmall = torch.where((r & (lo - 1)) < a, torch.full_like(a, lo), torch.zeros_like(a))
    v = torch.where(a >= lo, vbig, vsmall)
    return sgn * (v >> lo_e)


def align_head(m, e, dims, head=8):
    """Zarovnani mantis s exponenty po tokenech na spolecny zaklad (max - head) pres osy dims."""
    base = e.amax(dim=dims, keepdim=True) - head
    return rsr(m, base - e), base


def align_max(m, e, dims):
    base = e.amax(dim=dims, keepdim=True)
    return rsr(m, base - e), base


def imm(a, b):
    """Presne celociselne nasobeni matic (float64, kontrola meze 2^53)."""
    k = a.shape[-1]
    assert amax_int(a) * amax_int(b) * k < 2 ** 53, "imm: mez presnosti"
    return torch.matmul(a.double(), b.double()).long()


def rdiv(a, d):
    """round(a / d), pul nahoru (d > 0 konstanta)."""
    return torch.div(2 * a + d, 2 * d, rounding_mode="floor")


def add_bfp(x, y):
    """Soucet dvou BFP (m, e) po tenzoru; vysledek s 40 bity rezervy pod vetsim."""
    if x is None:
        return y
    (m1, e1), (m2, e2) = x, y
    t1 = e1 + amax_int(m1).bit_length() if amax_int(m1) else -10 ** 9
    t2 = e2 + amax_int(m2).bit_length() if amax_int(m2) else -10 ** 9
    if max(t1, t2) < -10 ** 8:
        return m1, e1
    base = max(t1, t2) - 40
    return rsr(m1, base - e1) + rsr(m2, base - e2), base


def kshift(fan_in):
    """Posun 1-bit vahy: 2^-round(log2 sqrt(fan_in))."""
    return int(math.floor(math.log2(math.sqrt(fan_in)) + 0.5))


# ---------------------------------------------------------------- model
class Model:
    """Parametry = int16 stin (q). Kroky stinu: vahy BitLinear jen znamenko, embedding 2^-16, zisk 2^-12."""

    def __init__(self, d, layers, heads, ff, ctx, rng):
        self.d, self.L, self.H, self.ff, self.T = d, layers, heads, ff, ctx
        self.dh = d // heads
        self.ks = (self.dh.bit_length() - 1) // 2          # skore attention * 2^-ks
        self.sqd = int(math.floor(math.sqrt(d) * 4096 + 0.5))
        self.names, self.p = [], {}

        def gauss(shape):   # ~N(0, 0,02) v LSB 2^-16 (Irwin-Hall ze 4 uniformnich)
            r = [rng.draw(shape) & 4095 for _ in range(4)]
            return ((r[0] + r[1] + r[2] + r[3] - 8190) * 2269) >> 12

        def unif(shape):
            return (rng.draw(shape) & 8191) - 4096

        def add(name, t):
            self.names.append(name)
            self.p[name] = t.clamp(-32767, 32767)

        add("tok", gauss((V, d)))
        add("pos", gauss((ctx, d)))
        for l in range(layers):
            add(f"{l}.ln1", torch.full((d,), 4096, dtype=I64))
            add(f"{l}.qkv", unif((3 * d, d)))
            add(f"{l}.proj", unif((d, d)))
            add(f"{l}.ln2", torch.full((d,), 4096, dtype=I64))
            add(f"{l}.fc1", unif((ff, d)))
            add(f"{l}.fc2", unif((d, ff)))
        add("lnf", torch.full((d,), 4096, dtype=I64))
        self.k = {"qkv": kshift(d), "proj": kshift(d), "fc1": kshift(d), "fc2": kshift(ff)}

    @staticmethod
    def sign(q):
        return torch.where(q >= 0, 1, -1).to(I64)

    # ---------------- RMS norma (vstup rezidualni proud Q16)
    def norm_fwd(self, x):
        ss = (x * x).sum(dim=-1, keepdim=True)
        assert int(ss.max()) < 2 ** 62
        sh = bitlen(ss) - 16
        sh = sh + (sh & 1)
        m = torch.where(sh >= 0, ss >> sh.clamp(min=0), ss << (-sh).clamp(min=0))
        rr = rsr(self.sqd * RSQ[(m >> 8).clamp(0, 255)], 16)
        xh = rsr(x * rr, 6 + (sh >> 1))                       # x^ v Q12
        return xh, rr, sh

    def norm_bwd(self, dy, xh, rr, sh, g):
        """dy = (m, e) gradient vystupu normy (y = x^ * g, Q24); vraci (dg, dx) jako BFP."""
        m, e = normalize(dy[0], dy[1], 16)
        n = m.shape[-1]
        dg = ((m * xh).reshape(-1, n).sum(0), e - 12)
        dxh = m * g
        dot = (dxh * xh).sum(dim=-1, keepdim=True)
        mean = rdiv(dot, n)
        t = dxh - rsr(xh * rsr(mean, 12), 12)
        dx = t * rr
        etok = e - 14 - (sh >> 1)
        dxm, base = align_max(dx, etok, tuple(range(dx.dim())))
        return dg, (dxm, int(base.reshape(-1)[0]))

    # ---------------- 1-bit linearni vrstva
    def lin_bwd(self, G, a, ea, q, k, rng):
        """G = (m, e) gradient vystupu; a, ea = vstup int6 a jeho exponenty po tokenech.
        Vraci (dx jako BFP, dW jako BFP). dx z gradientu int8 (SR), dW z gradientu 2^k (SR)."""
        m, e = normalize(G[0], G[1], 16)
        n_out, n_in = q.shape
        mm = m.reshape(-1, n_out)
        if amax_int(mm) == 0:
            return (torch.zeros(*a.shape[:-1], n_in, dtype=I64), 0), (torch.zeros_like(q), 0)
        gx, egx = quant_sr(mm, e, 8, rng)
        dx = imm(gx, self.sign(q)).reshape(*a.shape[:-1], n_in)
        lo_e = amax_int(mm).bit_length() - 7
        gw = pow2q(mm, lo_e, rng)
        aa, base = align_head(a, ea, tuple(range(a.dim())))
        dw = imm(gw.t(), aa.reshape(-1, n_in))
        return (dx, egx - k), (dw, e + lo_e + int(base.reshape(-1)[0]))

    # ---------------- dopredny pruchod
    def forward(self, idx, train=True):
        p, B, T = self.p, idx.shape[0], idx.shape[1]
        d, H, dh = self.d, self.H, self.dh
        x = p["tok"][idx] + p["pos"][:T]                     # rezidualni proud, Q16 (int32)
        cache = {"idx": idx, "layers": []}
        for l in range(self.L):
            c = {}
            # --- attention
            xh, rr, sh = self.norm_fwd(x)
            a1, s1 = quant_tok(xh * p[f"{l}.ln1"], 6)
            ea1 = s1 - 24
            o = imm(a1, self.sign(p[f"{l}.qkv"]).t())
            eo = (ea1 - self.k["qkv"]).unsqueeze(1)          # B,1,T,1
            q, k, v = (z.reshape(B, T, H, dh).transpose(1, 2) for z in o.split(d, dim=-1))
            q8, sq = quant_tok(q, 8)
            eq = eo + sq                                      # B,H,T,1
            emax = eo.amax(dim=2, keepdim=True)               # B,1,1,1
            kk, vv = rsr(k, emax - eo), rsr(v, emax - eo)
            sk = bitlen(kk.abs().amax(dim=(2, 3), keepdim=True)) - 7
            sv = bitlen(vv.abs().amax(dim=(2, 3), keepdim=True)) - 7
            k8, v8 = rsr(kk, sk).clamp(-128, 127), rsr(vv, sv).clamp(-128, 127)
            ek, ev = emax + sk, emax + sv                     # B,H,1,1
            sc = imm(q8, k8.transpose(-1, -2))                # B,H,T,T
            sfx = rsr(sc, -(eq + ek - self.ks) - 5)           # skore v Q5 (prirozeny log)
            mask = torch.ones(T, T, dtype=torch.bool).tril()
            sfx = torch.where(mask, sfx, torch.full_like(sfx, -(1 << 40)))
            dd = (sfx.amax(dim=-1, keepdim=True) - sfx).clamp(max=255)
            e8 = torch.where(mask, EXP8[dd], torch.zeros_like(dd))
            ssum = e8.sum(dim=-1, keepdim=True)
            rcp = torch.div((1 << 24) + (ssum >> 1), ssum, rounding_mode="floor")
            p8 = ((e8 * rcp + (1 << 15)) >> 16).clamp(max=255)
            yh = imm(p8, v8)                                  # B,H,T,dh, exponent ev - 8
            yal, eatt = align_max(yh, ev - 8, (1,))           # hlavy na spolecny exponent
            yatt = yal.transpose(1, 2).reshape(B, T, d)
            eatt = eatt.reshape(B, 1, 1)
            a2, s2 = quant_tok(yatt, 6)
            ea2 = eatt + s2
            po = imm(a2, self.sign(p[f"{l}.proj"]).t())
            x = x + rsr(po, -(ea2 - self.k["proj"]) - 16)
            if train:
                c.update(xh1=xh, rr1=rr, sh1=sh, a1=a1, ea1=ea1, q8=q8, eq=eq, k8=k8, ek=ek, v8=v8, ev=ev,
                         p8=p8, a2=a2, ea2=ea2)
            # --- MLP s ReLU^2
            xh, rr, sh = self.norm_fwd(x)
            a3, s3 = quant_tok(xh * p[f"{l}.ln2"], 6)
            ea3 = s3 - 24
            fo = imm(a3, self.sign(p[f"{l}.fc1"]).t())
            r = fo.clamp(min=0)
            a4, s4 = quant_tok(r * r, 6)
            ea4 = 2 * (ea3 - self.k["fc1"]) + s4
            f2 = imm(a4, self.sign(p[f"{l}.fc2"]).t())
            x = x + rsr(f2, -(ea4 - self.k["fc2"]) - 16)
            assert int(x.abs().max()) < 2 ** 31, "rezidualni proud pretekl int32"
            if train:
                c.update(xh2=xh, rr2=rr, sh2=sh, a3=a3, ea3=ea3, fo=fo, a4=a4, ea4=ea4)
                cache["layers"].append(c)
        xh, rr, sh = self.norm_fwd(x)
        n8, sn = quant_tok(xh * p["lnf"], 8)
        en = sn - 24
        logits = imm(n8, p["tok"].t())                        # exponent en - 16 po tokenech
        cache.update(xhf=xh, rrf=rr, shf=sh, n8=n8, en=en)
        return logits, en - 16, cache

    # ---------------- zpetny pruchod
    def backward(self, logits, el, y, cache, rng):
        p, idx = self.p, cache["idx"]
        B, T = idx.shape
        N = B * T
        d, H, dh = self.d, self.H, self.dh
        grads = {}
        # softmax + CE: gradient (p - onehot) v Q15, prumer pres N tokenu -> exponent -15 - log2 N
        lfx = rsr(logits, -el - 8)                            # Q8
        dd = (lfx.amax(dim=-1, keepdim=True) - lfx + 4) >> 3  # Q5
        e16 = torch.where(dd < 512, EXP16[dd.clamp(max=511)], torch.zeros_like(dd))
        ssum = e16.sum(dim=-1, keepdim=True)
        rcp = torch.div((1 << 31) + (ssum >> 1), ssum, rounding_mode="floor")
        pr = (e16 * rcp + (1 << 15)) >> 16
        D = pr - F.one_hot(y, V).to(I64) * (1 << 15)
        eD = -15 - int(round(math.log2(N)))
        # hlava (svazana s embeddingem)
        n8, en = cache["n8"], cache["en"]
        nn_, base = align_head(n8, en, (0, 1, 2))
        grads["tok"] = (imm(D.reshape(N, V).t(), nn_.reshape(N, d)), eD + int(base.reshape(-1)[0]))
        dn = (imm(D, p["tok"]), eD - 16)
        dg, dX = self.norm_bwd(dn, cache["xhf"], cache["rrf"], cache["shf"], p["lnf"])
        grads["lnf"] = dg
        for l in reversed(range(self.L)):
            c = cache["layers"][l]
            # --- MLP
            dH, grads[f"{l}.fc2"] = self.lin_bwd(dX, c["a4"], c["ea4"], p[f"{l}.fc2"], self.k["fc2"], rng)
            fo = c["fo"]
            dfo = dH[0] * fo.clamp(min=0)                     # d(r^2) = 2 r dr
            dfo, efo = align_max(dfo, dH[1] + 1 + c["ea3"] - self.k["fc1"], (0, 1, 2))
            dA3, grads[f"{l}.fc1"] = self.lin_bwd((dfo, int(efo.reshape(-1)[0])), c["a3"], c["ea3"],
                                                  p[f"{l}.fc1"], self.k["fc1"], rng)
            grads[f"{l}.ln2"], dXn = self.norm_bwd(dA3, c["xh2"], c["rr2"], c["sh2"], p[f"{l}.ln2"])
            dX = add_bfp(dX, dXn)
            # --- attention
            dY, grads[f"{l}.proj"] = self.lin_bwd(dX, c["a2"], c["ea2"], p[f"{l}.proj"], self.k["proj"], rng)
            G = self.attn_bwd(dY, c, rng)
            dA1, grads[f"{l}.qkv"] = self.lin_bwd(G, c["a1"], c["ea1"], p[f"{l}.qkv"], self.k["qkv"], rng)
            grads[f"{l}.ln1"], dXn = self.norm_bwd(dA1, c["xh1"], c["rr1"], c["sh1"], p[f"{l}.ln1"])
            dX = add_bfp(dX, dXn)
        m, e = dX
        dtok = torch.zeros(V, d, dtype=I64).index_add_(0, idx.reshape(-1), m.reshape(N, d))
        grads["tok"] = add_bfp(grads["tok"], (dtok, e))
        grads["pos"] = (torch.zeros(self.T, d, dtype=I64).index_add_(0, torch.arange(T).repeat(B),
                                                                      m.reshape(N, d)), e)
        return grads

    def attn_bwd(self, dY, c, rng):
        B, T, _ = c["a1"].shape
        H, dh = self.H, self.dh
        m, e = normalize(dY[0], dY[1], 16)
        dy8, ey = quant_sr(m.reshape(B, T, H, dh).transpose(1, 2), e, 8, rng)
        p8, q8, k8, v8 = c["p8"], c["q8"], c["k8"], c["v8"]
        dv = (imm(p8.transpose(-1, -2), dy8), ey - 8)
        dp = imm(dy8, v8.transpose(-1, -2))                   # exponent ey + ev
        rowdot = (p8 * dp).sum(dim=-1, keepdim=True)
        ds = p8 * (dp - rsr(rowdot, 8))                       # exponent ey + ev - 8
        dsa, eds = align_max(ds, c["ev"], (0, 1))
        dsn, edsn = normalize(dsa, ey - 8 + int(eds.reshape(-1)[0]), 16)
        ds8, eds8 = quant_sr(dsn, edsn, 8, rng)
        dq, eq_ = align_max(imm(ds8, k8), c["ek"], (0, 1))
        dq = (dq, eds8 - self.ks + int(eq_.reshape(-1)[0]))
        qa, qb = align_head(q8, c["eq"], (2,))                # B,H,1,1
        dk, ek_ = align_max(imm(ds8.transpose(-1, -2), qa), qb, (0, 1))
        dk = (dk, eds8 - self.ks + int(ek_.reshape(-1)[0]))
        # spolecny exponent pro q, k, v a slozeni do gradientu vystupu qkv (B, T, 3d)
        parts = [dq, dk, dv]
        tops = [pe + amax_int(pm).bit_length() for pm, pe in parts if amax_int(pm)]
        if not tops:
            return torch.zeros(B, T, 3 * self.d, dtype=I64), 0
        base = max(tops) - 40
        ms = [rsr(pm, base - pe).transpose(1, 2).reshape(B, T, self.d) for pm, pe in parts]
        return torch.cat(ms, dim=-1), base


# ---------------------------------------------------------------- Lion (int8 moment)
class Lion:
    """u = sign(7/8 m + 1/8 g); q -= u * 2^lr; m = 127/128 m + 1/128 g, ulozeno int8 (SR, skala 2^e)."""

    def __init__(self, model, lr_shift):
        self.lr_shift = lr_shift
        self.m = {n: (torch.zeros_like(t), 0) for n, t in model.p.items()}

    def step(self, model, grads, rng):
        for n in model.names:
            g, eg = grads[n]
            mq, em = self.m[n]
            ag, am = amax_int(g), amax_int(mq)
            if ag == 0 and am == 0:
                continue
            tg = eg + ag.bit_length() if ag else -10 ** 9
            tm = em + am.bit_length() if am else -10 ** 9
            base = max(tg, tm) - 30
            gc = rsr(g, base - eg)
            mc = rsr(mq, base - em) if am else torch.zeros_like(gc)
            u = torch.sign(7 * mc + gc)
            q = model.p[n]
            model.p[n] = (q - (u << self.lr_shift)).clamp(-32767, 32767)
            self.m[n] = quant_sr(127 * mc + gc, base - 7, 8, rng)


# ---------------------------------------------------------------- trenink a evaluace
def evaluate(model, data, split, stride=1, bs=256):
    """CE (bity/znak) a presnost; kontext = predchozich T znaku. CE z celociselnych logitu ve float64."""
    T = model.T
    pos = list(range(split, len(data), stride))
    ce, ok = 0.0, 0
    for i in range(0, len(pos), bs):
        pp = pos[i:i + bs]
        idx = torch.stack([data[q - T:q] for q in pp])
        logits, el, _ = model.forward(idx, train=False)
        lg = logits[:, -1, :].double() * torch.exp2(el[:, -1, :].double())
        tgt = data[pp]
        ce += F.cross_entropy(lg, tgt, reduction="sum").item()
        ok += (lg.argmax(1) == tgt).sum().item()
    return ce / len(pos) / math.log(2), ok / len(pos)


def generate(model, prompt, n, seed=7):
    """Vzorkovani z celociselnych logitu (pro ukazku; RNG pro vzorkovani float)."""
    g = torch.Generator().manual_seed(seed)
    seq = list(prompt)
    for _ in range(n):
        idx = torch.tensor([seq[-model.T:]], dtype=I64)
        logits, el, _ = model.forward(idx, train=False)
        lg = logits[0, -1].double() * float(2.0 ** int(el[0, -1, 0]))
        seq.append(int(torch.multinomial(torch.softmax(lg, 0), 1, generator=g)))
    return "".join(VOCAB[t] for t in seq[len(prompt):])


def state_hash(model, opt, rng):
    h = hashlib.sha256()
    for n in model.names:
        h.update(bytes(str(model.p[n].tolist()), "ascii"))
        h.update(bytes(str((opt.m[n][0].tolist(), opt.m[n][1])), "ascii"))
    h.update(bytes(str(rng.ctr), "ascii"))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=648)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr-shift", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--val-len", type=int, default=11152)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--hash-every", type=int, default=0, help="vypis hashe stavu (kontrola proti C)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--time-limit", type=float, default=1e12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    raw = open(a.corpus, "rb").read()
    data = (torch.frombuffer(bytearray(raw), dtype=torch.uint8) & 31).long()
    split = len(data) - a.val_len
    rng = RNG(a.seed)
    model = Model(a.d, a.layers, a.heads, a.ff, a.ctx, rng)
    opt = Lion(model, a.lr_shift)
    n_params = sum(t.numel() for t in model.p.values())
    print(f"intsim d={a.d} L={a.layers} H={a.heads} ff={a.ff}: {n_params} parametru, korpus {len(data)}", flush=True)

    start, log, best, train_s = 1, [], None, 0.0
    if a.ckpt and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, weights_only=False)
        model.p, opt.m, opt.lr_shift, rng.ctr = ck["p"], ck["m"], ck["lr_shift"], ck["ctr"]
        start, log, best, train_s = ck["step"] + 1, ck["log"], ck["best"], ck["train_s"]
        print(f"pokracuji od kroku {start}", flush=True)

    def save(step):
        if a.ckpt:
            torch.save({"p": model.p, "m": opt.m, "lr_shift": opt.lr_shift, "ctr": rng.ctr, "step": step,
                        "log": log, "best": best, "train_s": train_s}, a.ckpt + ".tmp")
            os.replace(a.ckpt + ".tmp", a.ckpt)

    t0 = t_seg = time.time()
    step = start - 1
    for step in range(start, a.steps + 1):
        if step in (a.steps // 2, a.steps * 3 // 4, a.steps * 7 // 8):
            opt.lr_shift -= 1
        ix = (a.ctx + rng.draw((a.batch,)) % (split - 1 - a.ctx)).tolist()
        x = torch.stack([data[i - a.ctx:i] for i in ix])
        y = torch.stack([data[i - a.ctx + 1:i + 1] for i in ix])
        logits, el, cache = model.forward(x)
        grads = model.backward(logits, el, y, cache, rng)
        opt.step(model, grads, rng)
        if a.hash_every and step % a.hash_every == 0:
            print(f"hash {step} {state_hash(model, opt, rng)}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            now = time.time()
            train_s += now - t_seg
            t_seg = now
            ce, acc = evaluate(model, data, split, stride=4)
            log.append({"step": step, "val_ce_bits": ce, "val_acc": acc, "hours": train_s / 3600})
            if best is None or ce < best[0]:
                best = (ce, step, copy.deepcopy(model.p))
            print(f"{step:6d}  val CE(1/4) {ce:.4f}  acc {acc:.4f}  lr_shift {opt.lr_shift}  ({train_s / 3600:.2f} h)",
                  flush=True)
            save(step)
            if time.time() - t0 > a.time_limit and step < a.steps:
                print(f"casovy limit, checkpoint v kroku {step}", flush=True)
                return
    if step < a.steps:
        return
    model.p = best[2]
    ce, acc = evaluate(model, data, split)
    sample = generate(model, data[split:split + a.ctx].tolist(), 400)
    res = {"model": "bittf-intsim", "params": n_params, "d": a.d, "layers": a.layers, "heads": a.heads, "ff": a.ff,
           "steps": a.steps, "batch": a.batch, "lr_shift": a.lr_shift, "seed": a.seed, "best_step": best[1],
           "val_ce_bits": ce, "val_acc": acc, "train_hours": train_s / 3600, "sample": sample, "log": log}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"VYSLEDEK intsim: CE {ce:.4f} acc {acc:.4f} (nejlepsi krok {best[1]})")
    print("ukazka:", sample[:200])


if __name__ == "__main__":
    main()
