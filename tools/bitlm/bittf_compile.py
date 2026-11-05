#!/usr/bin/env python3
"""Prekladac BitTF: model -> program jednoho kroku treninku pro ISA v1 (docs/bittf_isa.md).

Program presne odpovida sw/bittf/golden/bittf_ref.c (bt_forward + bt_backward_update) vcetne
vyberu davky z korpusu (bittf_train.c). Vystup:
  <out>.bin     instrukce (nahrat na adresu 'prog' z layoutu)
  <out>.layout  rozlozeni pameti: konfigurace, tenzory (nazev adresa sirka bez_znamenka pocet)

bittf_compile.py --d 128 --layers 5 --heads 4 --ff 648 --ctx 128 --batch 32 --corpus-len N --val-len 11152 -o out/m1
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bittf_isa as I  # noqa: E402
from bittf_isa import Opnd, Gather  # noqa: E402

V = 32
NONE = -1000000000


class T:
    def __init__(self, name, addr, n, width, unsigned=False):
        self.name, self.addr, self.n, self.width, self.unsigned = name, addr, n, width, unsigned


class Mem:
    def __init__(self):
        self.top = 4096
        self.tensors = []

    def alloc(self, name, n, width, unsigned=False):
        t = T(name, self.top, n, width, unsigned)
        self.top += (n * width + 63) // 64 * 64
        self.tensors.append(t)
        return t


class Prog:
    def __init__(self):
        self.code = []
        self.labels = {}
        self.fix = []        # (index, label)
        self.free = list(range(63, 0, -1))
        self.lab_n = 0

    # ---------------------------------------------------- registry
    def reg(self):
        return self.free.pop()

    def rel(self, *rs):
        for r in rs:
            assert r not in self.free and 0 < r < 64
            self.free.append(r)

    def newlabel(self):
        self.lab_n += 1
        return "L%d" % self.lab_n

    def label(self, name):
        self.labels[name] = len(self.code)

    # ---------------------------------------------------- emitace
    def emit(self, **kw):
        self.code.append(kw)

    def li(self, rd, v):
        lo = v & 0xFFFFFFFF
        slo = lo - (1 << 32) if lo >= 1 << 31 else lo
        hi = (v - slo) >> 32
        self.emit(op=I.OP["SCALAR"], fn=I.SFN["LI"], rd=rd, imm=slo, imm2=hi)

    def s(self, fn, rd=0, a=0, b=0, c=0, imm=0, D=None):
        self.emit(op=I.OP["SCALAR"], fn=I.SFN[fn], rd=rd, rs1=a, rs2=b, rs3=c, imm=imm, D=D)

    def br(self, fn, r, label):
        self.fix.append((len(self.code), label))
        self.emit(op=I.OP["BRANCH"], fn=I.BFN[fn], rs1=r, imm=0)

    def halt(self):
        self.emit(op=I.OP["HALT"])

    def ew(self, fn, n, D, A=None, B=None, C=None, tc="NONE", clamp=0, sym=False, mask=False,
           rs1=0, rs2=0, imm=0):
        fl = I.TC[tc]
        if clamp:
            fl |= I.F_CLAMPSYM if sym else I.F_CLAMP
        if mask:
            fl |= I.F_MASK
        f = I.EFN[fn]
        if f in I.EFN_RNG:
            fl |= I.F_RNG
        self.emit(op=I.OP["EW"], fn=f, flags=fl, n=n, D=D, A=A, B=B, C=C, rs1=rs1, rs2=rs2, imm=imm,
                  imm3=clamp)

    def red(self, fn, n, A=None, B=None, C=None, D=None, rd=0, pre="A", tc="NONE", rs1=0, imm=0):
        fl = (I.RPRE[pre] << I.F_PRE_SHIFT) | I.TC[tc]
        if rd:
            fl |= I.F_TOREG
        self.emit(op=I.OP["RED"], fn=I.RFN[fn], flags=fl, n=n, D=D, A=A, B=B, C=C, rd=rd, rs1=rs1,
                  imm=imm)

    def mm(self, n, D, A, B):
        self.emit(op=I.OP["MM"], n=n, D=D, A=A, B=B)

    def binary(self):
        for i, lab in self.fix:
            self.code[i]["imm"] = self.labels[lab]
        return b"".join(I.encode(**kw) for kw in self.code)

    # ---------------------------------------------------- pomocne vzory
    def amax_reg(self, t, n, rd):
        self.red("AMAX", (n,), A=Opnd(t, (1,)), rd=rd)

    def bitlen_sel(self, rd, ra, sub):
        """rd = ra ? bitlen(ra) - sub : 0  (normalize: posun jen pro nenulovy tenzor)"""
        self.s("BITLEN", rd, ra, imm=-sub)
        self.s("SEL", rd, rd, 0, ra)


# ------------------------------------------------------------------ pohledy
def rows(t, R, Cc, off=0):
    return Opnd(t, (1, Cc), off)


def rowv(t, off=0):
    return Opnd(t, (0, 1), off)


def colv(t, off=0):
    return Opnd(t, (1, 0), off)


def flat(t, off=0):
    return Opnd(t, (1,), off)


def kshift(fan_in):
    return int(math.floor(math.log2(fan_in) / 2 + 0.5))


def bitlen(x):
    return x.bit_length()


# ------------------------------------------------------------------ prekladac modelu
class Compiler:
    def __init__(self, a):
        self.a = a
        d, L, H, ff, Tc, B = a.d, a.layers, a.heads, a.ff, a.ctx, a.batch
        self.d, self.L, self.H, self.ff, self.T, self.B = d, L, H, ff, Tc, B
        self.N = N = B * Tc
        self.dh = dh = d // H
        self.ks = (bitlen(dh) - 1) // 2
        self.sqd = int(math.floor(math.sqrt(d) * 4096 + 0.5))
        self.k_d, self.k_ff = kshift(d), kshift(ff)
        self.split = a.corpus_len - a.val_len
        M = self.M = Mem()
        P = self.P = Prog()
        # parametry (poradi jako golden)
        self.params = []

        def par(name, rows_, cols):
            q = M.alloc("q." + name, rows_ * cols, 2)
            m = M.alloc("m." + name, rows_ * cols, 1)
            g = M.alloc("grad." + name, rows_ * cols, 8)
            self.params.append(dict(name=name, rows=rows_, cols=cols, n=rows_ * cols, q=q, m=m, g=g))
            return len(self.params) - 1

        self.p_tok = par("tok", V, d)
        self.p_pos = par("pos", Tc, d)
        self.p_l = []
        for l in range(L):
            self.p_l.append([par(f"{l}.ln1", 1, d), par(f"{l}.qkv", 3 * d, d), par(f"{l}.proj", d, d),
                             par(f"{l}.ln2", 1, d), par(f"{l}.fc1", ff, d), par(f"{l}.fc2", d, ff)])
        self.p_lnf = par("lnf", 1, d)
        np_ = len(self.params)
        self.EM = M.alloc("em", np_, 8)
        self.GEXP = M.alloc("gexp", np_, 8)
        self.LR = M.alloc("lr", 1, 8)
        self.corpus = M.alloc("corpus", a.corpus_len, 1, True)
        # data kroku
        self.rnd = M.alloc("rnd", B, 8)
        self.ofs = M.alloc("ofs", B, 4)
        self.X = M.alloc("x_idx", N, 1, True)
        self.Y = M.alloc("y_idx", N, 1, True)
        # dopredny pruchod
        A = M.alloc
        self.x = A("x", N * d, 4)
        wide = N * max(3 * d, ff)
        self.o = A("o", N * 3 * d, 4)
        self.y = A("y", wide, 8)
        att = B * H * Tc * Tc
        self.sc = A("sc", att, 8)
        self.yh = A("yh", N * d, 8)
        self.yatt = A("yatt", N * d, 8)
        self.po = A("po", wide, 8)
        self.logits = A("logits", N * V, 8)
        self.n8 = A("n8", N * d, 1)
        self.xhf, self.rrf, self.shf = A("xhf", N * d, 4), A("rrf", N, 4), A("shf", N, 4)
        self.en, self.el = A("en", N, 4), A("el", N, 4)
        self.eo = A("eo", N, 4)
        self.lc = []
        for l in range(L):
            c = dict(xh1=A(f"{l}.xh1", N * d, 4), rr1=A(f"{l}.rr1", N, 4), sh1=A(f"{l}.sh1", N, 4),
                     a1=A(f"{l}.a1", N * d, 1), ea1=A(f"{l}.ea1", N, 4),
                     q8=A(f"{l}.q8", N * d, 1), k8=A(f"{l}.k8", N * d, 1), v8=A(f"{l}.v8", N * d, 1),
                     eq=A(f"{l}.eq", B * H * Tc, 4), ek=A(f"{l}.ek", B * H, 4), ev=A(f"{l}.ev", B * H, 4),
                     p8=A(f"{l}.p8", att, 1, True), a2=A(f"{l}.a2", N * d, 1), ea2=A(f"{l}.ea2", N, 4),
                     xh2=A(f"{l}.xh2", N * d, 4), rr2=A(f"{l}.rr2", N, 4), sh2=A(f"{l}.sh2", N, 4),
                     a3=A(f"{l}.a3", N * d, 1), ea3=A(f"{l}.ea3", N, 4),
                     fo=A(f"{l}.fo", N * ff, 4), a4=A(f"{l}.a4", N * ff, 1), ea4=A(f"{l}.ea4", N, 4))
            self.lc.append(c)
        # pomocne vektory po radcich / hlavach
        R = max(N, B * H * Tc)
        self.t1, self.t2, self.t3, self.t4 = A("t1", R, 8), A("t2", R, 8), A("t3", R, 8), A("t4", R, 8)
        self.e1, self.e2 = A("e1", R, 4), A("e2", R, 4)
        # zpetny pruchod
        self.D = A("D", N * V, 8)
        self.Dt = A("Dt", N * V, 8)
        self.dX, self.dXn = A("dX", N * d, 8), A("dXn", N * d, 8)
        self.g1, self.g2 = A("g1", wide, 8), A("g2", wide, 8)
        self.gx, self.gq, self.gp = A("gx", wide, 8), A("gq", wide, 1), A("gp", wide, 2)
        self.aa = A("aa", wide, 2)
        self.buf = A("buf", N * d, 8)
        self.tt = A("tt", N * d, 8)
        self.dsm, self.ds8, self.dpt = A("dsm", att, 8), A("ds8", att, 1), A("dpt", att, 8)
        self.dy8 = A("dy8", N * d, 1)
        self.dq, self.dk, self.dv = A("dq", N * d, 8), A("dk", N * d, 8), A("dv", N * d, 8)
        self.qa = A("qa", N * d, 2)
        self.G = A("G", N * 3 * d, 8)
        self.dH, self.dA = A("dH", wide, 8), A("dA", wide, 8)
        self.oh = A("oh", N * V, 1)
        self.gc, self.mc = A("gc", self.maxparam(), 8), A("mc", self.maxparam(), 8)
        self.prog_addr = M.top

    def maxparam(self):
        return max(p["n"] for p in self.params)

    # ------------------------------------------------------------ primitiva golden
    def norm_fwd(self, x, xh, rr, sh):
        P, N, d = self.P, self.N, self.d
        P.red("SUM", (d, N), A=rows(x, N, d), D=rowv(self.t1), pre="AA")
        P.ew("NRMS", (N,), D=flat(sh), A=flat(self.t1))
        P.ew("NRMR", (N,), D=flat(rr), A=flat(self.t1), B=flat(sh), imm=self.sqd)
        P.ew("MULRSR", (d, N), D=rows(xh, N, d), A=rows(x, N, d), B=rowv(rr), C=rowv(sh), tc="ADDHALF", imm=6)

    def quant_tok(self, y, R, n, bits, out, s, eadd):
        """po radcich; s[r] = posun + eadd"""
        P = self.P
        P.red("AMAX", (n, R), A=rows(y, R, n), D=rowv(self.t2))
        P.ew("BITLENI", (R,), D=flat(s), A=flat(self.t2), imm=-(bits - 1))
        P.ew("RSR", (n, R), D=rows(out, R, n), A=rows(y, R, n), C=rowv(s), tc="ADD", clamp=bits)
        if eadd:
            P.ew("SEXPR", (R,), D=flat(s), C=flat(s), tc="ADD", imm=eadd)

    def mm_bin(self, a, pi, out):
        """out[n][o] = sum_k a[n][k] sign(W[o][k])"""
        p = self.params[pi]
        O, K = p["rows"], p["cols"]
        self.P.mm((O, self.N, 1, K), D=Opnd(out, (1, O)), A=Opnd(a, (0, K, 0, 1)),
                  B=Opnd(p["q"], (K, 0, 0, 1), mode=I.M_SIGN))

    def normalize(self, src, dst, n, e, bits):
        """dst = normalize(src); e += s (registr)"""
        P = self.P
        ra, rs = P.reg(), P.reg()
        P.amax_reg(src, n, ra)
        P.bitlen_sel(rs, ra, bits)
        P.ew("RSR", (n,), D=flat(dst), A=flat(src), rs1=rs)
        P.s("ADD", e, e, rs)
        P.rel(ra, rs)

    def quant_sr(self, src, dst, n, e, bits=8):
        P = self.P
        ra, rs = P.reg(), P.reg()
        P.amax_reg(src, n, ra)
        P.s("BITLEN", rs, ra, imm=-(bits - 1))
        P.ew("SRD", (n,), D=flat(dst), A=flat(src), rs1=rs, clamp=bits)
        P.s("ADD", e, e, rs)
        P.rel(ra, rs)

    def add_bfp(self, m1, e1, m2, e2, n):
        P = self.P
        a1, a2, t1, t2, rn = P.reg(), P.reg(), P.reg(), P.reg(), P.reg()
        skip = P.newlabel()
        P.amax_reg(m1, n, a1)
        P.amax_reg(m2, n, a2)
        P.li(rn, NONE)
        P.s("BITLEN", t1, a1)
        P.s("ADD", t1, t1, e1)
        P.s("SEL", t1, t1, rn, a1)
        P.s("BITLEN", t2, a2)
        P.s("ADD", t2, t2, e2)
        P.s("SEL", t2, t2, rn, a2)
        P.s("ADD", a1, a1, a2)
        P.br("BEQZ", a1, skip)
        P.s("MAX", t1, t1, t2, )
        P.s("ADD", t1, t1, imm=-40)           # base
        P.s("SUB", t2, t1, e1)                # base - e1
        P.s("SUB", a2, t1, e2)                # base - e2
        P.ew("RSR2ADD", (n,), D=flat(m1), A=flat(m1), B=flat(m2), rs1=t2, rs2=a2)
        P.s("ADD", e1, t1)
        P.label(skip)
        P.rel(a1, a2, t1, t2, rn)

    def st_gexp(self, pi, r):
        self.P.s("ST", a=r, D=Opnd(self.GEXP, off=pi))

    # ------------------------------------------------------------ vyber davky
    def batch(self):
        P, B, Tc = self.P, self.B, self.T
        P.ew("RNG", (B,), D=flat(self.rnd))
        P.ew("MODU", (B,), D=flat(self.ofs), A=flat(self.rnd), imm=self.split - 1 - Tc)
        for t, off in ((self.X, 0), (self.Y, 1)):
            P.ew("COPY", (Tc, B), D=Opnd(t, (1, Tc)),
                 A=Opnd(self.corpus, (1, 0), off=off, gather=Gather(self.ofs, (0, 1), 1)))

    # ------------------------------------------------------------ dopredny pruchod
    def forward(self):
        P, N, d, H, dh, ff, B, Tc = self.P, self.N, self.d, self.H, self.dh, self.ff, self.B, self.T
        tok, pos = self.params[self.p_tok]["q"], self.params[self.p_pos]["q"]
        P.ew("ADD", (d, Tc, B), D=Opnd(self.x, (1, d, Tc * d)),
             A=Opnd(tok, (1, 0, 0), gather=Gather(self.X, (0, 1, Tc), d)), B=Opnd(pos, (1, d, 0)))
        x = self.x
        for l in range(self.L):
            c = self.lc[l]
            pl = self.p_l[l]
            q = lambda k: self.params[pl[k]]["q"]  # noqa: E731
            # --- attention
            self.norm_fwd(x, c["xh1"], c["rr1"], c["sh1"])
            P.ew("MUL", (d, N), D=rows(self.y, N, d), A=rows(c["xh1"], N, d), B=colv(q(0)))
            self.quant_tok(self.y, N, d, 6, c["a1"], c["ea1"], -24)
            self.mm_bin(c["a1"], pl[1], self.o)
            P.ew("SEXPR", (N,), D=flat(self.eo), C=flat(c["ea1"]), tc="ADD", imm=-self.k_d)
            # emax po sekvenci -> e1[b]
            P.red("MAX", (Tc, B), A=Opnd(self.eo, (1, Tc)), D=rowv(self.e1))
            # q po tokenech: radek (b,h,t) delky dh ze sloupcu h*dh v o
            # smycky (k, t, h, b); o[(b*T+t)*3d + h*dh + k]
            oq = Opnd(self.o, (1, 3 * d, dh, Tc * 3 * d))
            P.red("AMAX", (dh, Tc, H, B), A=oq, D=Opnd(self.t2, (0, 1, Tc, H * Tc)))
            P.ew("BITLENI", (B * H * Tc,), D=flat(self.e2), A=flat(self.t2), imm=-7)
            P.ew("RSR", (dh, Tc, H, B), D=Opnd(c["q8"], (1, dh, Tc * dh, H * Tc * dh)), A=oq,
                 C=Opnd(self.e2, (0, 1, Tc, H * Tc)), tc="ADD", clamp=8)
            # eq[bh*T+t] = eo[b*T+t] + s
            P.ew("ADD", (Tc, H, B), D=Opnd(c["eq"], (1, Tc, H * Tc)), A=Opnd(self.eo, (1, 0, Tc)),
                 B=Opnd(self.e2, (1, Tc, H * Tc)))
            # k, v po (sekvence, hlava)
            # posun zarovnani sft[b*T+t] = emax[b] - eo[b*T+t]  (do t3 jako int)
            P.ew("SUB", (Tc, B), D=Opnd(self.t3, (1, Tc)), A=Opnd(self.e1, (0, 1)), B=Opnd(self.eo, (1, Tc)))
            for part, dst, ex in ((1, c["k8"], c["ek"]), (2, c["v8"], c["ev"])):
                src = Opnd(self.o, (1, 3 * d, dh, Tc * 3 * d), off=part * d)
                kv = Opnd(dst, (1, dh, Tc * dh, H * Tc * dh))
                # zarovnani na emax: dst = rsr(src, sft) (prozatim do yh jako int64)
                tmp = Opnd(self.yh, (1, dh, Tc * dh, H * Tc * dh))
                P.ew("RSR", (dh, Tc, H, B), D=tmp, A=src, C=Opnd(self.t3, (0, 1, 0, Tc)), tc="ADD")
                P.red("AMAX", (dh * Tc, H, B), A=Opnd(self.yh, (1, Tc * dh, H * Tc * dh)),
                      D=Opnd(self.t2, (0, 1, H)))
                P.ew("BITLENI", (B * H,), D=flat(self.e2), A=flat(self.t2), imm=-7)
                P.ew("RSR", (dh, Tc, H, B), D=kv, A=tmp, C=Opnd(self.e2, (0, 0, 1, H)), tc="ADD", clamp=8)
                # ex[bh] = emax[b] + s
                P.ew("ADD", (H, B), D=Opnd(ex, (1, H)), A=Opnd(self.e1, (0, 1)), B=Opnd(self.e2, (1, H)))
            # skore: sc[bh][i][j] = q8[bh][i] . k8[bh][j]
            BH = B * H
            P.mm((Tc, Tc, BH, dh), D=Opnd(self.sc, (1, Tc, Tc * Tc)), A=Opnd(c["q8"], (0, dh, Tc * dh, 1)),
                 B=Opnd(c["k8"], (dh, 0, Tc * dh, 1)))
            # posun po radcich (bh,i): sh = ks - 5 - eq - ek[bh]  -> e2
            P.ew("ADD", (Tc, BH), D=Opnd(self.e2, (1, Tc)), A=Opnd(c["eq"], (1, Tc)), B=Opnd(c["ek"], (0, 1)))
            rinf = P.reg()
            P.li(rinf, -(1 << 40))
            P.ew("RSR", (Tc, Tc, BH), D=Opnd(self.sc, (1, Tc, Tc * Tc)), A=Opnd(self.sc, (1, Tc, Tc * Tc)),
                 C=Opnd(self.e2, (0, 1, Tc)), tc="SUB", imm=self.ks - 5, mask=True, rs2=rinf)
            P.rel(rinf)
            P.red("MAX", (Tc, Tc * BH), A=Opnd(self.sc, (1, Tc)), D=rowv(self.t1))
            P.ew("EXP8", (Tc, Tc, BH), D=Opnd(self.sc, (1, Tc, Tc * Tc)), A=Opnd(self.sc, (1, Tc, Tc * Tc)),
                 B=Opnd(self.t1, (0, 1, Tc)), mask=True, rs2=0)
            P.red("SUM", (Tc, Tc * BH), A=Opnd(self.sc, (1, Tc)), D=rowv(self.t2))
            P.ew("RCP", (Tc * BH,), D=flat(self.t2), A=flat(self.t2), imm=24)
            P.ew("MULRSR", (Tc, Tc * BH), D=Opnd(c["p8"], (1, Tc)), A=Opnd(self.sc, (1, Tc)), B=rowv(self.t2),
                 imm=16, clamp=9)
            # yh[bh][i][k] = sum_j p8[bh][i][j] v8[bh][j][k]
            P.mm((dh, Tc, BH, Tc), D=Opnd(self.yh, (1, dh, Tc * dh)), A=Opnd(c["p8"], (0, Tc, Tc * Tc, 1)),
                 B=Opnd(c["v8"], (1, 0, Tc * dh, dh)))
            # eatt[b] = max_h ev[b,h] - 8  -> e1[b]
            P.red("MAX", (H, B), A=Opnd(c["ev"], (1, H)), D=rowv(self.e1))
            P.ew("SEXPR", (B,), D=flat(self.e1), C=flat(self.e1), tc="ADD", imm=-8)
            # sft2[b*H+h] = eatt[b] - ev[bh] + 8  -> e2
            P.ew("SUB", (H, B), D=Opnd(self.e2, (1, H)), A=Opnd(self.e1, (0, 1)), B=Opnd(c["ev"], (1, H)))
            P.ew("SEXPR", (B * H,), D=flat(self.e2), C=flat(self.e2), tc="ADD", imm=8)
            # yatt[(b*T+t)*d + h*dh + k] = rsr(yh[((b*H+h)*T+t)*dh+k], sft2[b*H+h]); smycky (k, h, t, b)
            P.ew("RSR", (dh, H, Tc, B), D=Opnd(self.yatt, (1, dh, d, Tc * d)),
                 A=Opnd(self.yh, (1, Tc * dh, dh, H * Tc * dh)), C=Opnd(self.e2, (0, 1, 0, H)), tc="ADD")
            self.quant_tok(self.yatt, N, d, 6, c["a2"], c["ea2"], 0)
            # ea2[b*T+t] += eatt[b]
            P.ew("ADD", (Tc, B), D=Opnd(c["ea2"], (1, Tc)), A=Opnd(c["ea2"], (1, Tc)), B=Opnd(self.e1, (0, 1)))
            self.mm_bin(c["a2"], pl[2], self.po)
            # x += rsr(po, k - 16 - ea2)
            P.ew("ADDRSR", (d, N), D=rows(x, N, d), A=rows(x, N, d), B=rows(self.po, N, d), C=rowv(c["ea2"]),
                 tc="SUB", imm=self.k_d - 16)
            # --- MLP
            self.norm_fwd(x, c["xh2"], c["rr2"], c["sh2"])
            P.ew("MUL", (d, N), D=rows(self.y, N, d), A=rows(c["xh2"], N, d), B=colv(q(3)))
            self.quant_tok(self.y, N, d, 6, c["a3"], c["ea3"], -24)
            self.mm_bin(c["a3"], pl[4], c["fo"])
            P.ew("RELU2", (N * ff,), D=flat(self.y), A=flat(c["fo"]))
            self.quant_tok(self.y, N, ff, 6, c["a4"], c["ea4"], 0)
            # ea4 += 2*(ea3 - k_fc1)
            P.ew("ADD", (N,), D=flat(c["ea4"]), A=flat(c["ea4"]), B=flat(c["ea3"]))
            P.ew("ADD", (N,), D=flat(c["ea4"]), A=flat(c["ea4"]), B=flat(c["ea3"]))
            P.ew("SEXPR", (N,), D=flat(c["ea4"]), C=flat(c["ea4"]), tc="ADD", imm=-2 * self.k_d)
            self.mm_bin(c["a4"], pl[5], self.po)
            P.ew("ADDRSR", (d, N), D=rows(x, N, d), A=rows(x, N, d), B=rows(self.po, N, d), C=rowv(c["ea4"]),
                 tc="SUB", imm=self.k_ff - 16)
        # hlava
        self.norm_fwd(x, self.xhf, self.rrf, self.shf)
        P.ew("MUL", (d, N), D=rows(self.y, N, d), A=rows(self.xhf, N, d), B=colv(self.params[self.p_lnf]["q"]))
        self.quant_tok(self.y, N, d, 8, self.n8, self.en, -24)
        P.ew("SEXPR", (N,), D=flat(self.el), C=flat(self.en), tc="ADD", imm=-16)
        P.mm((V, N, 1, d), D=Opnd(self.logits, (1, V)), A=Opnd(self.n8, (0, d, 0, 1)), B=Opnd(tok, (d, 0, 0, 1)))

    # ------------------------------------------------------------ zpetny pruchod
    def norm_bwd(self, dy, edy, xh, rr, sh, pi, dx, edx):
        """dy (N x d, exponent v registru edy) -> grad[pi], gexp[pi]; dx, edx (registr)"""
        P, N, d = self.P, self.N, self.d
        g = self.params[pi]["q"]
        e, emaxt = P.reg(), P.reg()
        P.s("ADD", e, edy)
        self.normalize(dy, self.buf, N * d, e, 16)
        P.red("SUM", (N, d), A=Opnd(self.buf, (d, 1)), B=Opnd(xh, (d, 1)), D=Opnd(self.params[pi]["g"], (0, 1)),
              pre="AB")
        P.s("ADD", emaxt, e, imm=-12)
        self.st_gexp(pi, emaxt)
        P.red("MAX", (N,), C=flat(sh), rd=emaxt, pre="SH", tc="SUBHALF", rs1=e, imm=-14)
        P.red("SUM", (d, N), A=rows(self.buf, N, d), B=colv(g), C=rows(xh, N, d), D=rowv(self.t1), pre="ABC")
        P.ew("MEANDIV", (N,), D=flat(self.t1), A=flat(self.t1), imm=d)
        P.ew("RSR", (N,), D=flat(self.t1), A=flat(self.t1), imm=12)
        P.ew("MULRSR", (d, N), D=rows(self.tt, N, d), A=rows(xh, N, d), B=rowv(self.t1), imm=12)
        P.ew("MULSUB", (d, N), D=rows(self.tt, N, d), A=rows(self.buf, N, d), B=colv(g), C=rows(self.tt, N, d))
        P.ew("SEXPR", (N,), D=flat(self.e1), C=flat(sh), tc="SUBHALF", rs1=e, imm=-14)
        P.ew("MULRSR", (d, N), D=rows(dx, N, d), A=rows(self.tt, N, d), B=rowv(rr), C=rowv(self.e1), tc="SUB",
             rs1=emaxt)
        P.s("ADD", edx, emaxt)
        P.rel(e, emaxt)

    def lin_bwd(self, G, eG, a, ea, pi, k, dx, edx):
        P, N = self.P, self.N
        p = self.params[pi]
        O, K = p["rows"], p["cols"]
        n_o = N * O
        e, rA, s2, base = P.reg(), P.reg(), P.reg(), P.reg()
        zero, done = P.newlabel(), P.newlabel()
        P.s("ADD", e, eG)
        self.normalize(G, self.gx, n_o, e, 16)
        P.amax_reg(self.gx, n_o, rA)
        P.br("BEQZ", rA, zero)
        P.s("BITLEN", s2, rA, imm=-7)
        P.ew("SRD", (n_o,), D=flat(self.gq), A=flat(self.gx), rs1=s2, clamp=8)
        # dx[n][k] = sum_o gq[n][o] sign(W[o][k])
        P.mm((K, N, 1, O), D=Opnd(dx, (1, K)), A=Opnd(self.gq, (0, O, 0, 1)),
             B=Opnd(p["q"], (1, 0, 0, K), mode=I.M_SIGN))
        P.s("ADD", edx, e, s2, imm=-k)
        P.ew("POW2Q", (n_o,), D=flat(self.gp), A=flat(self.gx), rs1=s2)
        P.red("MAX", (N,), A=flat(ea), rd=base)
        P.s("ADD", base, base, imm=-8)
        P.ew("RSR", (K, N), D=rows(self.aa, N, K), A=rows(a, N, K), C=rowv(ea), tc="SUB", rs1=base)
        # grad[o][k] = sum_n gp[n][o] aa[n][k]
        P.mm((K, O, 1, N), D=Opnd(p["g"], (1, K)), A=Opnd(self.gp, (0, 1, 0, O)), B=Opnd(self.aa, (1, 0, 0, K)))
        P.s("ADD", base, base, e)
        P.s("ADD", base, base, s2)
        self.st_gexp(pi, base)
        P.br("JMP", 0, done)
        P.label(zero)
        P.ew("SEXPR", (N * K,), D=flat(dx))
        P.li(edx, 0)
        P.ew("SEXPR", (p["n"],), D=flat(p["g"]))
        self.st_gexp(pi, 0)
        P.label(done)
        P.rel(e, rA, s2, base)

    def attn_bwd(self, c, dY, edY, eG):
        P, N, d, H, dh, B, Tc = self.P, self.N, self.d, self.H, self.dh, self.B, self.T
        BH, TT = B * H, Tc * Tc
        e, ey, evmax, eds, ekmax, qbmax, r1, r2 = (P.reg() for _ in range(8))
        P.s("ADD", e, edY)
        self.normalize(dY, self.g2, N * d, e, 16)
        # dy (b,t,h,k) -> (b,h,t,k)
        P.ew("COPY", (dh, Tc, H, B), D=Opnd(self.g1, (1, dh, Tc * dh, H * Tc * dh)),
             A=Opnd(self.g2, (1, d, dh, Tc * d)))
        P.s("ADD", ey, e)
        self.quant_sr(self.g1, self.dy8, N * d, ey)
        P.red("MAX", (BH,), A=flat(c["ev"]), rd=evmax)
        # dv[bh][j][k] = sum_i p8[bh][i][j] dy[bh][i][k]
        P.mm((dh, Tc, BH, Tc), D=Opnd(self.dv, (1, dh, Tc * dh)), A=Opnd(c["p8"], (0, 1, TT, Tc)),
             B=Opnd(self.dy8, (1, 0, Tc * dh, dh)))
        # dp[bh][i][j] = sum_k dy[bh][i][k] v8[bh][j][k]
        P.mm((Tc, Tc, BH, dh), D=Opnd(self.dpt, (1, Tc, TT)), A=Opnd(self.dy8, (0, dh, Tc * dh, 1)),
             B=Opnd(c["v8"], (dh, 0, Tc * dh, 1)))
        P.red("SUM", (Tc, Tc * BH), A=Opnd(c["p8"], (1, Tc)), B=Opnd(self.dpt, (1, Tc)), D=rowv(self.t1), pre="AB")
        P.ew("RSR", (Tc * BH,), D=flat(self.t1), A=flat(self.t1), imm=8)
        P.ew("SUB", (Tc, Tc * BH), D=Opnd(self.dpt, (1, Tc)), A=Opnd(self.dpt, (1, Tc)), B=rowv(self.t1))
        P.ew("MULRSR", (Tc, Tc, BH), D=Opnd(self.dsm, (1, Tc, TT)), A=Opnd(c["p8"], (1, Tc, TT)),
             B=Opnd(self.dpt, (1, Tc, TT)), C=Opnd(c["ev"], (0, 0, 1)), tc="SUB", rs1=evmax)
        P.s("ADD", eds, ey, evmax, imm=-8)
        self.normalize(self.dsm, self.dsm, BH * TT, eds, 16)
        self.quant_sr(self.dsm, self.ds8, BH * TT, eds)
        # qb[bh] = max_t eq - 8 -> e1 ; ekmax, qbmax
        P.red("MAX", (BH,), A=flat(c["ek"]), rd=ekmax)
        P.red("MAX", (Tc, BH), A=Opnd(c["eq"], (1, Tc)), D=rowv(self.e1))
        P.ew("SEXPR", (BH,), D=flat(self.e1), C=flat(self.e1), tc="ADD", imm=-8)
        P.red("MAX", (BH,), A=flat(self.e1), rd=qbmax)
        # qa = rsr(q8, qb[bh] - eq[bh,t]) ; posun do e2[bh*T+t]
        P.ew("SUB", (Tc, BH), D=Opnd(self.e2, (1, Tc)), A=Opnd(self.e1, (0, 1)), B=Opnd(c["eq"], (1, Tc)))
        P.ew("RSR", (dh, Tc * BH), D=Opnd(self.qa, (1, dh)), A=Opnd(c["q8"], (1, dh)), C=Opnd(self.e2, (0, 1)),
             tc="ADD")
        # dq[bh][i][k] = rsr(sum_j ds8[bh][i][j] k8[bh][j][k], ekmax - ek[bh])
        P.mm((dh, Tc, BH, Tc), D=Opnd(self.dq, (1, dh, Tc * dh)), A=Opnd(self.ds8, (0, Tc, TT, 1)),
             B=Opnd(c["k8"], (1, 0, Tc * dh, dh)))
        P.ew("RSR", (Tc * dh, BH), D=Opnd(self.dq, (1, Tc * dh)), A=Opnd(self.dq, (1, Tc * dh)),
             C=Opnd(c["ek"], (0, 1)), tc="SUB", rs1=ekmax)
        # dk[bh][j][k] = rsr(sum_i ds8[bh][i][j] qa[bh][i][k], qbmax - qb[bh])
        P.mm((dh, Tc, BH, Tc), D=Opnd(self.dk, (1, dh, Tc * dh)), A=Opnd(self.ds8, (0, 1, TT, Tc)),
             B=Opnd(self.qa, (1, 0, Tc * dh, dh)))
        P.ew("RSR", (Tc * dh, BH), D=Opnd(self.dk, (1, Tc * dh)), A=Opnd(self.dk, (1, Tc * dh)),
             C=Opnd(self.e1, (0, 1)), tc="SUB", rs1=qbmax)
        # exponenty: pe = (eds - ks + ekmax, eds - ks + qbmax, ey - 8)
        pe = [P.reg(), P.reg(), P.reg()]
        P.s("ADD", pe[0], eds, ekmax, imm=-self.ks)
        P.s("ADD", pe[1], eds, qbmax, imm=-self.ks)
        P.s("ADD", pe[2], ey, imm=-8)
        top, rn = P.reg(), P.reg()
        P.li(rn, NONE)
        P.s("ADD", top, rn)
        for i, t in enumerate((self.dq, self.dk, self.dv)):
            P.amax_reg(t, N * d, r1)
            P.s("BITLEN", r2, r1)
            P.s("ADD", r2, r2, pe[i])
            P.s("SEL", r2, r2, rn, r1)
            P.s("MAX", top, top, r2)
        zero, done = P.newlabel(), P.newlabel()
        P.s("SUB", r1, top, rn)
        P.br("BEQZ", r1, zero)
        P.s("ADD", top, top, imm=-40)          # base
        for i, t in enumerate((self.dq, self.dk, self.dv)):
            P.s("SUB", r1, top, pe[i])
            # G[(b*T+t)*3d + i*d + h*dh + k] = rsr(pm[((b*H+h)*T+t)*dh + k], base - pe); smycky (k, t, h, b)
            P.ew("RSR", (dh, Tc, H, B), D=Opnd(self.G, (1, 3 * d, dh, Tc * 3 * d), off=i * d),
                 A=Opnd(t, (1, dh, Tc * dh, H * Tc * dh)), rs1=r1)
        P.s("ADD", eG, top)
        P.br("JMP", 0, done)
        P.label(zero)
        P.ew("SEXPR", (N * 3 * d,), D=flat(self.G))
        P.li(eG, 0)
        P.label(done)
        P.rel(e, ey, evmax, eds, ekmax, qbmax, r1, r2, top, rn, *pe)

    def lion(self):
        P = self.P
        lr, ag, am, eg, em, tg, tm, rn = (P.reg() for _ in range(8))
        P.s("LD", lr, D=Opnd(self.LR))
        P.li(rn, NONE)
        for pi, p in enumerate(self.params):
            n, g, m, q = p["n"], p["g"], p["m"], p["q"]
            skip = P.newlabel()
            P.amax_reg(g, n, ag)
            P.amax_reg(m, n, am)
            P.s("ADD", tg, ag, am)
            P.br("BEQZ", tg, skip)
            P.s("LD", eg, D=Opnd(self.GEXP, off=pi))
            P.s("LD", em, D=Opnd(self.EM, off=pi))
            P.s("BITLEN", tg, ag)
            P.s("ADD", tg, tg, eg)
            P.s("SEL", tg, tg, rn, ag)
            P.s("BITLEN", tm, am)
            P.s("ADD", tm, tm, em)
            P.s("SEL", tm, tm, rn, am)
            P.s("MAX", tg, tg, tm)
            P.s("ADD", tg, tg, imm=-30)        # base
            P.s("SUB", tm, tg, eg)
            P.ew("RSR", (n,), D=flat(self.gc), A=flat(g), rs1=tm)
            P.s("SUB", tm, tg, em)
            P.ew("RSR", (n,), D=flat(self.mc), A=flat(m), rs1=tm)
            P.ew("LIONQ", (n,), D=flat(q), A=flat(q), B=flat(self.mc), C=flat(self.gc), rs1=lr, clamp=16, sym=True)
            P.ew("MULADDI", (n,), D=flat(g), A=flat(self.mc), B=flat(self.gc), imm=127)
            P.s("ADD", tg, tg, imm=-7)         # e = base - 7
            self.quant_sr(g, m, n, tg)
            P.s("ST", a=tg, D=Opnd(self.EM, off=pi))
            P.label(skip)
        P.rel(lr, ag, am, eg, em, tg, tm, rn)

    def backward(self):
        P, N, d, ff, B, Tc = self.P, self.N, self.d, self.ff, self.B, self.T
        tok = self.params[self.p_tok]["q"]
        log2n = bitlen(N) - 1
        # softmax + CE
        P.ew("RSR", (V, N), D=rows(self.Dt, N, V), A=rows(self.logits, N, V), C=rowv(self.el), tc="SUB", imm=-8)
        P.red("MAX", (V, N), A=rows(self.Dt, N, V), D=rowv(self.t1))
        P.ew("EXP16", (V, N), D=rows(self.D, N, V), A=rows(self.Dt, N, V), B=rowv(self.t1))
        P.red("SUM", (V, N), A=rows(self.D, N, V), D=rowv(self.t2))
        P.ew("RCP", (N,), D=flat(self.t2), A=flat(self.t2), imm=31)
        P.ew("MULRSR", (V, N), D=rows(self.D, N, V), A=rows(self.D, N, V), B=rowv(self.t2), imm=16)
        P.ew("OHSUB", (V, N), D=rows(self.D, N, V), A=rows(self.D, N, V), B=rowv(self.Y), imm=15)
        eD = -15 - log2n
        # hlava svazana s embeddingem
        base, eX, eH, eA, eXn, eG = (P.reg() for _ in range(6))
        P.red("MAX", (N,), A=flat(self.en), rd=base)
        P.s("ADD", base, base, imm=-8)
        P.ew("RSR", (d, N), D=rows(self.aa, N, d), A=rows(self.n8, N, d), C=rowv(self.en), tc="SUB", rs1=base)
        g0 = self.params[self.p_tok]["g"]
        P.mm((d, V, 1, N), D=Opnd(g0, (1, d)), A=Opnd(self.D, (0, 1, 0, V)), B=Opnd(self.aa, (1, 0, 0, d)))
        P.s("ADD", base, base, imm=eD)
        self.st_gexp(self.p_tok, base)
        # dn[n][k] = sum_v D[n][v] tok[v][k]
        P.mm((d, N, 1, V), D=Opnd(self.g1, (1, d)), A=Opnd(self.D, (0, V, 0, 1)), B=Opnd(tok, (1, 0, 0, d)))
        P.li(eA, eD - 16)
        self.norm_bwd(self.g1, eA, self.xhf, self.rrf, self.shf, self.p_lnf, self.dX, eX)
        for l in range(self.L - 1, -1, -1):
            c, pl = self.lc[l], self.p_l[l]
            # --- MLP
            self.lin_bwd(self.dX, eX, c["a4"], c["ea4"], pl[5], self.k_ff, self.dH, eH)
            # emaxt = max_n (eH + 1 + ea3 - k_fc1) ; y = rsr(dH * relu(fo), emaxt - (...))
            P.red("MAX", (N,), C=flat(c["ea3"]), rd=base, pre="SH", tc="ADD", rs1=eH, imm=1 - self.k_d)
            P.ew("SEXPR", (N,), D=flat(self.e1), C=flat(c["ea3"]), tc="ADD", rs1=eH, imm=1 - self.k_d)
            P.ew("MULRELURSR", (ff, N), D=rows(self.y, N, ff), A=rows(self.dH, N, ff), B=rows(c["fo"], N, ff),
                 C=rowv(self.e1), tc="SUB", rs1=base)
            self.lin_bwd(self.y, base, c["a3"], c["ea3"], pl[4], self.k_d, self.dA, eA)
            self.norm_bwd(self.dA, eA, c["xh2"], c["rr2"], c["sh2"], pl[3], self.dXn, eXn)
            self.add_bfp(self.dX, eX, self.dXn, eXn, N * d)
            # --- attention
            self.lin_bwd(self.dX, eX, c["a2"], c["ea2"], pl[2], self.k_d, self.dA, eA)
            self.attn_bwd(c, self.dA, eA, eG)
            self.lin_bwd(self.G, eG, c["a1"], c["ea1"], pl[1], self.k_d, self.dA, eA)
            self.norm_bwd(self.dA, eA, c["xh1"], c["rr1"], c["sh1"], pl[0], self.dXn, eXn)
            self.add_bfp(self.dX, eX, self.dXn, eXn, N * d)
        # embedding: dXn[v][k] = sum_n [idx[n] == v] dX[n][k]  (int64 -> RED, ne MM)
        P.ew("ONEHOT", (V, N), D=rows(self.oh, N, V), A=rowv(self.X))
        P.red("SUM", (N, d, V), A=Opnd(self.oh, (V, 0, 1)), B=Opnd(self.dX, (d, 1, 0)),
              D=Opnd(self.dXn, (0, 1, d)), pre="AB")
        P.s("LD", base, D=Opnd(self.GEXP, off=self.p_tok))
        self.add_bfp(g0, base, self.dXn, eX, V * d)
        self.st_gexp(self.p_tok, base)
        P.red("SUM", (B, d, Tc), A=Opnd(self.dX, (Tc * d, 1, d)), D=Opnd(self.params[self.p_pos]["g"], (0, 1, d)))
        self.st_gexp(self.p_pos, eX)
        P.rel(base, eX, eH, eA, eXn, eG)

    def build(self):
        self.marks = {}
        self.batch()
        self.marks["fwd"] = len(self.P.code)
        self.forward()
        self.marks["bwd"] = len(self.P.code)
        self.backward()
        self.marks["lion"] = len(self.P.code)
        self.lion()
        self.P.halt()
        return self.P.binary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=648)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus-len", type=int, required=True)
    ap.add_argument("--val-len", type=int, default=11152)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    c = Compiler(a)
    code = c.build()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".bin", "wb") as f:
        f.write(code)
    total = c.prog_addr + len(code)
    with open(a.out + ".layout", "w") as f:
        f.write(f"isa {I.ISA_VERSION}\n")
        f.write(f"config d {a.d} layers {a.layers} heads {a.heads} ff {a.ff} ctx {a.ctx} batch {a.batch} "
                f"corpus_len {a.corpus_len} val_len {a.val_len}\n")
        f.write(f"prog {c.prog_addr} {len(code) // (I.IWORDS * 4)}\n")
        f.write(f"memsize {(total + 4095) // 4096 * 4096}\n")
        f.write(f"params {len(c.params)}\n")
        for k, v in c.marks.items():
            f.write(f"mark {k} {v}\n")
        for t in c.M.tensors:
            f.write(f"t {t.name} {t.addr} {t.width} {int(t.unsigned)} {t.n}\n")
    print(f"program: {len(code) // (I.IWORDS * 4)} instrukci, pamet {total / 2**20:.1f} MiB -> {a.out}.bin/.layout")


if __name__ == "__main__":
    main()
