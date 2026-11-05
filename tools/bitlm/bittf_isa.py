#!/usr/bin/env python3
"""BitTF ISA v1 - jediny zdroj definic instrukci (Python prekladac, C simulator, RTL).

Popis viz docs/bittf_isa.md. `python3 tools/bitlm/bittf_isa.py` vygeneruje
sw/bittf/isa/bt_isa_ops.h (C) a rtl/bittf/bittf_isa.vh (Verilog).

Instrukce = 48 slov po 32 bitech (192 B), little endian.
"""
import os
import struct

ISA_VERSION = 1
IWORDS = 48

# ---------------------------------------------------------------- opcody
OP = dict(NOP=0, HALT=1, SCALAR=2, BRANCH=3, EW=4, RED=5, MM=6)

# skalarni funkce (op SCALAR)
SFN = dict(LI=0, ADD=1, SUB=2, MAX=3, MIN=4, BITLEN=5, SEL=6, SHRA=7, LD=8, ST=9,
           CSRR=10, CSRW=11, NE=12)
# vetveni (op BRANCH), cil = imm (index instrukce)
BFN = dict(BEQZ=0, BNEZ=1, JMP=2)
# CSR
CSR = dict(SEED=0, CTR=1, ERR=2, CYCLES=3)

# funkce po prvcich (op EW); SH = imm + R[rs1] + tc(C)
EFN = dict(COPY=0, ADD=1, SUB=2, MUL=3, RSR=4, ADDRSR=5, MULRSR=6, MULRELURSR=7, MULSUB=8,
           SRD=9, POW2Q=10, BITLENI=11, SEXPR=12, NRMS=13, NRMR=14, EXP8=15, RCP=16,
           EXP16=17, OHSUB=18, ONEHOT=19, RELU2=20, MEANDIV=21, RSR2ADD=22, LIONQ=23,
           MULADDI=24, RNG=25, MODU=26)
EFN_RNG = {EFN["SRD"], EFN["POW2Q"], EFN["RNG"]}

# ktere operandy funkce cte (C navic vzdy, kdyz tc != NONE)
_E = EFN
EFN_NEED_A = {v for k, v in _E.items() if k not in ("SEXPR", "RNG")}
EFN_NEED_B = {_E[k] for k in ("ADD", "SUB", "MUL", "ADDRSR", "MULRSR", "MULRELURSR", "MULSUB", "NRMR",
                              "EXP8", "EXP16", "OHSUB", "RSR2ADD", "LIONQ", "MULADDI")}
EFN_NEED_C = {_E[k] for k in ("MULSUB", "LIONQ")}

# redukce (op RED): kombinace a predzpracovani
RFN = dict(SUM=0, MAX=1, AMAX=2)
RPRE = dict(A=0, AA=1, AB=2, ABC=3, SH=4)

# tc(C): pricteni operandu C do posunu
TC = dict(NONE=0, ADD=1, SUB=2, ADDHALF=3, SUBHALF=4)

# priznaky (bity v horni pulce slova 0)
F_TC_MASK = 0x7          # EW, RED(SH)
F_CLAMP = 1 << 3         # EW: orezat na znamenkove cislo s imm3 bity
F_CLAMPSYM = 1 << 4      # EW: symetricky +-(2^(imm3-1) - 1)
F_MASK = 1 << 5          # EW: i0 > i1 -> vysledek R[rs2]
F_RNG = 1 << 6           # EW: novy proud nahodnych cisel (nastavi prekladac podle fn)
F_PRE_SHIFT = 8          # RED: bity 10:8 = predzpracovani
F_TOREG = 1 << 11        # RED: vysledek do R[rd]

# konfigurace operandu
M_NORMAL, M_SIGN = 0, 1  # MM: B jako znamenko (q >= 0 -> +1, jinak -1)

# rozlozeni slov instrukce
W_OPN = 5                # operand X (D=0, A=1, B=2, C=3) zacina na 5 + 6*X: base, cfg, s0..s3
W_GBASE, W_GCFG, W_GSTR, W_GSCALE = 29, 30, 31, 35
W_REGS, W_IMM, W_IMM2, W_IMM3 = 36, 37, 38, 39

NREG = 64


def lw_of(width):
    return {1: 0, 2: 1, 4: 2, 8: 3}[width]


class Opnd:
    """Operand: tenzor (base, sirka, znamenko) + kroky pro 4 smycky + volitelny gather."""

    def __init__(self, t, strides=(0, 0, 0, 0), off=0, mode=M_NORMAL, gather=None):
        self.t, self.off, self.mode, self.gather = t, off, mode, gather
        self.strides = tuple(strides) + (0,) * (4 - len(strides))

    def words(self):
        cfg = lw_of(self.t.width) | (4 if self.t.unsigned else 0) | (8 if self.gather else 0) | (self.mode << 4)
        return [self.t.addr + self.off * self.t.width, cfg] + [s & 0xFFFFFFFF for s in self.strides]


class Gather:
    """Souradnice z indexoveho tenzoru: k offsetu operandu se pricte idx[g . i] * scale."""

    def __init__(self, idx, strides, scale, off=0):
        self.idx, self.off, self.scale = idx, off, scale
        self.strides = tuple(strides) + (0,) * (4 - len(strides))


def encode(op, fn=0, flags=0, n=(1, 1, 1, 1), D=None, A=None, B=None, C=None,
           rs1=0, rs2=0, rd=0, rs3=0, imm=0, imm2=0, imm3=0):
    w = [0] * IWORDS
    w[0] = op | (fn << 8) | (flags << 16)
    n = tuple(n) + (1,) * (4 - len(n))
    for k in range(4):
        assert n[k] >= 1
        w[1 + k] = n[k]
    g = None
    for x, o in enumerate((D, A, B, C)):
        if o is None:
            continue
        w[W_OPN + 6 * x:W_OPN + 6 * x + 6] = o.words()
        if o.gather:
            assert g is None, "jen jeden gather na instrukci"
            g = o.gather
    if g:
        w[W_GBASE] = g.idx.addr + g.off * g.idx.width
        w[W_GCFG] = lw_of(g.idx.width) | (4 if g.idx.unsigned else 0)
        for k in range(4):
            w[W_GSTR + k] = g.strides[k] & 0xFFFFFFFF
        w[W_GSCALE] = g.scale & 0xFFFFFFFF
    w[W_REGS] = rs1 | (rs2 << 8) | (rd << 16) | (rs3 << 24)
    w[W_IMM], w[W_IMM2], w[W_IMM3] = imm & 0xFFFFFFFF, imm2 & 0xFFFFFFFF, imm3 & 0xFFFFFFFF
    return struct.pack("<%dI" % IWORDS, *w)


# ---------------------------------------------------------------- generovani hlavicek
def _defs(prefix, d, fmt):
    return "".join(fmt(prefix, k, v) for k, v in d.items())


def gen_c():
    c = lambda p, k, v: f"#define {p}{k} {v}\n"  # noqa: E731
    s = ["/* Vygenerovano tools/bitlm/bittf_isa.py - neupravovat rucne. */\n",
         "#ifndef BT_ISA_OPS_H\n#define BT_ISA_OPS_H\n",
         f"#define BT_ISA_VERSION {ISA_VERSION}\n#define BT_IWORDS {IWORDS}\n#define BT_NREG {NREG}\n"]
    for p, d in (("BT_OP_", OP), ("BT_S_", SFN), ("BT_B_", BFN), ("BT_CSR_", CSR), ("BT_E_", EFN),
                 ("BT_R_", RFN), ("BT_RP_", RPRE), ("BT_TC_", TC)):
        s.append(_defs(p, d, c))
    for k, v in dict(F_TC_MASK=F_TC_MASK, F_CLAMP=F_CLAMP, F_CLAMPSYM=F_CLAMPSYM, F_MASK=F_MASK,
                     F_RNG=F_RNG, F_PRE_SHIFT=F_PRE_SHIFT, F_TOREG=F_TOREG, M_NORMAL=M_NORMAL,
                     M_SIGN=M_SIGN, W_OPN=W_OPN, W_GBASE=W_GBASE, W_GCFG=W_GCFG, W_GSTR=W_GSTR,
                     W_GSCALE=W_GSCALE, W_REGS=W_REGS, W_IMM=W_IMM, W_IMM2=W_IMM2,
                     W_IMM3=W_IMM3).items():
        s.append(f"#define BT_{k} {v}\n")
    for nm, st in (("NEED_A", EFN_NEED_A), ("NEED_B", EFN_NEED_B), ("NEED_C", EFN_NEED_C)):
        m = sum(1 << v for v in st)
        s.append(f"#define BT_EFN_{nm} 0x{m:08x}u\n")
    m = sum(1 << v for v in EFN_RNG)
    s.append(f"#define BT_EFN_RNGMASK 0x{m:08x}u\n")
    s.append("#endif\n")
    return "".join(s)


def gen_v():
    v = lambda p, k, x: f"localparam {p}{k} = {x};\n"  # noqa: E731
    s = ["// Vygenerovano tools/bitlm/bittf_isa.py - neupravovat rucne.\n"]
    for p, d in (("OP_", OP), ("S_", SFN), ("B_", BFN), ("CSR_", CSR), ("E_", EFN), ("R_", RFN),
                 ("RP_", RPRE), ("TC_", TC)):
        s.append(_defs(p, d, v))
    for k, x in dict(F_CLAMP_BIT=3, F_CLAMPSYM_BIT=4, F_MASK_BIT=5, F_RNG_BIT=6, F_TOREG_BIT=11,
                     M_NORMAL=M_NORMAL, M_SIGN=M_SIGN, IWORDS=IWORDS).items():
        s.append(f"localparam {k} = {x};\n")
    for nm, st in (("NEED_A", EFN_NEED_A), ("NEED_B", EFN_NEED_B), ("NEED_C", EFN_NEED_C),
                   ("RNGMASK", EFN_RNG)):
        m = sum(1 << v for v in st)
        s.append(f"localparam [31:0] EFN_{nm} = 32'h{m:08x};\n")
    return "".join(s)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    for path, text in (("sw/bittf/isa/bt_isa_ops.h", gen_c()), ("rtl/bittf/bittf_isa.vh", gen_v())):
        p = os.path.join(root, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        print("zapsano", path)
