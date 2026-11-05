/*
 * BitTF ISA v1 - referencni simulator. Aritmetika je presne ta z golden modelu
 * (sw/bittf/golden/bittf_ref.c): rsr, srd, bitlen, pow2q, tabulky RSQ/EXP8/EXP16.
 */
#include "bt_isa.h"

#include <stdio.h>
#include <string.h>

#include "../golden/bittf_tables.h"

/* ------------------------------------------------------------------ aritmetika (= golden) */
static inline int64_t shl(int64_t a, int64_t s) { return (int64_t)((uint64_t)a << (s > 63 ? 63 : s)); }

static inline int64_t rsr(int64_t a, int64_t s)
{
    if (s > 0) {
        int sp = s > 62 ? 62 : (int)s;
        return (a + ((1LL << sp) >> 1)) >> sp;
    }
    return shl(a, -s);
}

static inline int64_t srd(int64_t a, int64_t s, uint32_t r)
{
    if (s > 0) {
        int sp = s > 32 ? 32 : (int)s;
        return (a + ((int64_t)r & ((1LL << sp) - 1))) >> sp;
    }
    return shl(a, -s);
}

static inline int bitlen(int64_t x) { return x ? 64 - __builtin_clzll((uint64_t)x) : 0; }
static inline int64_t clampi(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int64_t sgn(int64_t v) { return (v > 0) - (v < 0); }
static inline int64_t iabs(int64_t v) { return v < 0 ? -v : v; }
static inline int64_t floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
        q--;
    return q;
}

static inline uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

static inline int64_t pow2q1(int64_t m, int lo_e, uint32_t r)
{
    int64_t a = iabs(m), v;
    int e = bitlen(a) - 1;
    if (e < 0)
        e = 0;
    int64_t b = 1LL << e, lo = 1LL << lo_e;
    if (a >= lo)
        v = ((int64_t)r & (b - 1)) < a - b ? b << 1 : b;
    else
        v = ((int64_t)r & (lo - 1)) < a ? lo : 0;
    return sgn(m) * (v >> lo_e);
}

/* ------------------------------------------------------------------ pamet */
const char *bt_err_name(int e)
{
    static const char *n[] = {"OK", "hodnota se nevejde do sirky ulozeni", "preteceni akumulatoru MM",
                              "neznama instrukce", "pristup mimo pamet", "limit instrukci"};
    return e >= 0 && e <= 5 ? n[e] : "?";
}

int64_t bt_load(const bt_cpu *c, uint64_t addr, int lw, int uns)
{
    const uint8_t *p = c->mem + addr;
    switch (lw) {
    case 0: return uns ? (int64_t)p[0] : (int64_t)(int8_t)p[0];
    case 1: { uint16_t v; memcpy(&v, p, 2); return uns ? (int64_t)v : (int64_t)(int16_t)v; }
    case 2: { uint32_t v; memcpy(&v, p, 4); return uns ? (int64_t)v : (int64_t)(int32_t)v; }
    default: { int64_t v; memcpy(&v, p, 8); return v; }
    }
}

void bt_store(bt_cpu *c, uint64_t addr, int lw, int64_t v) { memcpy(c->mem + addr, &v, (size_t)1 << lw); }

static int fits(int64_t v, int lw, int uns)
{
    if (lw == 3)
        return 1;
    int b = 8 << lw;
    if (uns)
        return v >= 0 && v < (1LL << b);
    return v >= -(1LL << (b - 1)) && v < (1LL << (b - 1));
}

/* ------------------------------------------------------------------ dekodovani */
typedef struct {
    uint64_t base;
    int lw, uns, gath, mode;
    int64_t s[4];
} opnd;

typedef struct {
    int op, fn, flags;
    int64_t n[4];
    opnd o[4];          /* D A B C */
    uint64_t gbase;
    int glw, guns;
    int64_t gs[4], gscale;
    int rs1, rs2, rd, rs3;
    int64_t imm, imm2, imm3;
} instr;

static void decode(const uint32_t *w, instr *I)
{
    I->op = w[0] & 0xFF, I->fn = (w[0] >> 8) & 0xFF, I->flags = w[0] >> 16;
    for (int k = 0; k < 4; k++)
        I->n[k] = w[1 + k];
    for (int x = 0; x < 4; x++) {
        const uint32_t *q = w + BT_W_OPN + 6 * x;
        opnd *o = &I->o[x];
        o->base = q[0];
        o->lw = q[1] & 3, o->uns = (q[1] >> 2) & 1, o->gath = (q[1] >> 3) & 1, o->mode = (q[1] >> 4) & 7;
        for (int k = 0; k < 4; k++)
            o->s[k] = (int32_t)q[2 + k];
    }
    I->gbase = w[BT_W_GBASE], I->glw = w[BT_W_GCFG] & 3, I->guns = (w[BT_W_GCFG] >> 2) & 1;
    for (int k = 0; k < 4; k++)
        I->gs[k] = (int32_t)w[BT_W_GSTR + k];
    I->gscale = (int32_t)w[BT_W_GSCALE];
    I->rs1 = w[BT_W_REGS] & 0x3F, I->rs2 = (w[BT_W_REGS] >> 8) & 0x3F;
    I->rd = (w[BT_W_REGS] >> 16) & 0x3F, I->rs3 = (w[BT_W_REGS] >> 24) & 0x3F;
    I->imm = (int32_t)w[BT_W_IMM], I->imm2 = (int32_t)w[BT_W_IMM2], I->imm3 = (int32_t)w[BT_W_IMM3];
}

/* adresa prvku operandu pro indexy i[4] (gather podle instrukce) */
static inline int64_t eaddr(const bt_cpu *c, const instr *I, const opnd *o, const int64_t *i)
{
    int64_t e = o->s[0] * i[0] + o->s[1] * i[1] + o->s[2] * i[2] + o->s[3] * i[3];
    if (o->gath) {
        int64_t ga = (int64_t)I->gbase + ((I->gs[0] * i[0] + I->gs[1] * i[1] + I->gs[2] * i[2] + I->gs[3] * i[3]) << I->glw);
        e += bt_load(c, (uint64_t)ga, I->glw, I->guns) * I->gscale;
    }
    return (int64_t)o->base + (e << o->lw);
}

static inline int inmem(const bt_cpu *c, int64_t a, int lw) { return a >= 0 && (uint64_t)a + ((uint64_t)1 << lw) <= c->size; }

/* ------------------------------------------------------------------ EW */
static inline int64_t tcval(int tc, int64_t C)
{
    switch (tc) {
    case BT_TC_ADD: return C;
    case BT_TC_SUB: return -C;
    case BT_TC_ADDHALF: return C >> 1;
    case BT_TC_SUBHALF: return -(C >> 1);
    default: return 0;
    }
}

static inline int64_t ew_fn(const bt_cpu *c, const instr *I, int64_t A, int64_t B, int64_t C, const int64_t *i, uint32_t rnd)
{
    int64_t R1 = c->R[I->rs1], R2 = c->R[I->rs2];
    int64_t SH = I->imm + R1 + tcval(I->flags & BT_F_TC_MASK, C);
    switch (I->fn) {
    case BT_E_COPY: return A;
    case BT_E_ADD: return A + B;
    case BT_E_SUB: return A - B;
    case BT_E_MUL: return (int64_t)((uint64_t)A * (uint64_t)B);
    case BT_E_RSR: return rsr(A, SH);
    case BT_E_ADDRSR: return A + rsr(B, SH);
    case BT_E_MULRSR: return rsr(A * B, SH);
    case BT_E_MULRELURSR: return rsr(A * (B > 0 ? B : 0), SH);
    case BT_E_MULSUB: return A * B - C;
    case BT_E_SRD: return srd(A, SH, rnd);
    case BT_E_POW2Q: return pow2q1(A, (int)SH, rnd);
    case BT_E_BITLENI: return bitlen(A) + I->imm;
    case BT_E_SEXPR: return SH;
    case BT_E_NRMS: { int s = bitlen(A) - 16; return s + (s & 1); }
    case BT_E_NRMR: {
        int64_t mm = B >= 0 ? A >> (B > 63 ? 63 : B) : shl(A, -B);
        int64_t ix = clampi(mm >> 8, 0, 255);
        return rsr(I->imm * BT_RSQ[ix], 16);
    }
    case BT_E_EXP8: return BT_EXP8[clampi(B - A, 0, 255)];
    case BT_E_RCP: return A > 0 ? floordiv((1LL << I->imm) + (A >> 1), A) : 0;
    case BT_E_EXP16: { int64_t dd = (B - A + 4) >> 3; return dd >= 0 && dd < 512 ? BT_EXP16[dd] : 0; }
    case BT_E_OHSUB: return A - (i[0] == B ? (1LL << I->imm) : 0);
    case BT_E_ONEHOT: return i[0] == A;
    case BT_E_RELU2: { int64_t r = A > 0 ? A : 0; return r * r; }
    case BT_E_MEANDIV: return floordiv(2 * A + I->imm, 2 * I->imm);
    case BT_E_RSR2ADD: return rsr(A, R1) + rsr(B, R2);
    case BT_E_LIONQ: return A - sgn(7 * B + C) * (1LL << R1);
    case BT_E_MULADDI: return I->imm * A + B;
    case BT_E_RNG: return rnd;
    case BT_E_MODU: return (int64_t)((uint64_t)A % (uint64_t)(uint32_t)I->imm);
    }
    return 0;
}

static int exec_ew(bt_cpu *c, const instr *I)
{
    uint32_t key = 0;
    int need_a = (BT_EFN_NEED_A >> I->fn) & 1, need_b = (BT_EFN_NEED_B >> I->fn) & 1;
    int need_c = ((BT_EFN_NEED_C >> I->fn) & 1) || (I->flags & BT_F_TC_MASK);
    if ((BT_EFN_RNGMASK >> I->fn) & 1) {
        key = fmix32(c->csr[BT_CSR_SEED] ^ fmix32(c->csr[BT_CSR_CTR]));
        c->csr[BT_CSR_CTR]++;
    }
    int64_t lo = -(1LL << 62), hi = 1LL << 62;
    if (I->flags & (BT_F_CLAMP | BT_F_CLAMPSYM)) {
        hi = (1LL << (I->imm3 - 1)) - 1;
        lo = (I->flags & BT_F_CLAMPSYM) ? -hi : -hi - 1;
    }
    int clamp = (I->flags & (BT_F_CLAMP | BT_F_CLAMPSYM)) != 0;
    const opnd *D = &I->o[0], *A = &I->o[1], *B = &I->o[2], *C = &I->o[3];
    int64_t i[4];
    uint32_t f = 0;
    for (i[3] = 0; i[3] < I->n[3]; i[3]++)
        for (i[2] = 0; i[2] < I->n[2]; i[2]++)
            for (i[1] = 0; i[1] < I->n[1]; i[1]++)
                for (i[0] = 0; i[0] < I->n[0]; i[0]++, f++) {
                    int64_t a = 0, b = 0, cc = 0, v;
                    if (need_a) {
                        int64_t ad = eaddr(c, I, A, i);
                        if (!inmem(c, ad, A->lw)) return BT_ERR_MEM;
                        a = bt_load(c, (uint64_t)ad, A->lw, A->uns);
                    }
                    if (need_b) {
                        int64_t ad = eaddr(c, I, B, i);
                        if (!inmem(c, ad, B->lw)) return BT_ERR_MEM;
                        b = bt_load(c, (uint64_t)ad, B->lw, B->uns);
                    }
                    if (need_c) {
                        int64_t ad = eaddr(c, I, C, i);
                        if (!inmem(c, ad, C->lw)) return BT_ERR_MEM;
                        cc = bt_load(c, (uint64_t)ad, C->lw, C->uns);
                    }
                    if ((I->flags & BT_F_MASK) && i[0] > i[1])
                        v = c->R[I->rs2];
                    else {
                        uint32_t rnd = key ? fmix32((uint32_t)(f * 0x9E3779B1u) + key) : 0;
                        if ((BT_EFN_RNGMASK >> I->fn) & 1)
                            rnd = fmix32((uint32_t)(f * 0x9E3779B1u) + key);
                        v = ew_fn(c, I, a, b, cc, i, rnd);
                        if (clamp)
                            v = clampi(v, lo, hi);
                    }
                    int64_t ad = eaddr(c, I, D, i);
                    if (!inmem(c, ad, D->lw)) return BT_ERR_MEM;
                    if (!fits(v, D->lw, D->uns)) {
                        fprintf(stderr, "bt_isa: pc %u EW fn %d prvek %u hodnota %lld se nevejde (lw %d)\n",
                                c->pc, I->fn, f, (long long)v, D->lw);
                        return BT_ERR_FIT;
                    }
                    bt_store(c, (uint64_t)ad, D->lw, v);
                }
    c->nelem += f;
    return BT_OK;
}

/* ------------------------------------------------------------------ RED */
static int exec_red(bt_cpu *c, const instr *I)
{
    int pre = (I->flags >> BT_F_PRE_SHIFT) & 7, tc = I->flags & BT_F_TC_MASK;
    const opnd *D = &I->o[0], *A = &I->o[1], *B = &I->o[2], *C = &I->o[3];
    int need_a = pre != BT_RP_SH, need_b = pre == BT_RP_AB || pre == BT_RP_ABC;
    int need_c = pre == BT_RP_ABC || (pre == BT_RP_SH && tc);
    int64_t i[4];
    for (i[3] = 0; i[3] < I->n[3]; i[3]++)
        for (i[2] = 0; i[2] < I->n[2]; i[2]++)
            for (i[1] = 0; i[1] < I->n[1]; i[1]++) {
                int64_t acc = I->fn == BT_R_MAX ? INT64_MIN : 0;
                for (i[0] = 0; i[0] < I->n[0]; i[0]++) {
                    int64_t a = 0, b = 0, cc = 0, v;
                    if (need_a) {
                        int64_t ad = eaddr(c, I, A, i);
                        if (!inmem(c, ad, A->lw)) return BT_ERR_MEM;
                        a = bt_load(c, (uint64_t)ad, A->lw, A->uns);
                    }
                    if (need_b) {
                        int64_t ad = eaddr(c, I, B, i);
                        if (!inmem(c, ad, B->lw)) return BT_ERR_MEM;
                        b = bt_load(c, (uint64_t)ad, B->lw, B->uns);
                    }
                    if (need_c) {
                        int64_t ad = eaddr(c, I, C, i);
                        if (!inmem(c, ad, C->lw)) return BT_ERR_MEM;
                        cc = bt_load(c, (uint64_t)ad, C->lw, C->uns);
                    }
                    switch (pre) {
                    case BT_RP_AA: v = a * a; break;
                    case BT_RP_AB: v = a * b; break;
                    case BT_RP_ABC: v = a * b * cc; break;
                    case BT_RP_SH: v = I->imm + c->R[I->rs1] + tcval(tc, cc); break;
                    default: v = a;
                    }
                    if (I->fn == BT_R_SUM)
                        acc = (int64_t)((uint64_t)acc + (uint64_t)v);
                    else if (I->fn == BT_R_MAX)
                        acc = v > acc ? v : acc;
                    else
                        acc = iabs(v) > acc ? iabs(v) : acc;
                }
                c->nelem += I->n[0];
                if (I->flags & BT_F_TOREG) {
                    if (I->rd)
                        c->R[I->rd] = acc;
                } else {
                    int64_t z[4] = {0, i[1], i[2], i[3]};
                    int64_t ad = eaddr(c, I, D, z);
                    if (!inmem(c, ad, D->lw)) return BT_ERR_MEM;
                    if (!fits(acc, D->lw, D->uns)) {
                        fprintf(stderr, "bt_isa: pc %u RED hodnota %lld se nevejde (lw %d)\n", c->pc, (long long)acc, D->lw);
                        return BT_ERR_FIT;
                    }
                    bt_store(c, (uint64_t)ad, D->lw, acc);
                }
            }
    return BT_OK;
}

/* ------------------------------------------------------------------ MM */
/* D[i2][i1][i0] = sum_k A(i2,i1,k) * B(i2,i0,k); k < n3 */
static int exec_mm(bt_cpu *c, const instr *I)
{
    const opnd *D = &I->o[0], *A = &I->o[1], *B = &I->o[2];
    int64_t n0 = I->n[0], n1 = I->n[1], n2 = I->n[2], K = I->n[3];
    /* kontrola mezi: krajni adresy */
    for (int x = 0; x < 3; x++) {
        const opnd *o = &I->o[x];
        int64_t lo = 0, hi = 0;
        int64_t ext[4] = {n0 - 1, n1 - 1, n2 - 1, K - 1};
        for (int k = 0; k < 4; k++) {
            int64_t s = o->s[k] * ext[k];
            if (x == 0 && k == 3) s = 0;
            if (x == 1 && k == 0) s = 0;
            if (x == 2 && k == 1) s = 0;
            if (s < 0) lo += s; else hi += s;
        }
        if ((int64_t)o->base + (lo << o->lw) < 0 || !inmem(c, (int64_t)o->base + (hi << o->lw), o->lw))
            return BT_ERR_MEM;
    }
    int err = 0;
#pragma omp parallel for collapse(2) schedule(static) num_threads(c->threads > 0 ? c->threads : 1) if (c->threads != 1)
    for (int64_t i2 = 0; i2 < n2; i2++)
        for (int64_t i1 = 0; i1 < n1; i1++) {
            int64_t abase = (int64_t)A->base + ((A->s[2] * i2 + A->s[1] * i1) << A->lw);
            for (int64_t i0 = 0; i0 < n0; i0++) {
                int64_t bbase = (int64_t)B->base + ((B->s[2] * i2 + B->s[0] * i0) << B->lw);
                int64_t acc = 0;
                for (int64_t k = 0; k < K; k++) {
                    int64_t a = bt_load(c, (uint64_t)(abase + ((A->s[3] * k) << A->lw)), A->lw, A->uns);
                    int64_t b = bt_load(c, (uint64_t)(bbase + ((B->s[3] * k) << B->lw)), B->lw, B->uns);
                    if (B->mode == BT_M_SIGN)
                        b = b >= 0 ? 1 : -1;
                    acc += a * b;
                }
                if (acc >= (1LL << 47) || acc < -(1LL << 47))
                    err = 1;
                int64_t ad = (int64_t)D->base + ((D->s[2] * i2 + D->s[1] * i1 + D->s[0] * i0) << D->lw);
                if (!fits(acc, D->lw, D->uns))
                    err = 2;
                bt_store(c, (uint64_t)ad, D->lw, acc);
            }
        }
    c->nmac += (uint64_t)(n0 * n1 * n2 * K);
    if (err)
        fprintf(stderr, "bt_isa: pc %u MM %s\n", c->pc, err == 1 ? "preteceni 48 bitu" : "vysledek se nevejde");
    return err == 1 ? BT_ERR_MMOVF : err == 2 ? BT_ERR_FIT : BT_OK;
}

/* ------------------------------------------------------------------ skalarni a rizeni */
int bt_step(bt_cpu *c, uint32_t prog)
{
    uint64_t ia = (uint64_t)prog + (uint64_t)c->pc * BT_IWORDS * 4;
    if (ia + BT_IWORDS * 4 > c->size)
        return -BT_ERR_MEM;
    uint32_t w[BT_IWORDS];
    memcpy(w, c->mem + ia, sizeof w);
    instr I;
    decode(w, &I);
    c->ninstr++;
    c->csr[BT_CSR_CYCLES]++;
    uint32_t npc = c->pc + 1;
    int r = BT_OK;
    int64_t *R = c->R, R1 = R[I.rs1], R2 = R[I.rs2], R3 = R[I.rs3], v = 0;
    int wr = 0;
    switch (I.op) {
    case BT_OP_NOP: break;
    case BT_OP_HALT: return 1;
    case BT_OP_SCALAR:
        wr = 1;
        switch (I.fn) {
        case BT_S_LI: v = (int64_t)((uint64_t)I.imm + ((uint64_t)I.imm2 << 32)); break;
        case BT_S_ADD: v = R1 + R2 + I.imm; break;
        case BT_S_SUB: v = R1 - R2 + I.imm; break;
        case BT_S_MAX: v = R1 > R2 ? R1 : R2; break;
        case BT_S_MIN: v = R1 < R2 ? R1 : R2; break;
        case BT_S_BITLEN: v = bitlen(R1) + I.imm; break;
        case BT_S_SEL: v = R3 ? R1 : R2; break;
        case BT_S_SHRA: v = R1 >> I.imm; break;
        case BT_S_NE: v = R1 != R2; break;
        case BT_S_LD:
            if (!inmem(c, (int64_t)I.o[0].base, I.o[0].lw)) return -BT_ERR_MEM;
            v = bt_load(c, I.o[0].base, I.o[0].lw, I.o[0].uns);
            break;
        case BT_S_ST:
            wr = 0;
            if (!inmem(c, (int64_t)I.o[0].base, I.o[0].lw)) return -BT_ERR_MEM;
            if (!fits(R1, I.o[0].lw, I.o[0].uns)) return -BT_ERR_FIT;
            bt_store(c, I.o[0].base, I.o[0].lw, R1);
            break;
        case BT_S_CSRR: v = c->csr[I.imm & 3]; break;
        case BT_S_CSRW: wr = 0; c->csr[I.imm & 3] = (uint32_t)R1; break;
        default: return -BT_ERR_OP;
        }
        if (wr && I.rd)
            R[I.rd] = v;
        break;
    case BT_OP_BRANCH:
        if ((I.fn == BT_B_BEQZ && R1 == 0) || (I.fn == BT_B_BNEZ && R1 != 0) || I.fn == BT_B_JMP)
            npc = (uint32_t)I.imm;
        break;
    case BT_OP_EW: r = exec_ew(c, &I); break;
    case BT_OP_RED: r = exec_red(c, &I); break;
    case BT_OP_MM: r = exec_mm(c, &I); break;
    default: return -BT_ERR_OP;
    }
    if (r)
        return -r;
    c->pc = npc;
    return 0;
}

int bt_run(bt_cpu *c, uint32_t prog, uint64_t max_instr)
{
    c->pc = 0;
    for (uint64_t k = 0; k < max_instr; k++) {
        int r = bt_step(c, prog);
        if (r == 1)
            return BT_OK;
        if (r < 0) {
            c->csr[BT_CSR_ERR] = (uint32_t)-r;
            c->err_pc = c->pc;
            return -r;
        }
    }
    c->csr[BT_CSR_ERR] = BT_ERR_LIMIT;
    return BT_ERR_LIMIT;
}
