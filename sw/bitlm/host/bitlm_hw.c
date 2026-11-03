/* BitLM - ovladac HW jadra a bitove presne overeni (viz bitlm_hw.h). */
#include "bitlm_hw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bitlm_probe(const bitlm_dev *d)
{
    uint32_t id = d->rd(d->ctx, BLR_ID);
    uint32_t p = d->rd(d->ctx, BLR_PARAMS);
    uint32_t n = p & 15, e = (p >> 4) & 15, v = (p >> 8) & 15, ms = (p >> 12) & 15, h = (p >> 16) & 15;
#ifdef BL_MULTISCALE
    const uint32_t ms_ref = 1;
#else
    const uint32_t ms_ref = 0;
#endif
    if (id != BL_ID_VALUE) {
        fprintf(stderr, "bitlm: spatne ID 0x%08x (ocekavano 0x%08x)\n", id, BL_ID_VALUE);
        return -1;
    }
    if (n != BL_LOG2_N || e != BL_LOG2_E || v != 5 || h != BL_LOG2_H || ms != ms_ref) {
        fprintf(stderr, "bitlm: HW rozmery N=2^%u E=2^%u V=2^%u H=2^%u MS=%u nesedi s referenci "
                        "(2^%d, 2^%d, 2^5, 2^%d, MS=%u)\n", n, e, v, h, ms, BL_LOG2_N, BL_LOG2_E,
                        BL_LOG2_H, ms_ref);
        return -2;
    }
    return 0;
}

void bitlm_set_hparams(const bitlm_dev *d, const bl_hparams *hp)
{
    d->wr(d->ctx, BLR_HP0, (uint32_t)(hp->g2 & 31) << 24 | (uint32_t)(hp->g1 & 31) << 16 |
                           (uint32_t)(hp->s2 & 31) << 8 | (hp->s1 & 31));
    d->wr(d->ctx, BLR_HP1, (uint32_t)(hp->lrb & 31) << 24 | (uint32_t)(hp->lre & 31) << 16 |
                           (uint32_t)(hp->lr2 & 31) << 8 | (hp->lr1 & 31));
    d->wr(d->ctx, BLR_HP2, (uint32_t)hp->flags << 16 | hp->clip);
}

void bitlm_load_corpus(const bitlm_dev *d, const uint8_t *tok, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += 4) {
        uint32_t w = 0;
        for (uint32_t b = 0; b < 4 && i + b < len; b++)
            w |= (uint32_t)(tok[i + b] & 31) << (8 * b);
        d->wr(d->ctx, BLM_CORPUS + i, w);
    }
}

int bitlm_cmd(const bitlm_dev *d, uint32_t cmd, uint64_t max_polls)
{
    d->wr(d->ctx, BLR_CTRL, cmd);
    for (uint64_t i = 0; i < max_polls; i++) {
        uint32_t st = d->rd(d->ctx, BLR_STATUS);
        if (!(st & BL_ST_BUSY) && (st & BL_ST_DONE))
            return (int)(st & 7);
        if (d->idle) d->idle(d->ctx);
    }
    return -1;
}

int bitlm_init(const bitlm_dev *d, uint32_t seed)
{
    d->wr(d->ctx, BLR_SEED, seed);
    return bitlm_cmd(d, BL_CMD_INIT, 100000000ull);
}

int bitlm_train(const bitlm_dev *d, uint32_t len, uint32_t nsteps)
{
    d->wr(d->ctx, BLR_CORPUS_LEN, len);
    d->wr(d->ctx, BLR_NSTEPS, nsteps);
    return bitlm_cmd(d, BL_CMD_TRAIN, ~0ull);
}

int bitlm_generate(const bitlm_dev *d, const uint8_t ctx[BL_N], uint32_t count, uint8_t *out)
{
    if (count > BL_GEN_MAX) count = BL_GEN_MAX;
    for (int w = 0; w < BL_N / 4; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++) v |= (uint32_t)(ctx[w * 4 + b] & 31) << (8 * b);
        d->wr(d->ctx, BLR_CTX + 4 * w, v);
    }
    d->wr(d->ctx, BLR_GEN_LEN, count);
    int st = bitlm_cmd(d, BL_CMD_GEN, ~0ull);
    for (uint32_t i = 0; i < count; i += 4) {
        uint32_t v = d->rd(d->ctx, BLM_GEN + i);
        for (uint32_t b = 0; b < 4 && i + b < count; b++) out[i + b] = (v >> (8 * b)) & 31;
    }
    return st;
}

void bitlm_read_stats(const bitlm_dev *d, bitlm_stats *s)
{
    s->steps = d->rd(d->ctx, BLR_STEPS_DONE);
    s->correct = d->rd(d->ctx, BLR_CORRECT);
    s->sum_pt = d->rd(d->ctx, BLR_SUM_PT_LO) | (uint64_t)d->rd(d->ctx, BLR_SUM_PT_HI) << 32;
    s->cycles = d->rd(d->ctx, BLR_CYCLES_LO) | (uint64_t)d->rd(d->ctx, BLR_CYCLES_HI) << 32;
    s->rng = d->rd(d->ctx, BLR_RNG);
}

void bitlm_read_model(const bitlm_dev *d, bl_model *m)
{
    for (int j = 0; j < BL_H; j++)
        for (int k = 0; k < BL_D; k++)
            m->w1[j][k] = (int16_t)d->rd(d->ctx, BLM_W1 + 4u * (uint32_t)(j * BL_D + k));
    for (int i = 0; i < BL_V; i++)
        for (int j = 0; j < BL_H; j++)
            m->w2[i][j] = (int16_t)d->rd(d->ctx, BLM_W2 + 4u * (uint32_t)(i * BL_H + j));
    for (int c = 0; c < BL_V; c++)
        for (int e = 0; e < BL_E; e++)
            m->emb[c][e] = (int16_t)d->rd(d->ctx, BLM_EMB + 4u * (uint32_t)(c * BL_E + e));
    m->rng = d->rd(d->ctx, BLR_RNG);
}

void bitlm_write_model(const bitlm_dev *d, const bl_model *m)
{
    for (int j = 0; j < BL_H; j++)
        for (int k = 0; k < BL_D; k++)
            d->wr(d->ctx, BLM_W1 + 4u * (uint32_t)(j * BL_D + k), (uint16_t)m->w1[j][k]);
    for (int i = 0; i < BL_V; i++)
        for (int j = 0; j < BL_H; j++)
            d->wr(d->ctx, BLM_W2 + 4u * (uint32_t)(i * BL_H + j), (uint16_t)m->w2[i][j]);
    for (int c = 0; c < BL_V; c++)
        for (int e = 0; e < BL_E; e++)
            d->wr(d->ctx, BLM_EMB + 4u * (uint32_t)(c * BL_E + e), (uint16_t)m->emb[c][e]);
    d->wr(d->ctx, BLR_RNG, m->rng);
}

/* ------------------------------------------------------------- overeni */

static int compare_models(const bl_model *ref, const bl_model *hw, const char *what, int verbose)
{
    int bad = 0, shown = 0;
#define CMP(cond, ...)                                                     \
    do {                                                                   \
        if (cond) {                                                        \
            bad++;                                                         \
            if (verbose && shown < 8) { fprintf(stderr, "  NESHODA %s: ", what); \
                fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); shown++; } \
        }                                                                  \
    } while (0)
    for (int j = 0; j < BL_H; j++)
        for (int k = 0; k < BL_D; k++)
            CMP(ref->w1[j][k] != hw->w1[j][k], "W1[%d][%d] ref=%d hw=%d", j, k, ref->w1[j][k], hw->w1[j][k]);
    for (int i = 0; i < BL_V; i++)
        for (int j = 0; j < BL_H; j++)
            CMP(ref->w2[i][j] != hw->w2[i][j], "W2[%d][%d] ref=%d hw=%d", i, j, ref->w2[i][j], hw->w2[i][j]);
    for (int c = 0; c < BL_V; c++)
        for (int e = 0; e < BL_E; e++)
            CMP(ref->emb[c][e] != hw->emb[c][e], "EMB[%d][%d] ref=%d hw=%d", c, e, ref->emb[c][e], hw->emb[c][e]);
    CMP(ref->rng != hw->rng, "RNG ref=0x%08x hw=0x%08x", ref->rng, hw->rng);
#undef CMP
    return bad;
}

static int check_stats(const bitlm_dev *d, const bl_model *ref, const char *what, int verbose)
{
    bitlm_stats s;
    bitlm_read_stats(d, &s);
    int bad = (s.steps != ref->steps) + (s.correct != ref->correct) + (s.sum_pt != ref->sum_pt);
    if (bad && verbose)
        fprintf(stderr, "  NESHODA %s statistiky: steps %u/%u correct %u/%u sum_pt %llu/%llu\n", what,
                s.steps, ref->steps, s.correct, ref->correct,
                (unsigned long long)s.sum_pt, (unsigned long long)ref->sum_pt);
    return bad;
}

int bitlm_verify(const bitlm_dev *d, const uint8_t *corpus, uint32_t len,
                 uint32_t steps_per_round, int verbose)
{
    int fails = 0, r;
    static bl_model ref, hw;
    uint32_t seed = 0xC0FFEE11u;

    if (bitlm_probe(d)) return 1;

    /* 1) INIT */
    bl_hparams hp = bl_default_hparams();
    bl_init(&ref, seed, hp);
    bitlm_set_hparams(d, &hp);
    r = bitlm_init(d, seed);
    if (r != BL_ST_DONE) { fprintf(stderr, "INIT status %d\n", r); return 1; }
    memset(&hw, 0, sizeof(hw));
    bitlm_read_model(d, &hw);
    r = compare_models(&ref, &hw, "INIT", verbose);
    printf("[verify] INIT (seed 0x%08x): %s\n", seed, r ? "NESHODA" : "OK");
    fails += r;

    /* 2) trenink v nekolika davkach s ruznymi hyperparametry (vcetne krajnich) */
    bitlm_load_corpus(d, corpus, len);
    if (bitlm_cmd(d, BL_CMD_CLEAR_STATS, 1000) != BL_ST_DONE) fails++;
    static const bl_hparams rounds[] = {
        /* s1 s2 g1 g2 lr1 lr2 lre clip lrb flags */
        { 0, 4, 4, 0, 14, 8, 8, 32767, 12, 0 },   /* vychozi (vyladene) */
        { 0, 4, 4, 0, 6, 3, 3, 32767, 12, 0 },    /* velke kroky -> saturace/clip */
        { 2, 0, 0, 2, 0, 0, 0, 300, 12, 0 },      /* posuny 0, maly clip (g2=2: rezerva int32 pro H=2048) */
        { 3, 9, 7, 3, 20, 16, 18, 1000, 12, 0 },  /* velke posuny, zaokrouhleni */
        { 0, 4, 4, 0, 15, 9, 9, 32767, 12, 0 },   /* plan uceni: +1 */
    };
    for (unsigned k = 0; k < sizeof(rounds) / sizeof(rounds[0]); k++) {
        ref.hp = rounds[k];
        bitlm_set_hparams(d, &rounds[k]);
        for (uint32_t s = 0; s < steps_per_round; s++) bl_train_random(&ref, corpus, len);
        r = bitlm_train(d, len, steps_per_round);
        if (r != BL_ST_DONE) { fprintf(stderr, "TRAIN status %d\n", r); fails++; continue; }
        bitlm_read_model(d, &hw);
        char what[32];
        snprintf(what, sizeof(what), "TRAIN#%u", k);
        r = compare_models(&ref, &hw, what, verbose) + check_stats(d, &ref, what, verbose);
        bitlm_stats st;
        bitlm_read_stats(d, &st);
        printf("[verify] %s: %u kroku, hparams %u,%u,%u,%u,%u,%u,%u,%u: %s (%.0f taktu/krok)\n",
               what, steps_per_round, rounds[k].s1, rounds[k].s2, rounds[k].g1, rounds[k].g2,
               rounds[k].lr1, rounds[k].lr2, rounds[k].lre, rounds[k].clip, r ? "NESHODA" : "OK",
               (double)st.cycles / steps_per_round);
        fails += r;
    }

    /* 3) kratky korpus (len = N + 1 -> jedina mozna pozice) */
    {
        ref.hp = rounds[0];
        bitlm_set_hparams(d, &rounds[0]);
        for (int s = 0; s < 3; s++) bl_train_random(&ref, corpus, BL_N + 1);
        r = bitlm_train(d, BL_N + 1, 3);
        bitlm_read_model(d, &hw);
        r = (r != BL_ST_DONE) + compare_models(&ref, &hw, "TRAIN len=N+1", verbose);
        printf("[verify] TRAIN s korpusem N+1 tokenu: %s\n", r ? "NESHODA" : "OK");
        fails += r;
    }

    /* 4) chybny vstup: len <= N musi skoncit chybou a nic nezmenit */
    {
        r = bitlm_train(d, BL_N, 5);
        bitlm_read_model(d, &hw);
        int bad = (r != (BL_ST_DONE | BL_ST_ERR)) + compare_models(&ref, &hw, "TRAIN len=N", verbose);
        printf("[verify] TRAIN s len=N -> chyba bez zmeny vah: %s (status %d)\n", bad ? "NESHODA" : "OK", r);
        fails += bad;
    }

    /* 5) generovani */
    {
        uint8_t ctx_ref[BL_N], ctx_hw[BL_N], out_ref[200], out_hw[200];
        memcpy(ctx_ref, corpus + 100, BL_N);
        memcpy(ctx_hw, corpus + 100, BL_N);
        bl_generate(&ref, ctx_ref, 200, out_ref);
        r = bitlm_generate(d, ctx_hw, 200, out_hw);
        int bad = (r != BL_ST_DONE) + (memcmp(out_ref, out_hw, 200) != 0);
        bad += (d->rd(d->ctx, BLR_RNG) != ref.rng);
        for (int w = 0; w < BL_N / 4; w++) {
            uint32_t v = d->rd(d->ctx, BLR_CTX + 4 * w);
            for (int b = 0; b < 4; b++) bad += ((v >> (8 * b)) & 31) != ctx_ref[w * 4 + b];
        }
        printf("[verify] GEN 200 tokenu: %s\n", bad ? "NESHODA" : "OK");
        if (verbose) {
            printf("  HW : ");
            for (int i = 0; i < 80; i++) putchar(BL_VOCAB[out_hw[i]]);
            printf("\n  REF: ");
            for (int i = 0; i < 80; i++) putchar(BL_VOCAB[out_ref[i]]);
            printf("\n");
        }
        fails += bad;
    }

    /* 6) zapis vah z hosta (checkpoint) a zpetne cteni */
    {
        bl_model ck = ref;
        ck.w1[0][0] = -32767; ck.w1[BL_H - 1][BL_D - 1] = 32767;
        ck.w2[3][5] = 1234; ck.emb[31][31] = -4321;
        bitlm_write_model(d, &ck);
        bitlm_read_model(d, &hw);
        r = compare_models(&ck, &hw, "WRITE", verbose);
        printf("[verify] zapis/cteni vah z hosta: %s\n", r ? "NESHODA" : "OK");
        fails += r;
    }

    printf("[verify] VYSLEDEK: %s (%d neshod)\n", fails ? "SELHALO" : "HW JE BITOVE SHODNY S REFERENCI", fails);
    return fails;
}
