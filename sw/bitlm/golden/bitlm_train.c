#define _POSIX_C_SOURCE 199309L
/*
 * BitLM-nano - trenink referencnim modelem na CPU (stejna aritmetika jako HW).
 *
 * Pouziti:
 *   bitlm_train -c data/rur.tok [-n kroku] [-s seed] [-e interval_eval]
 *               [-l log.csv] [-w vahy.bin] [-g delka_generovani]
 *               [-p s1,s2,g1,g2,lr1,lr2,lre,clip,lrb,flags]
 *               [-d K] [-D max]   plan uceni: kazdych K kroku lr* posun +1 (max. D-krat)
 *               [-S k1,k2,...] [-W prefix]  snimky vah po krocich k1,k2,... do prefix<k>.bin
 *               [-v delka]  validace = poslednich <delka> tokenu (jinak 10 %)
 *
 * Korpus se deli: prvnich 90 % trenink, poslednich 10 % validace.
 * Validacni metriky: krizova entropie (bit/znak) a presnost top-1
 * pres vsechny pozice validacni casti. Pro srovnani se tisknou n-gram
 * baseline (uni/bi/trigram s Laplaceovym vyhlazenim, nauceny na treninkove casti).
 */
#include "bitlm_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint8_t *load_corpus(const char *path, uint32_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint8_t *buf = malloc(BL_CORPUS_MAX);
    size_t n = fread(buf, 1, BL_CORPUS_MAX, f);
    if (!feof(f) && fgetc(f) != EOF) {
        fprintf(stderr, "korpus je delsi nez %d tokenu, orezavam\n", BL_CORPUS_MAX);
    }
    fclose(f);
    for (size_t i = 0; i < n; i++) buf[i] &= (BL_V - 1);
    *len = (uint32_t)n;
    return buf;
}

typedef struct { double ce_bits; double acc; } eval_t;

static eval_t evaluate(const bl_model *m, const uint8_t *c, uint32_t from, uint32_t to)
{
    static bl_fwd f;
    double ce = 0;
    uint32_t ok = 0, n = 0;
    for (uint32_t pos = from; pos < to; pos++) {
        bl_forward(m, c + pos - BL_N, &f);
        uint32_t p = f.p[c[pos]];
        ce += -log2((p ? p : 1) / 65536.0);
        ok += (f.argmax == c[pos]);
        n++;
    }
    eval_t r = { ce / n, (double)ok / n };
    return r;
}

/* Diagnostika rozsahu aktivaci (pro ladeni skal). */
static void diagnostics(const bl_model *m, const uint8_t *c, uint32_t from, uint32_t to)
{
    static bl_fwd f;
    double dead = 0, sat = 0, s1 = 0, zr = 0;
    uint32_t n = 0, esat = 0;
    for (uint32_t pos = from; pos < to; pos += 7) {
        bl_forward(m, c + pos - BL_N, &f);
        int32_t zmin = f.z[0], zmax = f.z[0];
        for (int i = 1; i < BL_V; i++) {
            if (f.z[i] < zmin) zmin = f.z[i];
            if (f.z[i] > zmax) zmax = f.z[i];
        }
        for (int j = 0; j < BL_H; j++) {
            dead += f.a[j] == 0;
            sat += f.a[j] == 127;
        }
        s1 += f.s1dyn;
        zr += zmax - zmin;
        n++;
    }
    for (int i = 0; i < BL_V; i++)
        for (int e = 0; e < BL_E; e++) esat += (m->emb[i][e] >= 127 * 256 || m->emb[i][e] <= -127 * 256);
    printf("diag: a==0 %.1f %%, a==127 %.2f %%, prum. s1dyn %.2f, prum. rozpeti z %.1f, emb saturace %u/%d\n",
           100 * dead / (n * BL_H), 100 * sat / (n * BL_H), s1 / n, zr / n, esat, BL_V * BL_E);
}

static void ngram_baselines(const uint8_t *c, uint32_t split, uint32_t len)
{
    static uint32_t uni[BL_V], bi[BL_V][BL_V], tri[BL_V][BL_V][BL_V];
    for (uint32_t i = 2; i < split; i++) {
        uni[c[i]]++;
        bi[c[i - 1]][c[i]]++;
        tri[c[i - 2]][c[i - 1]][c[i]]++;
    }
    double e1 = 0, e2 = 0, e3 = 0;
    uint32_t n = 0;
    for (uint32_t i = split; i < len; i++) {
        uint32_t s1 = 0, s2 = 0, s3 = 0;
        for (int t = 0; t < BL_V; t++) {
            s1 += uni[t];
            s2 += bi[c[i - 1]][t];
            s3 += tri[c[i - 2]][c[i - 1]][t];
        }
        e1 += -log2((uni[c[i]] + 1.0) / (s1 + BL_V));
        e2 += -log2((bi[c[i - 1]][c[i]] + 1.0) / (s2 + BL_V));
        e3 += -log2((tri[c[i - 2]][c[i - 1]][c[i]] + 1.0) / (s3 + BL_V));
        n++;
    }
    printf("baseline val CE [bit/znak]: rovnomerne %.3f | unigram %.3f | bigram %.3f | trigram %.3f\n",
           log2(BL_V), e1 / n, e2 / n, e3 / n);
}

static void save_weights(const bl_model *m, const char *path)
{
    FILE *wf = fopen(path, "wb");
    if (!wf) { perror(path); exit(1); }
    fwrite(m->w1, sizeof(m->w1), 1, wf);
    fwrite(m->w2, sizeof(m->w2), 1, wf);
    fwrite(m->emb, sizeof(m->emb), 1, wf);
    fwrite(m->b2, sizeof(m->b2), 1, wf);
    fclose(wf);
}

static int is_snapshot(const char *list, uint32_t step)
{
    if (!list) return 0;
    const char *p = list;
    while (*p) {
        char *end;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (v == step) return 1;
        p = (*end == ',') ? end + 1 : end;
    }
    return 0;
}

static void print_tokens(const uint8_t *t, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) putchar(BL_VOCAB[t[i] & (BL_V - 1)]);
}

int main(int argc, char **argv)
{
    const char *corpus_path = NULL, *log_path = NULL, *w_path = NULL;
    const char *snap_list = NULL, *snap_prefix = "snap_";
    uint32_t steps = 1000000, seed = 12345, eval_every = 50000, gen_len = 400;
    uint32_t decay_every = 0, decay_max = 3; /* plan uceni: kazdych K kroku lr posun +1 */
    uint32_t val_len = 0;                    /* 0 = poslednich 10 % korpusu */
    bl_hparams hp = bl_default_hparams();

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc) corpus_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) steps = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) eval_every = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) w_path = argv[++i];
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) gen_len = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) decay_every = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-D") && i + 1 < argc) decay_max = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) snap_list = argv[++i];
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) val_len = (uint32_t)strtoul(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "-W") && i + 1 < argc) snap_prefix = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            unsigned v[10];
            if (sscanf(argv[++i], "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3],
                       &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) != 10) {
                fprintf(stderr, "-p ocekava 10 cisel: s1,s2,g1,g2,lr1,lr2,lre,clip,lrb,flags\n");
                return 2;
            }
            hp = (bl_hparams){ v[0], v[1], v[2], v[3], v[4], v[5], v[6], (uint16_t)v[7],
                               v[8], v[9] };
        } else {
            fprintf(stderr, "neznamy argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!corpus_path) {
        fprintf(stderr, "chybi -c korpus\n");
        return 2;
    }
    if (eval_every == 0) eval_every = steps;

    uint32_t len;
    uint8_t *c = load_corpus(corpus_path, &len);
    uint32_t split = val_len ? len - val_len : len * 9 / 10;
    printf("korpus %u tokenu, trenink %u, validace %u\n", len, split, len - split);
    printf("hparams s1=%u s2=%u g1=%u g2=%u lr1=%u lr2=%u lre=%u clip=%u lrb=%u flags=%u seed=%u\n",
           hp.s1, hp.s2, hp.g1, hp.g2, hp.lr1, hp.lr2, hp.lre, hp.clip, hp.lrb, hp.flags, seed);
    ngram_baselines(c, split, len);

    static bl_model m;
    bl_init(&m, seed, hp);

    FILE *log = log_path ? fopen(log_path, "w") : NULL;
    if (log) fprintf(log, "step,train_acc,train_mean_p,val_ce_bits,val_acc\n");

    eval_t ev = evaluate(&m, c, split, len);
    printf("%9s  %9s %9s  %11s %8s\n", "krok", "tr_acc", "tr_p", "val_CE", "val_acc");
    printf("%9u  %9s %9s  %11.4f %8.4f\n", 0u, "-", "-", ev.ce_bits, ev.acc);
    if (log) fprintf(log, "0,,,%.5f,%.5f\n", ev.ce_bits, ev.acc);

    char snap_path[512];
    if (is_snapshot(snap_list, 0)) {
        snprintf(snap_path, sizeof(snap_path), "%s%u.bin", snap_prefix, 0u);
        save_weights(&m, snap_path);
    }
    double t0 = wall();
    for (uint32_t s = 1; s <= steps; s++) {
        bl_train_random(&m, c, split);
        if (is_snapshot(snap_list, s)) {
            snprintf(snap_path, sizeof(snap_path), "%s%u.bin", snap_prefix, s);
            save_weights(&m, snap_path);
        }
        /* plan uceni: host v HW jen prepise registry lr* mezi davkami */
        if (decay_every && s % decay_every == 0 && s / decay_every <= decay_max) {
            m.hp.lr1++; m.hp.lr2++; m.hp.lre++; m.hp.lrb++;
        }
        if (s % eval_every == 0 || s == steps) {
            double tr_acc = (double)m.correct / m.steps;
            double tr_p = (double)m.sum_pt / m.steps / 65536.0;
            ev = evaluate(&m, c, split, len);
            printf("%9u  %9.4f %9.4f  %11.4f %8.4f\n", s, tr_acc, tr_p, ev.ce_bits, ev.acc);
            if (log) {
                fprintf(log, "%u,%.5f,%.5f,%.5f,%.5f\n", s, tr_acc, tr_p, ev.ce_bits, ev.acc);
                fflush(log);
            }
            m.steps = m.correct = 0;
            m.sum_pt = 0;
        }
    }
    double secs = wall() - t0; /* vcetne prubeznych evaluaci */
    printf("trenink CPU (referencni model, vcetne evaluaci): %.1f s, %.0f kroku/s\n", secs, steps / secs);
    if (log) fclose(log);

    diagnostics(&m, c, split, len);

    /* 1-bit vahy: podil kladnych znamenek */
    uint32_t pos1 = 0, pos2 = 0;
    for (int j = 0; j < BL_H; j++)
        for (int k = 0; k < BL_D; k++)
            pos1 += m.w1[j][k] >= ((hp.flags & BL_F_CENTER) ? (m.rs1[j] >> BL_LOG2_D) : 0);
    for (int i = 0; i < BL_V; i++)
        for (int j = 0; j < BL_H; j++)
            pos2 += m.w2[i][j] >= ((hp.flags & BL_F_CENTER) ? (m.rs2[i] >> BL_LOG2_H) : 0);
    printf("1-bit vahy: W1 +1 %.1f %%, W2 +1 %.1f %%\n", 100.0 * pos1 / (BL_H * BL_D),
           100.0 * pos2 / (BL_V * BL_H));

    if (gen_len) {
        uint8_t ctx[BL_N];
        memcpy(ctx, c + split, BL_N);
        uint8_t *out = malloc(gen_len);
        bl_generate(&m, ctx, gen_len, out);
        printf("--- prompt (z validace): \"");
        print_tokens(c + split, BL_N);
        printf("\"\n--- generovano:\n");
        print_tokens(out, gen_len);
        printf("\n---\n");
        free(out);
    }

    if (w_path) {
        save_weights(&m, w_path);
        printf("vahy ulozeny do %s\n", w_path);
    }
    free(c);
    return 0;
}
