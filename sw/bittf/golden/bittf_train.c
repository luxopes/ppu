/*
 * bittf_train - trenink BitTF golden modelem (bitove shodny s tools/bitlm/intsim.py).
 *
 * bittf_train korpus.tok [--d 128 --layers 5 --heads 4 --ff 648 --ctx 128 --batch 32
 *                        --steps 8000 --lr-shift 4 --seed 12345 --val-len 11152
 *                        --eval-every 1000 --hash-every 0 --trace 0]
 */
#include "bittf_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* CE (bity/znak) na kazde stride-te pozici validace; float jen pro mereni */
static void evaluate(const bt_model *m, const uint8_t *data, long len, long split, int stride, double *ce, double *acc)
{
    enum { EB = 64 };
    int T = m->T;
    bt_work *w = bt_work_new(m, EB);
    uint8_t *idx = (uint8_t *)malloc((size_t)EB * T);
    long pos[EB], cnt = 0, ok = 0;
    double s = 0;
    for (long p = split; p < len;) {
        int nb = 0;
        for (; nb < EB && p < len; p += stride, nb++) {
            pos[nb] = p;
            memcpy(idx + (long)nb * T, data + p - T, (size_t)T);
        }
        for (int b = nb; b < EB; b++)
            memcpy(idx + (long)b * T, idx, (size_t)T);
        int64_t *lg;
        int *el;
        bt_forward(m, w, idx, 0, &lg, &el);
        for (int b = 0; b < nb; b++) {
            long n = (long)b * T + T - 1;
            double v[BT_V], mx = -1e300, z = 0;
            int best = 0;
            for (int k = 0; k < BT_V; k++) {
                v[k] = ldexp((double)lg[n * BT_V + k], el[n]);
                if (v[k] > mx)
                    mx = v[k], best = k;
            }
            for (int k = 0; k < BT_V; k++)
                z += exp(v[k] - mx);
            int tgt = data[pos[b]];
            s += (mx + log(z) - v[tgt]) / log(2.0);
            ok += best == tgt;
            cnt++;
        }
    }
    *ce = s / cnt;
    *acc = (double)ok / cnt;
    free(idx);
    bt_work_free(w);
}

int main(int argc, char **argv)
{
    int d = 128, L = 5, H = 4, ff = 648, T = 128, B = 32, steps = 8000, lr = 4, eval_every = 1000;
    int hash_every = 0, trace = 0;
    long val_len = 11152;
    unsigned seed = 12345;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
#define OPT(s, v) else if (!strcmp(argv[i], s) && i + 1 < argc) v = atoi(argv[++i]);
        if (argv[i][0] != '-')
            path = argv[i];
        OPT("--d", d) OPT("--layers", L) OPT("--heads", H) OPT("--ff", ff) OPT("--ctx", T)
        OPT("--batch", B) OPT("--steps", steps) OPT("--lr-shift", lr) OPT("--eval-every", eval_every)
        OPT("--hash-every", hash_every) OPT("--trace", trace)
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--val-len") && i + 1 < argc) val_len = atol(argv[++i]);
        else {
            fprintf(stderr, "neznamy argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!path) {
        fprintf(stderr, "pouziti: bittf_train korpus.tok [volby]\n");
        return 2;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc((size_t)len);
    if (fread(data, 1, (size_t)len, f) != (size_t)len)
        return 1;
    fclose(f);
    for (long i = 0; i < len; i++)
        data[i] &= 31;
    long split = len - val_len;

    bt_rng rng;
    bt_rng_init(&rng, seed);
    bt_model m;
    bt_model_init(&m, d, L, H, ff, T, lr, &rng);
    long np = 0;
    for (int i = 0; i < m.np; i++)
        np += m.p[i].n;
    printf("bittf d=%d L=%d H=%d ff=%d: %ld parametru, korpus %ld\n", d, L, H, ff, np, len);
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = (uint8_t *)malloc((size_t)B * T), *y = (uint8_t *)malloc((size_t)B * T);
    uint32_t *r = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)B);
    double t0 = now();
    for (int step = 1; step <= steps; step++) {
        if (step == steps / 2 || step == steps * 3 / 4 || step == steps * 7 / 8)
            m.lr_shift--;
        bt_rng_draw(&rng, B, r);
        for (int b = 0; b < B; b++) {
            long ix = T + (long)(r[b] % (uint32_t)(split - 1 - T));
            memcpy(x + (long)b * T, data + ix - T, (size_t)T);
            memcpy(y + (long)b * T, data + ix - T + 1, (size_t)T);
        }
        int64_t *lg;
        int *el;
        bt_forward(&m, w, x, 1, &lg, &el);
        bt_backward_update(&m, w, y, &rng, step <= trace, step);
        if (step <= trace)
            printf("trace %d state %08x\n", step, bt_state_hash(&m, &rng));
        if (hash_every && step % hash_every == 0)
            printf("hash %d %08x\n", step, bt_state_hash(&m, &rng));
        if (eval_every && (step % eval_every == 0 || step == steps)) {
            double ce, acc;
            evaluate(&m, data, len, split, 4, &ce, &acc);
            printf("%6d  val CE(1/4) %.4f  acc %.4f  lr_shift %d  (%.1f s)\n", step, ce, acc, m.lr_shift, now() - t0);
        }
        fflush(stdout);
    }
    return 0;
}
