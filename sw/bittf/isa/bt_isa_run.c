/*
 * bt_isa_run - spusti program prekladace (tools/bitlm/bittf_compile.py) na simulatoru ISA
 * a po kazdem kroku ho porovna s golden modelem (sw/bittf/golden).
 *
 * bt_isa_run <prefix programu> <korpus.tok> [--steps 3] [--total-steps 8000] [--seed 12345]
 *            [--lr-shift 4] [--check 1] [--threads 0]
 *
 * --check 1: po doprednem pruchodu porovna logity, pred Lionem gradienty, po kroku hash stavu.
 * --check 0: jen hash stavu po kroku.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bt_layout.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static bt_layout L;

/* bezi do pc == stop (nebo HALT pri stop < 0) */
static void run_until(bt_cpu *c, int stop)
{
    while (stop < 0 || (int)c->pc != stop) {
        int r = bt_step(c, (uint32_t)L.prog_addr);
        if (r == 1) {
            if (stop >= 0) {
                fprintf(stderr, "HALT pred znackou %d\n", stop);
                exit(1);
            }
            return;
        }
        if (r < 0) {
            fprintf(stderr, "chyba ISA na pc %u: %s\n", c->pc, bt_err_name(-r));
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "pouziti: bt_isa_run <prefix> <korpus.tok> [volby]\n");
        return 2;
    }
    int steps = 3, total = 8000, lr = 4, check = 1, threads = 0;
    unsigned seed = 12345;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--total-steps")) total = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = (unsigned)strtoul(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--lr-shift")) lr = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--check")) check = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "neznamy argument %s\n", argv[i]);
            return 2;
        }
    }
    if (bt_layout_load(&L, argv[1]))
        return 1;
    bt_cpu c;
    bt_rng rng;
    bt_model m;
    uint8_t *data = bt_dev_init(&c, &L, argv[1], argv[2], &m, &rng, seed, lr);
    c.threads = threads;
    long split = L.corpus_len - L.val_len;
    int B = L.B, Tn = L.T, N = B * Tn;
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = malloc((size_t)N), *y = malloc((size_t)N);
    uint32_t *r = malloc(sizeof(uint32_t) * (size_t)B);
    int64_t maxn = (int64_t)N * BT_V;
    for (int i = 0; i < L.nt; i++)
        if (L.t[i].n > maxn && strncmp(L.t[i].name, "corpus", 6))
            maxn = L.t[i].n;
    int64_t *buf = malloc(sizeof(int64_t) * (size_t)maxn);
    char nm[48];
    printf("program %d instrukci, pamet %.1f MiB, model d=%d L=%d H=%d ff=%d T=%d B=%d\n", L.prog_n,
           L.memsize / 1048576.0, L.d, L.L, L.H, L.ff, L.T, B);
    int bad = 0;
    double tg = 0, td = 0;
    for (int step = 1; step <= steps && !bad; step++) {
        if (step == total / 2 || step == total * 3 / 4 || step == total * 7 / 8)
            m.lr_shift--;
        bt_store(&c, bt_t(&L, "lr")->addr, 3, m.lr_shift);
        double t0 = now();
        bt_rng_draw(&rng, B, r);
        for (int b = 0; b < B; b++) {
            long ix = Tn + (long)(r[b] % (uint32_t)(split - 1 - Tn));
            memcpy(x + (long)b * Tn, data + ix - Tn, (size_t)Tn);
            memcpy(y + (long)b * Tn, data + ix - Tn + 1, (size_t)Tn);
        }
        int64_t *lg;
        int *el;
        bt_forward(&m, w, x, 1, &lg, &el);
        tg += now() - t0;
        c.pc = 0;
        if (check) {
            t0 = now();
            run_until(&c, L.mark_bwd);
            td += now() - t0;
            bt_rd_tensor(&c, bt_t(&L, "x_idx"), buf);
            for (int n = 0; n < N; n++)
                if (buf[n] != x[n]) {
                    printf("krok %d: NESHODA vstupnich tokenu (n=%d)\n", step, n);
                    bad = 1;
                    break;
                }
            bt_rd_tensor(&c, bt_t(&L, "logits"), buf);
            uint32_t hd = bt_chk(buf, (int64_t)N * BT_V), hg = bt_chk(lg, (int64_t)N * BT_V);
            bt_rd_tensor(&c, bt_t(&L, "el"), buf);
            int64_t *elv = malloc(sizeof(int64_t) * (size_t)N);
            for (int n = 0; n < N; n++)
                elv[n] = el[n];
            uint32_t ed = bt_chk(buf, N), eg = bt_chk(elv, N);
            free(elv);
            printf("krok %d logity %s (%08x/%08x, el %08x/%08x)\n", step, hd == hg && ed == eg ? "SHODA" : "NESHODA",
                   hd, hg, ed, eg);
            if (hd != hg || ed != eg)
                bad = 1;
            t0 = now();
            bt_backward(&m, w, y, &rng, 0, step);
            tg += now() - t0;
            t0 = now();
            run_until(&c, L.mark_lion);
            td += now() - t0;
            int ng = 0;
            for (int i = 0; i < m.np; i++) {
                int ge;
                const int64_t *gg = bt_work_grad(w, i, &ge);
                snprintf(nm, sizeof nm, "grad.%s", m.p[i].name);
                bt_rd_tensor(&c, bt_t(&L, nm), buf);
                int64_t de = bt_load(&c, bt_t(&L, "gexp")->addr + 8 * (uint64_t)i, 3, 0);
                uint32_t a1 = bt_chk(buf, m.p[i].n), a2 = bt_chk(gg, m.p[i].n);
                if (a1 != a2 || de != ge) {
                    if (ng < 8)
                        printf("  gradient %s NESHODA (%08x/%08x, exp %lld/%d)\n", m.p[i].name, a1, a2, (long long)de, ge);
                    ng++;
                }
            }
            printf("krok %d gradienty %s (%d/%d neshod)\n", step, ng ? "NESHODA" : "SHODA", ng, m.np);
            if (ng)
                bad = 1;
            t0 = now();
            bt_lion(&m, w, &rng);
            tg += now() - t0;
        } else {
            t0 = now();
            bt_backward_update(&m, w, y, &rng, 0, step);
            tg += now() - t0;
        }
        t0 = now();
        run_until(&c, -1);
        td += now() - t0;
        uint32_t h1 = bt_dev_hash(&c, &L, &m), h2 = bt_state_hash(&m, &rng);
        printf("krok %d stav %s (%08x/%08x)  golden %.1f s, ISA %.1f s, %.2f G MAC, %.1f M prvku\n", step,
               h1 == h2 ? "SHODA" : "NESHODA", h1, h2, tg, td, c.nmac / 1e9, c.nelem / 1e6);
        if (h1 != h2)
            bad = 1;
        fflush(stdout);
    }
    printf(bad ? "VYSLEDEK: NESHODA\n" : "VYSLEDEK: SHODA\n");
    return bad;
}
