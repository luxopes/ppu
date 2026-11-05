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

#include "../golden/bittf_ref.h"
#include "bt_isa.h"

typedef struct {
    char name[32];
    uint64_t addr;
    int width, uns;
    int64_t n;
} tens;

static tens *tt;
static int ntt;
static int cfg_d, cfg_L, cfg_H, cfg_ff, cfg_T, cfg_B;
static long cfg_corpus, cfg_val;
static uint64_t prog_addr, memsize;
static int prog_n, mark_fwd, mark_bwd, mark_lion;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static const tens *T(const char *name)
{
    for (int i = 0; i < ntt; i++)
        if (!strcmp(tt[i].name, name))
            return &tt[i];
    fprintf(stderr, "layout: chybi tenzor %s\n", name);
    exit(1);
}

static int lw(int w) { return w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : 3; }

static void rd_tensor(const bt_cpu *c, const tens *t, int64_t *out)
{
    for (int64_t i = 0; i < t->n; i++)
        out[i] = bt_load(c, t->addr + (uint64_t)(i * t->width), lw(t->width), t->uns);
}

static void wr_tensor(bt_cpu *c, const tens *t, const int64_t *v)
{
    for (int64_t i = 0; i < t->n; i++)
        bt_store(c, t->addr + (uint64_t)(i * t->width), lw(t->width), v[i]);
}

static void load_layout(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    char line[512], key[32];
    tt = calloc(4096, sizeof *tt);
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%31s", key) != 1)
            continue;
        if (!strcmp(key, "config"))
            sscanf(line, "config d %d layers %d heads %d ff %d ctx %d batch %d corpus_len %ld val_len %ld", &cfg_d,
                   &cfg_L, &cfg_H, &cfg_ff, &cfg_T, &cfg_B, &cfg_corpus, &cfg_val);
        else if (!strcmp(key, "prog"))
            sscanf(line, "prog %lu %d", &prog_addr, &prog_n);
        else if (!strcmp(key, "memsize"))
            sscanf(line, "memsize %lu", &memsize);
        else if (!strcmp(key, "mark")) {
            char m[32];
            int v;
            sscanf(line, "mark %31s %d", m, &v);
            if (!strcmp(m, "fwd")) mark_fwd = v;
            if (!strcmp(m, "bwd")) mark_bwd = v;
            if (!strcmp(m, "lion")) mark_lion = v;
        } else if (!strcmp(key, "t")) {
            tens *t = &tt[ntt++];
            sscanf(line, "t %31s %lu %d %d %ld", t->name, &t->addr, &t->width, &t->uns, &t->n);
        }
    }
    fclose(f);
}

/* bezi do pc == stop (nebo HALT pri stop < 0) */
static void run_until(bt_cpu *c, int stop)
{
    while (stop < 0 || (int)c->pc != stop) {
        int r = bt_step(c, (uint32_t)prog_addr);
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

static uint32_t dev_hash(const bt_cpu *c, const bt_model *m)
{
    uint32_t h = 0;
    char nm[48];
    int64_t *buf = malloc(sizeof(int64_t) * 4 * 1024 * 1024);
    const tens *em = T("em");
    for (int i = 0; i < m->np; i++) {
        snprintf(nm, sizeof nm, "q.%s", m->p[i].name);
        const tens *t = T(nm);
        rd_tensor(c, t, buf);
        h = bt_fmix32(h ^ bt_chk(buf, t->n));
        snprintf(nm, sizeof nm, "m.%s", m->p[i].name);
        t = T(nm);
        rd_tensor(c, t, buf);
        h = bt_fmix32(h ^ bt_chk(buf, t->n));
        h = bt_fmix32(h ^ (uint32_t)bt_load(c, em->addr + 8 * (uint64_t)i, 3, 0));
    }
    free(buf);
    return bt_fmix32(h ^ c->csr[BT_CSR_CTR]);
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
    char path[1024];
    snprintf(path, sizeof path, "%s.layout", argv[1]);
    load_layout(path);
    bt_cpu c;
    memset(&c, 0, sizeof c);
    c.size = memsize;
    c.mem = calloc(1, memsize);
    c.threads = threads;
    snprintf(path, sizeof path, "%s.bin", argv[1]);
    FILE *f = fopen(path, "rb");
    if (!f || fread(c.mem + prog_addr, 1, (size_t)prog_n * BT_IWORDS * 4, f) != (size_t)prog_n * BT_IWORDS * 4) {
        fprintf(stderr, "nelze nacist %s\n", path);
        return 1;
    }
    fclose(f);
    /* korpus */
    f = fopen(argv[2], "rb");
    if (!f) {
        perror(argv[2]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len != cfg_corpus) {
        fprintf(stderr, "delka korpusu %ld != %ld z layoutu\n", len, cfg_corpus);
        return 1;
    }
    uint8_t *data = malloc((size_t)len);
    if (fread(data, 1, (size_t)len, f) != (size_t)len)
        return 1;
    fclose(f);
    for (long i = 0; i < len; i++)
        data[i] &= 31;
    memcpy(c.mem + T("corpus")->addr, data, (size_t)len);
    long split = len - cfg_val;

    /* golden + pocatecni stav zarizeni */
    bt_rng rng;
    bt_rng_init(&rng, seed);
    bt_model m;
    bt_model_init(&m, cfg_d, cfg_L, cfg_H, cfg_ff, cfg_T, lr, &rng);
    char nm[48];
    for (int i = 0; i < m.np; i++) {
        snprintf(nm, sizeof nm, "q.%s", m.p[i].name);
        wr_tensor(&c, T(nm), m.p[i].q);
    }
    c.csr[BT_CSR_SEED] = seed;
    c.csr[BT_CSR_CTR] = rng.ctr;
    int B = cfg_B, Tn = cfg_T, N = B * Tn;
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = malloc((size_t)N), *y = malloc((size_t)N);
    uint32_t *r = malloc(sizeof(uint32_t) * (size_t)B);
    int64_t *buf = malloc(sizeof(int64_t) * (size_t)N * 64 > sizeof(int64_t) * 4 * 1024 * 1024 ? sizeof(int64_t) * (size_t)N * 64 : sizeof(int64_t) * 4 * 1024 * 1024);
    printf("program %d instrukci, pamet %.1f MiB, model d=%d L=%d H=%d ff=%d T=%d B=%d\n", prog_n, memsize / 1048576.0,
           cfg_d, cfg_L, cfg_H, cfg_ff, cfg_T, B);
    int bad = 0;
    double tg = 0, td = 0;
    for (int step = 1; step <= steps && !bad; step++) {
        if (step == total / 2 || step == total * 3 / 4 || step == total * 7 / 8)
            m.lr_shift--;
        bt_store(&c, T("lr")->addr, 3, m.lr_shift);
        /* golden */
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
        /* zarizeni: vyber davky + dopredu */
        t0 = now();
        c.pc = 0;
        if (check) {
            run_until(&c, mark_bwd);
            td += now() - t0;
            rd_tensor(&c, T("x_idx"), buf);
            for (int n = 0; n < N; n++)
                if (buf[n] != x[n]) {
                    printf("krok %d: NESHODA vstupnich tokenu (n=%d: %lld != %d)\n", step, n, (long long)buf[n], x[n]);
                    bad = 1;
                    break;
                }
            rd_tensor(&c, T("logits"), buf);
            uint32_t hd = bt_chk(buf, (int64_t)N * BT_V), hg = bt_chk(lg, (int64_t)N * BT_V);
            rd_tensor(&c, T("el"), buf);
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
            run_until(&c, mark_lion);
            td += now() - t0;
            int ng = 0;
            for (int i = 0; i < m.np; i++) {
                int ge;
                const int64_t *gg = bt_work_grad(w, i, &ge);
                snprintf(nm, sizeof nm, "grad.%s", m.p[i].name);
                rd_tensor(&c, T(nm), buf);
                int64_t de = bt_load(&c, T("gexp")->addr + 8 * (uint64_t)i, 3, 0);
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
            t0 = now();
            run_until(&c, -1);
            td += now() - t0;
        } else {
            t0 = now();
            bt_backward_update(&m, w, y, &rng, 0, step);
            tg += now() - t0;
            t0 = now();
            run_until(&c, -1);
            td += now() - t0;
        }
        uint32_t h1 = dev_hash(&c, &m), h2 = bt_state_hash(&m, &rng);
        printf("krok %d stav %s (%08x/%08x)  golden %.1f s, ISA %.1f s, %.2f G MAC, %.1f M prvku\n", step,
               h1 == h2 ? "SHODA" : "NESHODA", h1, h2, tg, td, c.nmac / 1e9, c.nelem / 1e6);
        if (h1 != h2)
            bad = 1;
        fflush(stdout);
    }
    printf(bad ? "VYSLEDEK: NESHODA\n" : "VYSLEDEK: SHODA\n");
    return bad;
}
