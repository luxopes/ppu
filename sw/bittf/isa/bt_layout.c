#include "bt_layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int lw(int w) { return w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : 3; }

int bt_layout_load(bt_layout *l, const char *prefix)
{
    char path[1024], line[512], key[32];
    memset(l, 0, sizeof *l);
    snprintf(path, sizeof path, "%s.layout", prefix);
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return -1;
    }
    l->t = calloc(8192, sizeof *l->t);
    l->V = BT_V;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%31s", key) != 1)
            continue;
        if (!strcmp(key, "config"))
            sscanf(line, "config d %d layers %d heads %d ff %d ctx %d batch %d corpus_len %ld val_len %ld", &l->d,
                   &l->L, &l->H, &l->ff, &l->T, &l->B, &l->corpus_len, &l->val_len);
        else if (!strcmp(key, "vocab"))
            sscanf(line, "vocab %d", &l->V);
        else if (!strcmp(key, "prog"))
            sscanf(line, "prog %lu %d", &l->prog_addr, &l->prog_n);
        else if (!strcmp(key, "memsize"))
            sscanf(line, "memsize %lu", &l->memsize);
        else if (!strcmp(key, "mark")) {
            char m[32];
            int v;
            sscanf(line, "mark %31s %d", m, &v);
            if (!strcmp(m, "fwd")) l->mark_fwd = v;
            if (!strcmp(m, "bwd")) l->mark_bwd = v;
            if (!strcmp(m, "lion")) l->mark_lion = v;
        } else if (!strcmp(key, "t")) {
            bt_tens *t = &l->t[l->nt++];
            sscanf(line, "t %31s %lu %d %d %ld", t->name, &t->addr, &t->width, &t->uns, &t->n);
        }
    }
    fclose(f);
    return 0;
}

const bt_tens *bt_t(const bt_layout *l, const char *name)
{
    for (int i = 0; i < l->nt; i++)
        if (!strcmp(l->t[i].name, name))
            return &l->t[i];
    fprintf(stderr, "layout: chybi tenzor %s\n", name);
    exit(1);
}

const char *bt_addr_name(const bt_layout *l, uint64_t addr, int64_t *elem)
{
    for (int i = 0; i < l->nt; i++) {
        const bt_tens *t = &l->t[i];
        if (addr >= t->addr && addr < t->addr + (uint64_t)(t->n * t->width)) {
            *elem = (int64_t)(addr - t->addr) / t->width;
            return t->name;
        }
    }
    *elem = -1;
    return addr >= l->prog_addr ? "program" : "?";
}

void bt_rd_tensor(const bt_cpu *c, const bt_tens *t, int64_t *out)
{
    for (int64_t i = 0; i < t->n; i++)
        out[i] = bt_load(c, t->addr + (uint64_t)(i * t->width), lw(t->width), t->uns);
}

void bt_wr_tensor(bt_cpu *c, const bt_tens *t, const int64_t *v)
{
    for (int64_t i = 0; i < t->n; i++)
        bt_store(c, t->addr + (uint64_t)(i * t->width), lw(t->width), v[i]);
}

uint8_t *bt_dev_init(bt_cpu *c, const bt_layout *l, const char *prefix, const char *corpus, bt_model *m,
                     bt_rng *rng, unsigned seed, int lr_shift)
{
    char path[1024];
    memset(c, 0, sizeof *c);
    c->size = l->memsize;
    c->mem = calloc(1, l->memsize);
    snprintf(path, sizeof path, "%s.bin", prefix);
    FILE *f = fopen(path, "rb");
    size_t pn = (size_t)l->prog_n * BT_IWORDS * 4;
    if (!f || fread(c->mem + l->prog_addr, 1, pn, f) != pn) {
        fprintf(stderr, "nelze nacist %s\n", path);
        exit(1);
    }
    fclose(f);
    f = fopen(corpus, "rb");
    if (!f) {
        perror(corpus);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len != l->corpus_len) {
        fprintf(stderr, "delka korpusu %ld != %ld z layoutu\n", len, l->corpus_len);
        exit(1);
    }
    uint8_t *data = malloc((size_t)len);
    if (fread(data, 1, (size_t)len, f) != (size_t)len)
        exit(1);
    fclose(f);
    if (l->V == BT_V)
        for (long i = 0; i < len; i++)
            data[i] &= 31;
    for (long i = 0; i < len; i++)
        if (data[i] >= l->V) {
            fprintf(stderr, "token %d na pozici %ld mimo slovnik %d\n", data[i], i, l->V);
            exit(1);
        }
    memcpy(c->mem + bt_t(l, "corpus")->addr, data, (size_t)len);
    bt_rng_init(rng, seed);
    bt_model_init_v(m, l->V, l->d, l->L, l->H, l->ff, l->T, lr_shift, rng);
    char nm[48];
    for (int i = 0; i < m->np; i++) {
        snprintf(nm, sizeof nm, "q.%s", m->p[i].name);
        bt_wr_tensor(c, bt_t(l, nm), m->p[i].q);
    }
    c->csr[BT_CSR_SEED] = seed;
    c->csr[BT_CSR_CTR] = rng->ctr;
    return data;
}

uint32_t bt_dev_hash(const bt_cpu *c, const bt_layout *l, const bt_model *m)
{
    uint32_t h = 0;
    char nm[48];
    const bt_tens *em = bt_t(l, "em");
    for (int i = 0; i < m->np; i++) {
        for (int k = 0; k < 2; k++) {
            snprintf(nm, sizeof nm, "%s.%s", k ? "m" : "q", m->p[i].name);
            const bt_tens *t = bt_t(l, nm);
            int64_t *buf = malloc(sizeof(int64_t) * (size_t)t->n);
            bt_rd_tensor(c, t, buf);
            h = bt_fmix32(h ^ bt_chk(buf, t->n));
            free(buf);
        }
        h = bt_fmix32(h ^ (uint32_t)bt_load(c, em->addr + 8 * (uint64_t)i, 3, 0));
    }
    return bt_fmix32(h ^ c->csr[BT_CSR_CTR]);
}
