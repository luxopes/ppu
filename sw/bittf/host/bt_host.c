/* Ovladac hosta pro BitTF na FPGA, viz bt_host.h. */
#include "bt_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bt_host_probe(bt_dev *d, int verbose)
{
    uint32_t id = d->rd32(d->ctx, BT_REG_ID), ext = d->rd32(d->ctx, BT_REG_EXT);
    if (verbose)
        printf("BitTF: ID %08x, EXT %08x (DDR %s)\n", id, ext, (ext & 1) ? "pripravena" : "NEPRIPRAVENA");
    if (id != BT_ID)
        return 1;
    if (!(ext & 1))
        return 2;
    return 0;
}

void bt_host_load(bt_dev *d, const bt_layout *l, const bt_cpu *init)
{
    d->mem_wr(d->ctx, 0, init->mem, l->memsize);
    d->wr32(d->ctx, BT_REG_MEMLO, 0);
    d->wr32(d->ctx, BT_REG_MEMHI, 0);
    d->wr32(d->ctx, BT_REG_PROG, (uint32_t)l->prog_addr);
    d->wr32(d->ctx, BT_REG_SEED, init->csr[BT_CSR_SEED]);
    d->wr32(d->ctx, BT_REG_CTR, init->csr[BT_CSR_CTR]);
}

int bt_host_step(bt_dev *d, const bt_layout *l, int64_t lr_shift, uint32_t *cycles)
{
    d->mem_wr(d->ctx, bt_t(l, "lr")->addr, &lr_shift, 8);
    uint32_t c0 = d->rd32(d->ctx, BT_REG_CYCLES);
    d->wr32(d->ctx, BT_REG_CTRL, 1);
    uint32_t s;
    for (;;) {
        s = d->rd32(d->ctx, BT_REG_STATUS);
        if (s & 2)
            break;
        d->idle(d->ctx);
    }
    if (cycles)
        *cycles = d->rd32(d->ctx, BT_REG_CYCLES) - c0;
    return (int)(s >> 8 & 0xFF);
}

/* tenzory stavu: q.<p>, m.<p> pro vsechny parametry + em */
static int state_tensors(const bt_layout *l, const bt_tens **out)
{
    int n = 0;
    for (int i = 0; i < l->nt; i++) {
        const char *nm = l->t[i].name;
        if (!strncmp(nm, "q.", 2) || !strncmp(nm, "m.", 2) || !strcmp(nm, "em"))
            out[n++] = &l->t[i];
    }
    return n;
}

uint32_t bt_host_hash(bt_dev *d, const bt_layout *l, const bt_model *m)
{
    bt_cpu c;
    memset(&c, 0, sizeof c);
    c.mem = calloc(l->memsize, 1);
    c.size = l->memsize;
    const bt_tens **ts = malloc(sizeof *ts * (size_t)l->nt);
    int n = state_tensors(l, ts);
    for (int i = 0; i < n; i++)
        d->mem_rd(d->ctx, ts[i]->addr, c.mem + ts[i]->addr, (uint64_t)ts[i]->n * (uint64_t)ts[i]->width);
    c.csr[BT_CSR_CTR] = d->rd32(d->ctx, BT_REG_CTR);
    uint32_t h = bt_dev_hash(&c, l, m);
    free(ts);
    free(c.mem);
    return h;
}

int bt_host_dump(bt_dev *d, const bt_layout *l, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return 1;
    const bt_tens **ts = malloc(sizeof *ts * (size_t)l->nt);
    int n = state_tensors(l, ts);
    fprintf(f, "bittf-state 1\nconfig d %d layers %d heads %d ff %d ctx %d batch %d\nctr %u\ntensors %d\n", l->d, l->L,
            l->H, l->ff, l->T, l->B, d->rd32(d->ctx, BT_REG_CTR), n);
    for (int i = 0; i < n; i++)
        fprintf(f, "t %s %d %d %ld\n", ts[i]->name, ts[i]->width, ts[i]->uns, (long)ts[i]->n);
    fprintf(f, "data\n");
    int rc = 0;
    for (int i = 0; i < n; i++) {
        uint64_t nb = (uint64_t)ts[i]->n * (uint64_t)ts[i]->width;
        uint8_t *b = malloc(nb);
        d->mem_rd(d->ctx, ts[i]->addr, b, nb);
        if (fwrite(b, 1, nb, f) != nb)
            rc = 1;
        free(b);
    }
    free(ts);
    if (fclose(f))
        rc = 1;
    return rc;
}
