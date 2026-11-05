/* Nacteni programu a rozlozeni pameti z prekladace + spolecna inicializace (runner, testbench RTL). */
#ifndef BT_LAYOUT_H
#define BT_LAYOUT_H

#include <stdint.h>

#include "../golden/bittf_ref.h"
#include "bt_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[32];
    uint64_t addr;
    int width, uns;
    int64_t n;
} bt_tens;

typedef struct {
    int V, d, L, H, ff, T, B;
    long corpus_len, val_len;
    uint64_t prog_addr, memsize;
    int prog_n, mark_fwd, mark_bwd, mark_lion;
    bt_tens *t;
    int nt;
} bt_layout;

int bt_layout_load(bt_layout *l, const char *prefix);
const bt_tens *bt_t(const bt_layout *l, const char *name);
void bt_rd_tensor(const bt_cpu *c, const bt_tens *t, int64_t *out);
void bt_wr_tensor(bt_cpu *c, const bt_tens *t, const int64_t *v);
/* pamet zarizeni: program + korpus (&31 jen u slovniku 32) + golden inicializace parametru; vraci korpus (malloc) */
uint8_t *bt_dev_init(bt_cpu *c, const bt_layout *l, const char *prefix, const char *corpus, bt_model *m,
                     bt_rng *rng, unsigned seed, int lr_shift);
uint32_t bt_dev_hash(const bt_cpu *c, const bt_layout *l, const bt_model *m);
/* nazev tenzoru, do ktereho patri adresa (ladeni) */
const char *bt_addr_name(const bt_layout *l, uint64_t addr, int64_t *elem);

#ifdef __cplusplus
}
#endif
#endif
