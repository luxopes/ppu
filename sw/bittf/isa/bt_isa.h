/*
 * BitTF ISA v1 - referencni simulator instrukci (docs/bittf_isa.md).
 * RTL (rtl/bittf) musi po kazde instrukci dat stejny obsah pameti a registru.
 */
#ifndef BT_ISA_H
#define BT_ISA_H

#include <stdint.h>

#include "bt_isa_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { BT_OK = 0, BT_ERR_FIT = 1, BT_ERR_MMOVF = 2, BT_ERR_OP = 3, BT_ERR_MEM = 4, BT_ERR_LIMIT = 5 };

typedef struct {
    uint8_t *mem;
    uint64_t size;
    int64_t R[BT_NREG];
    uint32_t csr[4];
    uint32_t pc;            /* index instrukce od zacatku programu */
    uint64_t ninstr;        /* statistiky */
    uint64_t nelem, nmac;
    uint32_t err_pc;
    int threads;            /* 0 = jak urci OpenMP */
} bt_cpu;

/* provede program od adresy prog (pc = 0) do HALT; vraci BT_OK nebo kod chyby (csr[ERR]) */
int bt_run(bt_cpu *c, uint32_t prog, uint64_t max_instr);
/* provede jednu instrukci na pc; vraci 1 = HALT, 0 = pokracuj, <0 = -kod chyby */
int bt_step(bt_cpu *c, uint32_t prog);
const char *bt_err_name(int e);

int64_t bt_load(const bt_cpu *c, uint64_t addr, int lw, int uns);
void bt_store(bt_cpu *c, uint64_t addr, int lw, int64_t v);

#ifdef __cplusplus
}
#endif
#endif
