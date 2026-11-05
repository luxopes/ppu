/* Ovladac hosta pro BitTF na FPGA (registry bittf_fpga_top + pamet jadra). Zarizeni je abstraktni
 * (bt_dev), takze stejny kod bezi na AWS F2 (bt_f2_pci.c) i proti Verilator simulaci (tb_cl.cpp). */
#ifndef BT_HOST_H
#define BT_HOST_H

#include <stdint.h>

#include "../isa/bt_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/* registry (offset v BAR0 / OCL) */
#define BT_REG_CTRL      0x00
#define BT_REG_STATUS    0x04
#define BT_REG_PROG      0x08
#define BT_REG_SEED      0x10
#define BT_REG_CTR       0x14
#define BT_REG_ERR       0x18
#define BT_REG_CYCLES    0x1C
#define BT_REG_MEMLO     0x20
#define BT_REG_MEMHI     0x24
#define BT_REG_ID        0x28
#define BT_REG_EXT       0x2C
#define BT_ID            0x42544632u

typedef struct {
    void *ctx;
    void (*wr32)(void *ctx, uint32_t off, uint32_t v);
    uint32_t (*rd32)(void *ctx, uint32_t off);
    /* pamet jadra od adresy 0 (na F2: DDR pres BAR4) */
    void (*mem_wr)(void *ctx, uint64_t addr, const void *src, uint64_t n);
    void (*mem_rd)(void *ctx, uint64_t addr, void *dst, uint64_t n);
    /* chvile cekani mezi dotazy na stav (sim: takty, HW: usleep) */
    void (*idle)(void *ctx);
} bt_dev;

/* kontrola ID a pripravenosti pameti; 0 = OK */
int bt_host_probe(bt_dev *d, int verbose);
/* nahraje obraz pameti (program, korpus, parametry) a registry jadra */
void bt_host_load(bt_dev *d, const bt_layout *l, const bt_cpu *init);
/* jeden krok tréninku: zapise lr, spusti program, ceka na konec; vraci kod chyby jadra (0 = OK) */
int bt_host_step(bt_dev *d, const bt_layout *l, int64_t lr_shift, uint32_t *cycles);
/* precte stav parametru (q, m, em) a CTR a spocita hash jako bt_dev_hash */
uint32_t bt_host_hash(bt_dev *d, const bt_layout *l, const bt_model *m);
/* ulozi q.*, m.*, em a CTR do souboru (textova hlavicka + surova data); 0 = OK */
int bt_host_dump(bt_dev *d, const bt_layout *l, const char *path);

/* cely beh (spolecny pro F2 i simulaci): bt_host_main(d, argc, argv) s argumenty
 *   <prefix programu> <korpus.tok> [--steps N] [--total-steps 8000] [--seed 12345] [--lr-shift 4]
 *   [--check K] [--dump soubor]
 * --check K: golden model na CPU bezi soubezne a hash stavu se porovna kazdych K kroku (0 = bez kontroly).
 * Vraci 0 pri uspechu. */
int bt_host_main(bt_dev *d, int argc, char **argv);

#ifdef __cplusplus
}
#endif
#endif
