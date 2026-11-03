/*
 * BitLM - ovladac HW jadra (rtl/accelerator/bitlm_core.v) nezavisly na sbernici.
 *
 * Pristup k registrum je abstrahovan dvojici funkci rd/wr, takze stejny kod
 * bezi nad Verilator simulaci (tools/sim/bitlm/tb_bitlm.cpp) i nad AWS F2
 * (sw/bitlm/host/bitlm_f2.c, fpga_pci_peek/poke na OCL BAR0).
 */
#ifndef BITLM_HW_H
#define BITLM_HW_H

#include <stdint.h>
#include "../golden/bitlm_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

/* registry (bajtove offsety) */
#define BLR_ID          0x000
#define BLR_PARAMS      0x004
#define BLR_CTRL        0x008
#define BLR_STATUS      0x00C
#define BLR_SEED        0x010
#define BLR_NSTEPS      0x014
#define BLR_CORPUS_LEN  0x018
#define BLR_GEN_LEN     0x01C
#define BLR_HP0         0x020   /* g2<<24 | g1<<16 | s2<<8 | s1 */
#define BLR_HP1         0x024   /* lrb<<24 | lre<<16 | lr2<<8 | lr1 */
#define BLR_HP2         0x028   /* flags<<16 | clip */
#define BLR_STEPS_DONE  0x02C
#define BLR_CORRECT     0x030
#define BLR_SUM_PT_LO   0x034
#define BLR_SUM_PT_HI   0x038
#define BLR_RNG         0x03C
#define BLR_CYCLES_LO   0x040
#define BLR_CYCLES_HI   0x044
#define BLR_UPD_CTR     0x048   /* citac kroku pro stochasticke zaokrouhleni */
#define BLR_CTX         0x080   /* 4 tokeny na slovo */

/* pametova okna */
#define BLM_GEN         0x010000   /* 4 Ki tokenu */
#define BLM_CORPUS      0x400000   /* 4 MiB: az 4 Mi tokenu */
#define BLM_W1          0x800000   /* 4 MiB: az 1 Mi vah */
#define BLM_W2          0xC00000   /* 1 MiB: az 256 Ki vah */
#define BLM_EMB         0xD00000   /* 64 KiB: az 16 Ki hodnot */

#define BL_CMD_INIT        1
#define BL_CMD_TRAIN       2
#define BL_CMD_GEN         3
#define BL_CMD_CLEAR_STATS 4

#define BL_ST_BUSY  1u
#define BL_ST_DONE  2u
#define BL_ST_ERR   4u

#define BL_ID_VALUE 0x424C4D31u

typedef struct {
    uint32_t (*rd)(void *ctx, uint32_t addr);
    void (*wr)(void *ctx, uint32_t addr, uint32_t val);
    void *ctx;
    /* volitelne: cekani mezi dotazy na STATUS (NULL = aktivni cekani) */
    void (*idle)(void *ctx);
} bitlm_dev;

typedef struct {
    uint32_t steps, correct;
    uint64_t sum_pt, cycles;
    uint32_t rng;
} bitlm_stats;

int bitlm_probe(const bitlm_dev *d);  /* 0 = OK (ID i rozmery odpovidaji) */
void bitlm_set_hparams(const bitlm_dev *d, const bl_hparams *hp);
void bitlm_load_corpus(const bitlm_dev *d, const uint8_t *tok, uint32_t len);
/* prikazy vraci STATUS po dokonceni (BL_ST_ERR pri chybe), -1 timeout */
int bitlm_cmd(const bitlm_dev *d, uint32_t cmd, uint64_t max_polls);
int bitlm_init(const bitlm_dev *d, uint32_t seed);
int bitlm_train(const bitlm_dev *d, uint32_t len, uint32_t nsteps);
int bitlm_generate(const bitlm_dev *d, const uint8_t ctx[BL_N], uint32_t count, uint8_t *out);
void bitlm_read_stats(const bitlm_dev *d, bitlm_stats *s);
/* precte vahy z HW do struktury modelu (w1, w2, emb, rng, hp) */
void bitlm_read_model(const bitlm_dev *d, bl_model *m);
/* zapise vahy z modelu do HW (checkpoint) */
void bitlm_write_model(const bitlm_dev *d, const bl_model *m);

/*
 * Bitove presne overeni HW proti referencnimu modelu:
 * INIT, nekolik davek treninku s ruznymi hyperparametry, generovani.
 * Vraci pocet neshod (0 = HW je bitove shodny s referenci).
 */
int bitlm_verify(const bitlm_dev *d, const uint8_t *corpus, uint32_t len,
                 uint32_t steps_per_round, int verbose);

#ifdef __cplusplus
}
#endif
#endif
