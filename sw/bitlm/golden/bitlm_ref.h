/*
 * BitLM-nano: bitove presny referencni model 1-bitoveho jazykoveho modelu.
 *
 * Tento soubor je SPECIFIKACE chovani HW jadra rtl/accelerator/bitlm_core.v.
 * Kazda zmena aritmetiky zde musi byt promitnuta i do RTL (a naopak);
 * Verilator testbench tools/sim/bitlm porovnava oba modely bit po bitu.
 *
 * Model (znakovy jazykovy model, slovnik 32 znaku):
 *   kontext N=8 znaku -> embedding E=32 (int8 hodnoty, int16 stinove, trenovatelny)
 *   -> x (D=256, int8) -> BitLinear W1 (H x D, 1 bit: +1/-1) -> ReLU/clamp (uint7,
 *   dynamicka skala 2^-s) -> a (H=512) -> BitLinear W2 (V x H, 1 bit) + bias
 *   -> softmax (Q16, LUT) -> p (V=32)
 *
 * 1-bitove vahy: dopredny i zpetny pruchod pouziva pouze znamenko vahy
 * (scitani/odcitani, zadne nasobeni vah). Pro trenink se drzi "stinove"
 * (latentni) vahy int16, jejichz znamenko je 1-bitova vaha - stejny princip
 * jako BitNet (straight-through estimator). Optimalizace: online SGD
 * (batch 1) s ucicimi kroky ve forme bitovych posunu.
 */
#ifndef BITLM_REF_H
#define BITLM_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BL_V 32                 /* slovnik */
/* Rozmery lze pro experimenty zmenit pri prekladu (-DBL_LOG2_N=4 ...);
 * HW jadro ma stejne parametry v rtl/accelerator/bitlm_core.v. */
#ifndef BL_LOG2_N
#define BL_LOG2_N 3
#endif
#ifndef BL_LOG2_E
#define BL_LOG2_E 5
#endif
#ifndef BL_LOG2_H
#define BL_LOG2_H 9
#endif
#define BL_N (1 << BL_LOG2_N)        /* delka kontextu v tokenech (8) */
#define BL_E (1 << BL_LOG2_E)        /* dimenze embeddingu (32) */
/*
 * Viceurovnovy kontext (BL_MULTISCALE): kontext 128 tokenu se mapuje do 16 slotu.
 * Poslednich 8 znaku ma vlastni slot, starsi znaky se prumeruji v oknech
 * 4, 4, 8, 8, 16, 16, 32, 32 znaku (od nejstarsiho: 32, 32, 16, 16, 8, 8, 4, 4,
 * pak 8 x 1). Prumer = soucet stinovych embeddingu >> (8 + log2(okno)), takze
 * v HW jen scitani a posun. Bez BL_MULTISCALE ma kazdy token vlastni slot.
 */
#ifdef BL_MULTISCALE
#if BL_LOG2_N != 7
#error "BL_MULTISCALE predpoklada kontext 128 tokenu (BL_LOG2_N=7)"
#endif
#define BL_LOG2_SLOTS 4
#else
#define BL_LOG2_SLOTS BL_LOG2_N
#endif
#define BL_SLOTS (1 << BL_LOG2_SLOTS)
#define BL_LOG2_D (BL_LOG2_SLOTS + BL_LOG2_E)
#define BL_D (1 << BL_LOG2_D)        /* vstup W1 (256) */
#define BL_H (1 << BL_LOG2_H)        /* skryta vrstva (512) */
#ifndef BL_LOG2_CORPUS
#define BL_LOG2_CORPUS 22
#endif
#define BL_CORPUS_MAX (1 << BL_LOG2_CORPUS)  /* max. delka korpusu v tokenech (HW pamet, 4 Mi) */
#define BL_GEN_MAX 4096         /* max. delka generovaneho textu (HW BRAM) */
#define BL_Q16 65536

/* Hyperparametry - v HW jsou to registry, meni se bez prekladu bitstreamu. */
typedef struct {
    uint8_t s1;    /* minimalni posun h -> a (skutecny je dynamicky, absmax) */
    uint8_t s2;    /* posun logitu -> z (skala W2 / teplota softmaxu) */
    uint8_t g1;    /* posun gradientu logitu (Q16 -> gz) */
    uint8_t g2;    /* posun gradientu skryte vrstvy (da -> dh) */
    uint8_t lr1;   /* ucici krok W1 (posun) */
    uint8_t lr2;   /* ucici krok W2 (posun) */
    uint8_t lre;   /* ucici krok embeddingu (posun) */
    uint16_t clip; /* oriznuti stinovych vah do <-clip, clip>, max 32767 */
    uint8_t lrb;   /* ucici krok vystupniho biasu (posun) */
    uint8_t flags; /* BL_F_CENTER | BL_F_BIAS */
} bl_hparams;

#define BL_F_CENTER 1  /* 1-bit vaha = sign(w - prumer radku), jako BitNet */
#define BL_F_BIAS   2  /* trenovatelny bias vystupnich logitu */

typedef struct {
    int16_t w1[BL_H][BL_D];   /* stinove vahy W1, 1-bit vaha = (w >= mu ? +1 : -1) */
    int16_t w2[BL_V][BL_H];   /* stinove vahy W2 */
    int16_t emb[BL_V][BL_E];  /* stinovy embedding int16, pouzity x = clamp(emb >> 8, -127, 127) */
    int16_t b2[BL_V];         /* bias logitu */
    int32_t rs1[BL_H];        /* soucty radku W1 (udrzovane pri zapisu radku) */
    int32_t rs2[BL_V];        /* soucty radku W2 */
    uint32_t rng;             /* xorshift32, nikdy 0 */
    bl_hparams hp;
    /* statistiky treninku (od posledniho vynulovani) */
    uint32_t steps;
    uint32_t correct;         /* pocet kroku, kde argmax == cil */
    uint64_t sum_pt;          /* soucet p[cil] v Q16 */
} bl_model;

typedef struct {
    int8_t x[BL_D];
    int32_t h[BL_H];
    uint8_t a[BL_H];
    uint8_t s1dyn;            /* pouzity posun h -> a */
    int32_t z[BL_V];
    uint32_t e[BL_V];
    uint32_t sum;
    uint32_t r;               /* floor(2^32 / sum) */
    uint32_t p[BL_V];         /* Q16 */
    int argmax;
} bl_fwd;

bl_hparams bl_default_hparams(void);
uint32_t bl_rng_next(uint32_t *state);
void bl_init(bl_model *m, uint32_t seed, bl_hparams hp);
void bl_forward(const bl_model *m, const uint8_t ctx[BL_N], bl_fwd *f);
void bl_train_step(bl_model *m, const uint8_t ctx[BL_N], uint8_t target);
/* Jeden krok s nahodnou pozici v korpusu (stejne jako HW). */
void bl_train_random(bl_model *m, const uint8_t *corpus, uint32_t len);
/* Vzorkovani dalsiho znaku (spotrebuje 1 hodnotu RNG). */
uint8_t bl_sample(bl_model *m, const uint8_t ctx[BL_N]);
/* Generovani: ctx se posouva, vystup do out[0..count-1]. */
void bl_generate(bl_model *m, uint8_t ctx[BL_N], uint32_t count, uint8_t *out);

extern const char BL_VOCAB[BL_V + 1];

#ifdef __cplusplus
}
#endif
#endif
