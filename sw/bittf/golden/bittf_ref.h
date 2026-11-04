/*
 * BitTF golden model - bitove presny celociselny trenink 1-bit transformeru tak, jak ho bude
 * pocitat cip (specifikace docs/bitlm_poc.md kap. 11, aritmetika docs/bittf_golden.md).
 *
 * Je to doslovny prepis tools/bitlm/intsim.py: stejne zaokrouhleni, stejny hashovy RNG,
 * stejne poradi nahodnych proudu. Shodu overuje tools/bitlm/check_bittf_golden.sh
 * (kontrolni soucty logitu, gradientu a stavu po kazdem kroku).
 */
#ifndef BITTF_REF_H
#define BITTF_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BT_V 32

typedef struct {
    uint32_t seed, ctr;
} bt_rng;

typedef struct {
    char name[16];
    int rows, cols, n;
    int64_t *q;     /* int16 stin */
    int64_t *m;     /* int8 moment Lionu (mantisa) */
    int em;         /* exponent momentu */
} bt_param;

typedef struct {
    int64_t *m;
    int e;
} bt_bfp;           /* mantisy + jeden exponent (po tenzoru) */

typedef struct {
    int d, L, H, ff, T, dh, ks, sqd;
    int k_qkv, k_proj, k_fc1, k_fc2;
    int np;
    bt_param *p;    /* tok, pos, [ln1 qkv proj ln2 fc1 fc2] x L, lnf */
    int lr_shift;
} bt_model;

/* hodnoty pro trenink jedne davky (B sekvenci po T tokenech) */
typedef struct bt_work bt_work;

void bt_rng_init(bt_rng *r, uint32_t seed);
void bt_rng_draw(bt_rng *r, int64_t n, uint32_t *out);
uint32_t bt_fmix32(uint32_t h);
uint32_t bt_chk(const int64_t *v, int64_t n);

void bt_model_init(bt_model *m, int d, int L, int H, int ff, int T, int lr_shift, bt_rng *r);
void bt_model_free(bt_model *m);
uint32_t bt_state_hash(const bt_model *m, const bt_rng *r);

bt_work *bt_work_new(const bt_model *m, int B);
void bt_work_free(bt_work *w);

/* dopredny pruchod: idx[B*T]; logits[B*T*V] a exponenty el[B*T] (ukazatele do w) */
void bt_forward(const bt_model *m, bt_work *w, const uint8_t *idx, int train,
                int64_t **logits, int **el);
/* zpetny pruchod + Lion; y[B*T] = cilove tokeny; trace != 0 vypise kontrolni soucty */
void bt_backward_update(bt_model *m, bt_work *w, const uint8_t *y, bt_rng *r, int trace, int step);

#ifdef __cplusplus
}
#endif
#endif
