/*
 * BitTF golden model - viz bittf_ref.h. Doslovny prepis tools/bitlm/intsim.py.
 * Vsechny aritmeticke kroky jsou celociselne; float jen v bittf_train.c pro mereni CE.
 */
#include "bittf_ref.h"
#include "bittf_tables.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NEG_INF_SC (-(1LL << 40))
#define TOP_NONE (-1000000000)

/* ------------------------------------------------------------------ primitiva */
static inline int64_t shl(int64_t a, int64_t s) { return (int64_t)((uint64_t)a << s); }

/* a / 2^s zaokrouhlene pul nahoru; s <= 0 presny posun vlevo */
static inline int64_t rsr(int64_t a, int64_t s)
{
    if (s > 0) {
        int sp = s > 62 ? 62 : (int)s;
        return (a + ((1LL << sp) >> 1)) >> sp;
    }
    return shl(a, -s);
}

/* floor(a / 2^s + r / 2^s): stochasticke zaokrouhleni */
static inline int64_t srd(int64_t a, int64_t s, uint32_t r)
{
    if (s > 0) {
        int sp = s > 32 ? 32 : (int)s;
        return (a + ((int64_t)r & ((1LL << sp) - 1))) >> sp;
    }
    return shl(a, -s);
}

static inline int bitlen(int64_t x) { return x ? 64 - __builtin_clzll((uint64_t)x) : 0; }
static inline int64_t clampi(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int64_t sgn(int64_t v) { return (v > 0) - (v < 0); }
static inline int64_t iabs(int64_t v) { return v < 0 ? -v : v; }
static inline int64_t floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
        q--;
    return q;
}
static inline int imax(int a, int b) { return a > b ? a : b; }

static int64_t amax(const int64_t *m, int64_t n)
{
    int64_t a = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = iabs(m[i]);
        if (v > a)
            a = v;
    }
    return a;
}

static void *xmalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) {
        fprintf(stderr, "bittf: malo pameti\n");
        exit(1);
    }
    return p;
}
#define ALLOC64(n) ((int64_t *)xmalloc(sizeof(int64_t) * (size_t)(n)))
#define ALLOCI(n) ((int *)xmalloc(sizeof(int) * (size_t)(n)))

/* ------------------------------------------------------------------ RNG */
uint32_t bt_fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

void bt_rng_init(bt_rng *r, uint32_t seed) { r->seed = seed, r->ctr = 0; }

void bt_rng_draw(bt_rng *r, int64_t n, uint32_t *out)
{
    uint32_t key = bt_fmix32(r->seed ^ bt_fmix32(r->ctr));
    r->ctr++;
    for (int64_t i = 0; i < n; i++)
        out[i] = bt_fmix32((uint32_t)((uint32_t)i * 0x9E3779B1u) + key);
}

static uint32_t *rbuf;
static int64_t rbuf_n;
static const uint32_t *draw(bt_rng *r, int64_t n)
{
    if (n > rbuf_n) {
        free(rbuf);
        rbuf = (uint32_t *)xmalloc(sizeof(uint32_t) * (size_t)n);
        rbuf_n = n;
    }
    bt_rng_draw(r, n, rbuf);
    return rbuf;
}

uint32_t bt_chk(const int64_t *v, int64_t n)
{
    uint32_t s = 0;
    for (int64_t i = 0; i < n; i++)
        s += bt_fmix32((uint32_t)v[i] ^ bt_fmix32((uint32_t)i));
    return s;
}

/* ------------------------------------------------------------------ kvantizace */
static void normalize(int64_t *m, int64_t n, int *e, int bits)
{
    int64_t a = amax(m, n);
    if (!a)
        return;
    int s = bitlen(a) - bits;
    for (int64_t i = 0; i < n; i++)
        m[i] = rsr(m[i], s);
    *e += s;
}

/* po radcich delky n: int bits se skalou 2^s; s[r] */
static void quant_tok(const int64_t *y, int64_t rows, int n, int bits, int64_t *out, int *s)
{
    int64_t q = (1LL << (bits - 1)) - 1;
    for (int64_t r = 0; r < rows; r++) {
        const int64_t *yr = y + r * n;
        int sh = bitlen(amax(yr, n)) - (bits - 1);
        for (int i = 0; i < n; i++)
            out[r * n + i] = clampi(rsr(yr[i], sh), -q - 1, q);
        s[r] = sh;
    }
}

static void quant_sr(int64_t *m, int64_t n, int *e, int bits, bt_rng *rng)
{
    int64_t q = (1LL << (bits - 1)) - 1;
    int s = bitlen(amax(m, n)) - (bits - 1);
    const uint32_t *r = draw(rng, n);
    for (int64_t i = 0; i < n; i++)
        m[i] = clampi(srd(m[i], s, r[i]), -q - 1, q);
    *e += s;
}

/* gradient pro dW: znamenko * 2^k nad 2^lo_e, nestranne */
static void pow2q(const int64_t *m, int64_t n, int lo_e, bt_rng *rng, int64_t *out)
{
    const uint32_t *r = draw(rng, n);
    int64_t lo = 1LL << lo_e;
    for (int64_t i = 0; i < n; i++) {
        int64_t a = iabs(m[i]), v;
        int e = bitlen(a) - 1;
        if (e < 0)
            e = 0;
        int64_t b = 1LL << e;
        if (a >= lo)
            v = ((int64_t)r[i] & (b - 1)) < a - b ? b << 1 : b;
        else
            v = ((int64_t)r[i] & (lo - 1)) < a ? lo : 0;
        out[i] = sgn(m[i]) * (v >> lo_e);
    }
}

/* soucet BFP do x: 40 bitu rezervy pod vetsim vrcholem */
static void add_bfp(int64_t *m1, int *e1, const int64_t *m2, int e2, int64_t n)
{
    int64_t a1 = amax(m1, n), a2 = amax(m2, n);
    int t1 = a1 ? *e1 + bitlen(a1) : TOP_NONE;
    int t2 = a2 ? e2 + bitlen(a2) : TOP_NONE;
    if (imax(t1, t2) < -100000000)
        return;
    int base = imax(t1, t2) - 40;
    for (int64_t i = 0; i < n; i++)
        m1[i] = rsr(m1[i], base - *e1) + rsr(m2[i], base - e2);
    *e1 = base;
}

/* ------------------------------------------------------------------ model */
static int kshift(int fan_in) { return (int)floor(log2((double)fan_in) / 2 + 0.5); }

static bt_param *addp(bt_model *m, const char *name, int rows, int cols)
{
    bt_param *p = &m->p[m->np++];
    snprintf(p->name, sizeof p->name, "%s", name);
    p->rows = rows, p->cols = cols, p->n = rows * cols;
    p->q = ALLOC64(p->n);
    p->m = ALLOC64(p->n);
    p->em = 0;
    return p;
}

static void init_gauss(bt_param *p, bt_rng *r)
{
    int64_t *acc = ALLOC64(p->n);
    for (int k = 0; k < 4; k++) {
        const uint32_t *u = draw(r, p->n);
        for (int i = 0; i < p->n; i++)
            acc[i] += u[i] & 4095;
    }
    for (int i = 0; i < p->n; i++)
        p->q[i] = clampi(((acc[i] - 8190) * 2269) >> 12, -32767, 32767);
    free(acc);
}

static void init_unif(bt_param *p, bt_rng *r)
{
    const uint32_t *u = draw(r, p->n);
    for (int i = 0; i < p->n; i++)
        p->q[i] = (int64_t)(u[i] & 8191) - 4096;
}

static void init_gain(bt_param *p)
{
    for (int i = 0; i < p->n; i++)
        p->q[i] = 4096;
}

void bt_model_init(bt_model *m, int d, int L, int H, int ff, int T, int lr_shift, bt_rng *r)
{
    bt_model_init_v(m, BT_V, d, L, H, ff, T, lr_shift, r);
}

void bt_model_init_v(bt_model *m, int V, int d, int L, int H, int ff, int T, int lr_shift, bt_rng *r)
{
    char nm[16];
    memset(m, 0, sizeof *m);
    if (V < 2 || V > BT_VMAX) {
        fprintf(stderr, "bittf: slovnik %d mimo 2..%d\n", V, BT_VMAX);
        exit(1);
    }
    m->V = V;
    m->d = d, m->L = L, m->H = H, m->ff = ff, m->T = T, m->dh = d / H;
    m->ks = (bitlen(m->dh) - 1) / 2;
    m->sqd = (int)floor(sqrt((double)d) * 4096 + 0.5);
    m->k_qkv = m->k_proj = m->k_fc1 = kshift(d);
    m->k_fc2 = kshift(ff);
    m->lr_shift = lr_shift;
    m->p = (bt_param *)xmalloc(sizeof(bt_param) * (size_t)(3 + 6 * L));
    init_gauss(addp(m, "tok", m->V, d), r);
    init_gauss(addp(m, "pos", T, d), r);
    for (int l = 0; l < L; l++) {
        snprintf(nm, sizeof nm, "%d.ln1", l), init_gain(addp(m, nm, 1, d));
        snprintf(nm, sizeof nm, "%d.qkv", l), init_unif(addp(m, nm, 3 * d, d), r);
        snprintf(nm, sizeof nm, "%d.proj", l), init_unif(addp(m, nm, d, d), r);
        snprintf(nm, sizeof nm, "%d.ln2", l), init_gain(addp(m, nm, 1, d));
        snprintf(nm, sizeof nm, "%d.fc1", l), init_unif(addp(m, nm, ff, d), r);
        snprintf(nm, sizeof nm, "%d.fc2", l), init_unif(addp(m, nm, d, ff), r);
    }
    init_gain(addp(m, "lnf", 1, d));
}

void bt_model_free(bt_model *m)
{
    for (int i = 0; i < m->np; i++)
        free(m->p[i].q), free(m->p[i].m);
    free(m->p);
}

uint32_t bt_state_hash(const bt_model *m, const bt_rng *r)
{
    uint32_t h = 0;
    for (int i = 0; i < m->np; i++) {
        h = bt_fmix32(h ^ bt_chk(m->p[i].q, m->p[i].n));
        h = bt_fmix32(h ^ bt_chk(m->p[i].m, m->p[i].n));
        h = bt_fmix32(h ^ (uint32_t)m->p[i].em);
    }
    return bt_fmix32(h ^ r->ctr);
}

#define P_TOK(m) (&(m)->p[0])
#define P_POS(m) (&(m)->p[1])
#define P_L(m, l, k) (&(m)->p[2 + 6 * (l) + (k)])   /* 0 ln1 1 qkv 2 proj 3 ln2 4 fc1 5 fc2 */
#define P_LNF(m) (&(m)->p[2 + 6 * (m)->L])

/* ------------------------------------------------------------------ pracovni pamet */
typedef struct {
    int64_t *xh1, *rr1, *a1, *q8, *k8, *v8, *p8, *a2, *xh2, *rr2, *a3, *fo, *a4;
    int *sh1, *ea1, *eq, *ek, *ev, *ea2, *sh2, *ea3, *ea4;
} layer_cache;

struct bt_work {
    int B, N;
    uint8_t *idx;
    int64_t *x, *o, *y, *sc, *yh, *yatt, *po, *logits, *n8, *xhf, *rrf;
    int *eo, *stmp, *el, *en, *shf;
    layer_cache *lc;
    /* zpetny pruchod */
    int64_t *D, *dX, *dXn, *g1, *g2, *gx, *gw, *aa, *dsm, *dsb, *dqk, *qa, *dv, *G, *dg;
    int64_t **grad;
    int *gexp;
};

bt_work *bt_work_new(const bt_model *m, int B)
{
    bt_work *w = (bt_work *)xmalloc(sizeof *w);
    int N = B * m->T, d = m->d, ff = m->ff, H = m->H, T = m->T;
    int64_t att = (int64_t)B * H * T * T, wide = (int64_t)N * (3 * d > ff ? 3 * d : ff);
    w->B = B, w->N = N;
    w->idx = (uint8_t *)xmalloc((size_t)N);
    w->x = ALLOC64((int64_t)N * d);
    w->o = ALLOC64((int64_t)N * 3 * d);
    w->y = ALLOC64(wide);
    w->sc = ALLOC64(att);
    w->yh = ALLOC64((int64_t)N * d);
    w->yatt = ALLOC64((int64_t)N * d);
    w->po = ALLOC64(wide);
    w->logits = ALLOC64((int64_t)N * m->V);
    w->n8 = ALLOC64((int64_t)N * d);
    w->xhf = ALLOC64((int64_t)N * d);
    w->rrf = ALLOC64(N);
    w->eo = ALLOCI(N), w->stmp = ALLOCI((int64_t)B * H * T), w->el = ALLOCI(N), w->en = ALLOCI(N);
    w->shf = ALLOCI(N);
    w->lc = (layer_cache *)xmalloc(sizeof(layer_cache) * (size_t)m->L);
    for (int l = 0; l < m->L; l++) {
        layer_cache *c = &w->lc[l];
        c->xh1 = ALLOC64((int64_t)N * d), c->rr1 = ALLOC64(N), c->a1 = ALLOC64((int64_t)N * d);
        c->q8 = ALLOC64((int64_t)N * d), c->k8 = ALLOC64((int64_t)N * d), c->v8 = ALLOC64((int64_t)N * d);
        c->p8 = ALLOC64(att), c->a2 = ALLOC64((int64_t)N * d);
        c->xh2 = ALLOC64((int64_t)N * d), c->rr2 = ALLOC64(N), c->a3 = ALLOC64((int64_t)N * d);
        c->fo = ALLOC64((int64_t)N * ff), c->a4 = ALLOC64((int64_t)N * ff);
        c->sh1 = ALLOCI(N), c->ea1 = ALLOCI(N), c->eq = ALLOCI((int64_t)B * H * T);
        c->ek = ALLOCI((int64_t)B * H), c->ev = ALLOCI((int64_t)B * H), c->ea2 = ALLOCI(N);
        c->sh2 = ALLOCI(N), c->ea3 = ALLOCI(N), c->ea4 = ALLOCI(N);
    }
    w->D = ALLOC64((int64_t)N * m->V);
    w->dX = ALLOC64((int64_t)N * d), w->dXn = ALLOC64((int64_t)N * d);
    w->g1 = ALLOC64(wide), w->g2 = ALLOC64(wide), w->gx = ALLOC64(wide), w->gw = ALLOC64(wide);
    w->aa = ALLOC64(wide);
    w->dsm = ALLOC64(att), w->dsb = ALLOC64(att);
    w->dqk = ALLOC64((int64_t)N * d), w->qa = ALLOC64((int64_t)N * d), w->dv = ALLOC64((int64_t)N * d);
    w->G = ALLOC64((int64_t)N * 3 * d);
    w->dg = ALLOC64(d);
    w->grad = (int64_t **)xmalloc(sizeof(int64_t *) * (size_t)m->np);
    w->gexp = ALLOCI(m->np);
    for (int i = 0; i < m->np; i++)
        w->grad[i] = ALLOC64(m->p[i].n);
    return w;
}

void bt_work_free(bt_work *w)
{
    (void)w;   /* proces po treninku konci; pamet uvolni OS */
}

/* ------------------------------------------------------------------ maticove operace */
/* out[n][o] = sum_k a[n][k] * sign(W[o][k])  (1-bit vahy: jen scitani) */
static void mm_bin(const int64_t *a, const bt_param *W, int64_t N, int64_t *out)
{
    int O = W->rows, K = W->cols;
    int8_t *s = (int8_t *)xmalloc((size_t)O * K);
    for (int64_t i = 0; i < (int64_t)O * K; i++)
        s[i] = W->q[i] >= 0 ? 1 : -1;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        const int64_t *an = a + n * K;
        for (int o = 0; o < O; o++) {
            const int8_t *so = s + (int64_t)o * K;
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += an[k] * so[k];
            out[n * O + o] = acc;
        }
    }
    free(s);
}

/* out[n][k] = sum_o g[n][o] * sign(W[o][k])  (zpetny pruchod dx) */
static void mm_bin_t(const int64_t *g, const bt_param *W, int64_t N, int64_t *out)
{
    int O = W->rows, K = W->cols;
    int8_t *s = (int8_t *)xmalloc((size_t)O * K);
    for (int64_t i = 0; i < (int64_t)O * K; i++)
        s[i] = W->q[i] >= 0 ? 1 : -1;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        int64_t *on = out + n * K;
        memset(on, 0, sizeof(int64_t) * (size_t)K);
        for (int o = 0; o < O; o++) {
            int64_t gv = g[n * O + o];
            if (!gv)
                continue;
            const int8_t *so = s + (int64_t)o * K;
            for (int k = 0; k < K; k++)
                on[k] += gv * so[k];
        }
    }
    free(s);
}

/* out[o][k] = sum_n g[n][o] * a[n][k] */
static void mm_tn(const int64_t *g, const int64_t *a, int64_t N, int O, int K, int64_t *out)
{
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        int64_t *oo = out + (int64_t)o * K;
        memset(oo, 0, sizeof(int64_t) * (size_t)K);
        for (int64_t n = 0; n < N; n++) {
            int64_t gv = g[n * O + o];
            if (!gv)
                continue;
            const int64_t *an = a + n * K;
            for (int k = 0; k < K; k++)
                oo[k] += gv * an[k];
        }
    }
}

/* ------------------------------------------------------------------ RMS norma */
static void norm_fwd(const bt_model *m, const int64_t *x, int64_t N, int64_t *xh, int64_t *rr, int *sh)
{
    int d = m->d;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        const int64_t *xn = x + n * d;
        int64_t ss = 0;
        for (int c = 0; c < d; c++)
            ss += xn[c] * xn[c];
        int s = bitlen(ss) - 16;
        s += s & 1;
        int64_t mm = s >= 0 ? ss >> s : shl(ss, -s);
        int64_t ix = clampi(mm >> 8, 0, 255);
        int64_t r = rsr((int64_t)m->sqd * BT_RSQ[ix], 16);
        for (int c = 0; c < d; c++)
            xh[n * d + c] = rsr(xn[c] * r, 6 + (s >> 1));
        rr[n] = r;
        sh[n] = s;
    }
}

/* dy (zkopirovano do buf) -> dg (do g_out, exponent *eg) a dx (do dx, exponent *edx) */
static void norm_bwd(const bt_model *m, const int64_t *dy, int edy, int64_t N, const int64_t *xh,
                     const int64_t *rr, const int *sh, const int64_t *g, int64_t *buf,
                     int64_t *g_out, int *eg, int64_t *dx, int *edx)
{
    int d = m->d, e = edy;
    memcpy(buf, dy, sizeof(int64_t) * (size_t)(N * d));
    normalize(buf, N * d, &e, 16);
    for (int c = 0; c < d; c++) {
        int64_t s = 0;
        for (int64_t n = 0; n < N; n++)
            s += buf[n * d + c] * xh[n * d + c];
        g_out[c] = s;
    }
    *eg = e - 12;
    int emaxt = TOP_NONE;
    for (int64_t n = 0; n < N; n++)
        emaxt = imax(emaxt, e - 14 - (sh[n] >> 1));
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        int64_t dot = 0;
        for (int c = 0; c < d; c++)
            dot += buf[n * d + c] * g[c] * xh[n * d + c];
        int64_t mean = floordiv(2 * dot + d, 2 * (int64_t)d);
        int64_t mr = rsr(mean, 12);
        int et = e - 14 - (sh[n] >> 1);
        for (int c = 0; c < d; c++) {
            int64_t dxh = buf[n * d + c] * g[c];
            int64_t t = dxh - rsr(xh[n * d + c] * mr, 12);
            dx[n * d + c] = rsr(t * rr[n], emaxt - et);
        }
    }
    *edx = emaxt;
}

/* ------------------------------------------------------------------ dopredny pruchod */
void bt_forward(const bt_model *m, bt_work *w, const uint8_t *idx, int train, int64_t **logits, int **el)
{
    int B = w->B, T = m->T, N = w->N, d = m->d, H = m->H, dh = m->dh, ff = m->ff;
    int64_t *x = w->x;
    memcpy(w->idx, idx, (size_t)N);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < d; c++)
            x[(int64_t)n * d + c] = P_TOK(m)->q[idx[n] * d + c] + P_POS(m)->q[(n % T) * d + c];

    for (int l = 0; l < m->L; l++) {
        layer_cache *c = &w->lc[train ? l : 0];
        const bt_param *ln1 = P_L(m, l, 0), *qkv = P_L(m, l, 1), *proj = P_L(m, l, 2);
        const bt_param *ln2 = P_L(m, l, 3), *fc1 = P_L(m, l, 4), *fc2 = P_L(m, l, 5);
        /* --- attention */
        norm_fwd(m, x, N, c->xh1, c->rr1, c->sh1);
        for (int64_t i = 0; i < (int64_t)N * d; i++)
            w->y[i] = c->xh1[i] * ln1->q[i % d];
        quant_tok(w->y, N, d, 6, c->a1, c->ea1);
        for (int n = 0; n < N; n++)
            c->ea1[n] -= 24;
        mm_bin(c->a1, qkv, N, w->o);
        for (int n = 0; n < N; n++)
            w->eo[n] = c->ea1[n] - m->k_qkv;
        for (int b = 0; b < B; b++) {
            int emax = TOP_NONE;
            for (int t = 0; t < T; t++)
                emax = imax(emax, w->eo[b * T + t]);
            for (int h = 0; h < H; h++) {
                int bh = b * H + h;
                /* q po tokenech */
                for (int t = 0; t < T; t++) {
                    const int64_t *src = w->o + ((int64_t)(b * T + t)) * 3 * d + h * dh;
                    int64_t *dst = c->q8 + ((int64_t)bh * T + t) * dh;
                    int s = bitlen(amax(src, dh)) - 7;
                    for (int k = 0; k < dh; k++)
                        dst[k] = clampi(rsr(src[k], s), -128, 127);
                    c->eq[bh * T + t] = w->eo[b * T + t] + s;
                }
                /* k, v po (sekvence, hlava) */
                for (int part = 1; part <= 2; part++) {
                    int64_t *dst = (part == 1 ? c->k8 : c->v8) + (int64_t)bh * T * dh;
                    int64_t a = 0;
                    for (int t = 0; t < T; t++) {
                        const int64_t *src = w->o + ((int64_t)(b * T + t)) * 3 * d + part * d + h * dh;
                        for (int k = 0; k < dh; k++) {
                            dst[t * dh + k] = rsr(src[k], emax - w->eo[b * T + t]);
                            if (iabs(dst[t * dh + k]) > a)
                                a = iabs(dst[t * dh + k]);
                        }
                    }
                    int s = bitlen(a) - 7;
                    for (int i = 0; i < T * dh; i++)
                        dst[i] = clampi(rsr(dst[i], s), -128, 127);
                    (part == 1 ? c->ek : c->ev)[bh] = emax + s;
                }
            }
        }
#pragma omp parallel for schedule(static)
        for (int bh = 0; bh < B * H; bh++) {
            const int64_t *q8 = c->q8 + (int64_t)bh * T * dh, *k8 = c->k8 + (int64_t)bh * T * dh;
            const int64_t *v8 = c->v8 + (int64_t)bh * T * dh;
            int64_t *sc = w->sc + (int64_t)bh * T * T, *p8 = c->p8 + (int64_t)bh * T * T;
            int64_t *yh = w->yh + (int64_t)bh * T * dh;
            for (int i = 0; i < T; i++) {
                int sh = -(c->eq[bh * T + i] + c->ek[bh] - m->ks) - 5;
                int64_t mx = NEG_INF_SC;
                for (int j = 0; j < T; j++) {
                    int64_t v = NEG_INF_SC;
                    if (j <= i) {
                        int64_t s = 0;
                        for (int k = 0; k < dh; k++)
                            s += q8[i * dh + k] * k8[j * dh + k];
                        v = rsr(s, sh);
                    }
                    sc[i * T + j] = v;
                    if (v > mx)
                        mx = v;
                }
                int64_t ssum = 0;
                for (int j = 0; j < T; j++) {
                    int64_t e8 = 0;
                    if (j <= i)
                        e8 = BT_EXP8[clampi(mx - sc[i * T + j], -(1LL << 62), 255)];
                    p8[i * T + j] = e8;
                    ssum += e8;
                }
                int64_t rcp = floordiv((1LL << 24) + (ssum >> 1), ssum);
                for (int j = 0; j < T; j++) {
                    int64_t p = (p8[i * T + j] * rcp + (1LL << 15)) >> 16;
                    p8[i * T + j] = p > 255 ? 255 : p;
                }
                for (int k = 0; k < dh; k++) {
                    int64_t s = 0;
                    for (int j = 0; j <= i; j++)
                        s += p8[i * T + j] * v8[j * dh + k];
                    yh[i * dh + k] = s;
                }
            }
        }
        /* hlavy na spolecny exponent (po sekvenci) a kvantizace vstupu proj */
        for (int b = 0; b < B; b++) {
            int eatt = TOP_NONE;
            for (int h = 0; h < H; h++)
                eatt = imax(eatt, c->ev[b * H + h] - 8);
            for (int h = 0; h < H; h++)
                for (int t = 0; t < T; t++)
                    for (int k = 0; k < dh; k++)
                        w->yatt[((int64_t)(b * T + t)) * d + h * dh + k] =
                            rsr(w->yh[(((int64_t)(b * H + h)) * T + t) * dh + k], eatt - (c->ev[b * H + h] - 8));
            for (int t = 0; t < T; t++)
                w->stmp[b * T + t] = eatt;
        }
        quant_tok(w->yatt, N, d, 6, c->a2, c->ea2);
        for (int n = 0; n < N; n++)
            c->ea2[n] += w->stmp[n];
        mm_bin(c->a2, proj, N, w->po);
        for (int n = 0; n < N; n++)
            for (int j = 0; j < d; j++)
                x[(int64_t)n * d + j] += rsr(w->po[(int64_t)n * d + j], -(c->ea2[n] - m->k_proj) - 16);
        /* --- MLP s ReLU^2 */
        norm_fwd(m, x, N, c->xh2, c->rr2, c->sh2);
        for (int64_t i = 0; i < (int64_t)N * d; i++)
            w->y[i] = c->xh2[i] * ln2->q[i % d];
        quant_tok(w->y, N, d, 6, c->a3, c->ea3);
        for (int n = 0; n < N; n++)
            c->ea3[n] -= 24;
        mm_bin(c->a3, fc1, N, c->fo);
        for (int64_t i = 0; i < (int64_t)N * ff; i++) {
            int64_t r = c->fo[i] > 0 ? c->fo[i] : 0;
            w->y[i] = r * r;
        }
        quant_tok(w->y, N, ff, 6, c->a4, c->ea4);
        for (int n = 0; n < N; n++)
            c->ea4[n] += 2 * (c->ea3[n] - m->k_fc1);
        mm_bin(c->a4, fc2, N, w->po);
        for (int n = 0; n < N; n++)
            for (int j = 0; j < d; j++) {
                int64_t v = x[(int64_t)n * d + j] + rsr(w->po[(int64_t)n * d + j], -(c->ea4[n] - m->k_fc2) - 16);
                if (iabs(v) >= (1LL << 31)) {
                    fprintf(stderr, "bittf: rezidualni proud pretekl int32\n");
                    exit(1);
                }
                x[(int64_t)n * d + j] = v;
            }
    }
    norm_fwd(m, x, N, w->xhf, w->rrf, w->shf);
    for (int64_t i = 0; i < (int64_t)N * d; i++)
        w->y[i] = w->xhf[i] * P_LNF(m)->q[i % d];
    quant_tok(w->y, N, d, 8, w->n8, w->en);
    for (int n = 0; n < N; n++) {
        w->en[n] -= 24;
        w->el[n] = w->en[n] - 16;
        for (int v = 0; v < m->V; v++) {
            int64_t s = 0;
            for (int k = 0; k < d; k++)
                s += w->n8[(int64_t)n * d + k] * P_TOK(m)->q[v * d + k];
            w->logits[(int64_t)n * m->V + v] = s;
        }
    }
    *logits = w->logits;
    *el = w->el;
}

/* ------------------------------------------------------------------ zpetny pruchod */
/* G (N x O, exponent eG) -> dx (N x K) do dx/edx, dW do grad[pi]/gexp[pi] */
static void lin_bwd(const bt_model *m, bt_work *w, const int64_t *G, int eG, const int64_t *a, const int *ea,
                    int pi, int k, int64_t *dx, int *edx, bt_rng *rng)
{
    const bt_param *W = &m->p[pi];
    int O = W->rows, K = W->cols;
    int64_t N = w->N, n_o = N * O;
    int e = eG;
    memcpy(w->gx, G, sizeof(int64_t) * (size_t)n_o);
    normalize(w->gx, n_o, &e, 16);
    int64_t A = amax(w->gx, n_o);
    if (!A) {
        memset(dx, 0, sizeof(int64_t) * (size_t)(N * K));
        *edx = 0;
        memset(w->grad[pi], 0, sizeof(int64_t) * (size_t)W->n);
        w->gexp[pi] = 0;
        return;
    }
    /* kopie pro dW (pow2q bere normalizovany gradient pred kvantizaci int8) */
    memcpy(w->gw, w->gx, sizeof(int64_t) * (size_t)n_o);
    int egx = e;
    quant_sr(w->gx, n_o, &egx, 8, rng);
    mm_bin_t(w->gx, W, N, dx);
    *edx = egx - k;
    int lo_e = bitlen(A) - 7;
    pow2q(w->gw, n_o, lo_e, rng, w->gw);
    int base = TOP_NONE;
    for (int64_t n = 0; n < N; n++)
        base = imax(base, ea[n]);
    base -= 8;
    for (int64_t n = 0; n < N; n++)
        for (int i = 0; i < K; i++)
            w->aa[n * K + i] = rsr(a[n * K + i], base - ea[n]);
    mm_tn(w->gw, w->aa, N, O, K, w->grad[pi]);
    w->gexp[pi] = e + lo_e + base;
}

/* gradient vystupu attention dY (N x d) -> gradient vystupu qkv do w->G, vraci exponent */
static int attn_bwd(const bt_model *m, bt_work *w, layer_cache *c, const int64_t *dY, int edY, bt_rng *rng)
{
    int B = w->B, T = m->T, d = m->d, H = m->H, dh = m->dh;
    int64_t N = w->N, BH = (int64_t)B * H, TT = (int64_t)T * T;
    int e = edY;
    int64_t *dy8 = w->g1;   /* B,H,T,dh */
    memcpy(w->g2, dY, sizeof(int64_t) * (size_t)(N * d));
    normalize(w->g2, N * d, &e, 16);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int t = 0; t < T; t++)
                for (int k = 0; k < dh; k++)
                    dy8[(((int64_t)(b * H + h)) * T + t) * dh + k] = w->g2[((int64_t)(b * T + t)) * d + h * dh + k];
    int ey = e;
    quant_sr(dy8, N * d, &ey, 8, rng);
    int evmax = TOP_NONE;
    for (int64_t bh = 0; bh < BH; bh++)
        evmax = imax(evmax, c->ev[bh]);
#pragma omp parallel for schedule(static)
    for (int64_t bh = 0; bh < BH; bh++) {
        const int64_t *p8 = c->p8 + bh * TT, *v8 = c->v8 + bh * T * dh, *dy = dy8 + bh * T * dh;
        int64_t *dv = w->dv + bh * T * dh, *ds = w->dsm + bh * TT;
        for (int j = 0; j < T; j++)
            for (int k = 0; k < dh; k++) {
                int64_t s = 0;
                for (int i = j; i < T; i++)
                    s += p8[i * T + j] * dy[i * dh + k];
                dv[j * dh + k] = s;
            }
        for (int i = 0; i < T; i++) {
            int64_t rowdot = 0;
            for (int j = 0; j < T; j++) {
                int64_t s = 0;
                for (int k = 0; k < dh; k++)
                    s += dy[i * dh + k] * v8[j * dh + k];
                ds[i * T + j] = s;            /* dp */
                rowdot += p8[i * T + j] * s;
            }
            int64_t rd = rsr(rowdot, 8);
            for (int j = 0; j < T; j++)
                ds[i * T + j] = rsr(p8[i * T + j] * (ds[i * T + j] - rd), evmax - c->ev[bh]);
        }
    }
    int eds = ey - 8 + evmax;
    normalize(w->dsm, BH * TT, &eds, 16);
    quant_sr(w->dsm, BH * TT, &eds, 8, rng);
    /* dq = ds8 k8, dk = ds8^T qa */
    int ekmax = TOP_NONE, qbmax = TOP_NONE;
    int *qb = w->stmp;   /* B*H */
    for (int64_t bh = 0; bh < BH; bh++) {
        ekmax = imax(ekmax, c->ek[bh]);
        int mx = TOP_NONE;
        for (int t = 0; t < T; t++)
            mx = imax(mx, c->eq[bh * T + t]);
        qb[bh] = mx - 8;
        qbmax = imax(qbmax, qb[bh]);
    }
    int64_t *dq = w->dqk, *dk = w->qa;   /* qa se pocita do g2 */
    int64_t *qa = w->g2;
#pragma omp parallel for schedule(static)
    for (int64_t bh = 0; bh < BH; bh++) {
        const int64_t *ds8 = w->dsm + bh * TT, *k8 = c->k8 + bh * T * dh, *q8 = c->q8 + bh * T * dh;
        int64_t *dqb = dq + bh * T * dh, *dkb = dk + bh * T * dh, *qab = qa + bh * T * dh;
        for (int t = 0; t < T; t++)
            for (int k = 0; k < dh; k++)
                qab[t * dh + k] = rsr(q8[t * dh + k], qb[bh] - c->eq[bh * T + t]);
        for (int i = 0; i < T; i++)
            for (int k = 0; k < dh; k++) {
                int64_t s = 0;
                for (int j = 0; j < T; j++)
                    s += ds8[i * T + j] * k8[j * dh + k];
                dqb[i * dh + k] = rsr(s, ekmax - c->ek[bh]);
            }
        for (int j = 0; j < T; j++)
            for (int k = 0; k < dh; k++) {
                int64_t s = 0;
                for (int i = 0; i < T; i++)
                    s += ds8[i * T + j] * qab[i * dh + k];
                dkb[j * dh + k] = rsr(s, qbmax - qb[bh]);
            }
    }
    int pe[3] = {eds - m->ks + ekmax, eds - m->ks + qbmax, ey - 8};
    int64_t *pm[3] = {dq, dk, w->dv};
    int top = TOP_NONE;
    for (int i = 0; i < 3; i++) {
        int64_t a = amax(pm[i], N * d);
        if (a)
            top = imax(top, pe[i] + bitlen(a));
    }
    if (top == TOP_NONE) {
        memset(w->G, 0, sizeof(int64_t) * (size_t)(N * 3 * d));
        return 0;
    }
    int base = top - 40;
    for (int i = 0; i < 3; i++)
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int t = 0; t < T; t++)
                    for (int k = 0; k < dh; k++)
                        w->G[((int64_t)(b * T + t)) * 3 * d + i * d + h * dh + k] =
                            rsr(pm[i][(((int64_t)(b * H + h)) * T + t) * dh + k], base - pe[i]);
    return base;
}

static void lion_step(bt_model *m, bt_work *w, bt_rng *rng)
{
    for (int pi = 0; pi < m->np; pi++) {
        bt_param *p = &m->p[pi];
        int64_t *g = w->grad[pi];
        int eg = w->gexp[pi];
        int64_t ag = amax(g, p->n), am = amax(p->m, p->n);
        if (!ag && !am)
            continue;
        int tg = ag ? eg + bitlen(ag) : TOP_NONE;
        int tm = am ? p->em + bitlen(am) : TOP_NONE;
        int base = imax(tg, tm) - 30;
        for (int i = 0; i < p->n; i++) {
            int64_t gc = rsr(g[i], base - eg);
            int64_t mc = am ? rsr(p->m[i], base - p->em) : 0;
            int64_t u = sgn(7 * mc + gc);
            p->q[i] = clampi(p->q[i] - u * (1LL << m->lr_shift), -32767, 32767);
            g[i] = 127 * mc + gc;    /* novy moment (exponent base - 7) */
        }
        int e = base - 7;
        quant_sr(g, p->n, &e, 8, rng);
        memcpy(p->m, g, sizeof(int64_t) * (size_t)p->n);
        p->em = e;
    }
}

void bt_backward_update(bt_model *m, bt_work *w, const uint8_t *y, bt_rng *rng, int trace, int step)
{
    bt_backward(m, w, y, rng, trace, step);
    bt_lion(m, w, rng);
}

void bt_lion(bt_model *m, bt_work *w, bt_rng *rng) { lion_step(m, w, rng); }

const int64_t *bt_work_grad(const bt_work *w, int pi, int *exp)
{
    *exp = w->gexp[pi];
    return w->grad[pi];
}

void bt_backward(bt_model *m, bt_work *w, const uint8_t *y, bt_rng *rng, int trace, int step)
{
    int B = w->B, T = m->T, d = m->d, ff = m->ff;
    int64_t N = w->N;
    int log2n = bitlen(N) - 1;
    /* softmax + CE */
    for (int64_t n = 0; n < N; n++) {
        int64_t lfx[BT_VMAX], e16[BT_VMAX], mx = NEG_INF_SC * 1024, ssum = 0;
        for (int v = 0; v < m->V; v++) {
            lfx[v] = rsr(w->logits[n * m->V + v], -w->el[n] - 8);
            if (lfx[v] > mx)
                mx = lfx[v];
        }
        for (int v = 0; v < m->V; v++) {
            int64_t dd = (mx - lfx[v] + 4) >> 3;
            e16[v] = dd < 512 ? BT_EXP16[dd] : 0;
            ssum += e16[v];
        }
        int64_t rcp = floordiv((1LL << 31) + (ssum >> 1), ssum);
        for (int v = 0; v < m->V; v++)
            w->D[n * m->V + v] = ((e16[v] * rcp + (1LL << 15)) >> 16) - (v == y[n] ? (1LL << 15) : 0);
    }
    int eD = -15 - log2n;
    /* hlava svazana s embeddingem */
    int base = TOP_NONE;
    for (int64_t n = 0; n < N; n++)
        base = imax(base, w->en[n]);
    base -= 8;
    for (int64_t n = 0; n < N; n++)
        for (int k = 0; k < d; k++)
            w->aa[n * d + k] = rsr(w->n8[n * d + k], base - w->en[n]);
    mm_tn(w->D, w->aa, N, m->V, d, w->grad[0]);
    w->gexp[0] = eD + base;
    int64_t *dn = w->g1;
    for (int64_t n = 0; n < N; n++)
        for (int k = 0; k < d; k++) {
            int64_t s = 0;
            for (int v = 0; v < m->V; v++)
                s += w->D[n * m->V + v] * P_TOK(m)->q[v * d + k];
            dn[n * d + k] = s;
        }
    int eX, pl = 2 + 6 * m->L;
    norm_bwd(m, dn, eD - 16, N, w->xhf, w->rrf, w->shf, P_LNF(m)->q, w->g2, w->grad[pl], &w->gexp[pl],
             w->dX, &eX);
    int64_t *dH = w->po, *dA = w->yatt;
    for (int l = m->L - 1; l >= 0; l--) {
        layer_cache *c = &w->lc[l];
        int pb = 2 + 6 * l, eH, eA, eXn;
        /* --- MLP */
        lin_bwd(m, w, w->dX, eX, c->a4, c->ea4, pb + 5, m->k_fc2, dH, &eH, rng);
        int emaxt = TOP_NONE;
        for (int64_t n = 0; n < N; n++)
            emaxt = imax(emaxt, eH + 1 + c->ea3[n] - m->k_fc1);
        for (int64_t n = 0; n < N; n++)
            for (int f = 0; f < ff; f++) {
                int64_t r = c->fo[n * ff + f] > 0 ? c->fo[n * ff + f] : 0;
                w->y[n * ff + f] = rsr(dH[n * ff + f] * r, emaxt - (eH + 1 + c->ea3[n] - m->k_fc1));
            }
        lin_bwd(m, w, w->y, emaxt, c->a3, c->ea3, pb + 4, m->k_fc1, dA, &eA, rng);
        norm_bwd(m, dA, eA, N, c->xh2, c->rr2, c->sh2, P_L(m, l, 3)->q, w->g2, w->grad[pb + 3],
                 &w->gexp[pb + 3], w->dXn, &eXn);
        add_bfp(w->dX, &eX, w->dXn, eXn, N * d);
        /* --- attention */
        lin_bwd(m, w, w->dX, eX, c->a2, c->ea2, pb + 2, m->k_proj, dA, &eA, rng);
        int eG = attn_bwd(m, w, c, dA, eA, rng);
        lin_bwd(m, w, w->G, eG, c->a1, c->ea1, pb + 1, m->k_qkv, dA, &eA, rng);
        norm_bwd(m, dA, eA, N, c->xh1, c->rr1, c->sh1, P_L(m, l, 0)->q, w->g2, w->grad[pb],
                 &w->gexp[pb], w->dXn, &eXn);
        add_bfp(w->dX, &eX, w->dXn, eXn, N * d);
    }
    /* embedding: token a pozice */
    memset(w->dXn, 0, sizeof(int64_t) * (size_t)(m->V * d));
    for (int64_t n = 0; n < N; n++)
        for (int k = 0; k < d; k++)
            w->dXn[w->idx[n] * d + k] += w->dX[n * d + k];
    add_bfp(w->grad[0], &w->gexp[0], w->dXn, eX, (int64_t)m->V * d);
    memset(w->grad[1], 0, sizeof(int64_t) * (size_t)(T * d));
    for (int64_t n = 0; n < N; n++)
        for (int k = 0; k < d; k++)
            w->grad[1][(n % T) * d + k] += w->dX[n * d + k];
    w->gexp[1] = eX;
    (void)B;
    if (trace) {
        int64_t *elv = ALLOC64(N);
        for (int64_t n = 0; n < N; n++)
            elv[n] = w->el[n];
        printf("trace %d logits %08x el %08x\n", step, bt_chk(w->logits, N * m->V), bt_chk(elv, N));
        free(elv);
        for (int pi = 0; pi < m->np; pi++)
            printf("trace %d grad %s %08x %d\n", step, m->p[pi].name, bt_chk(w->grad[pi], m->p[pi].n), w->gexp[pi]);
    }
}
