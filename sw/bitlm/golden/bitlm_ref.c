/*
 * BitLM-nano - bitove presny referencni model (viz bitlm_ref.h).
 * Vsechny posuny vpravo jsou aritmeticke (floor), stejne jako >>> ve Verilogu.
 */
#include "bitlm_ref.h"
#include "bitlm_exp_lut.h"

#include <string.h>

const char BL_VOCAB[BL_V + 1] = " abcdefghijklmnopqrstuvwxyz.,?!-";

/* Aritmeticky posun vpravo s zaokrouhlenim dolu, nezavisly na prekladaci. */
static inline int64_t asr(int64_t v, unsigned s)
{
    return v >= 0 ? (v >> s) : -(((-v) - 1) >> s) - 1;
}

/* Posun se zaokrouhlenim (round half up): (v + 2^(s-1)) >>> s.
 * Na rozdil od floor nema systematicky posun - nutne pro gradienty. */
static inline int64_t rsr(int64_t v, unsigned s)
{
    return s ? asr(v + ((int64_t)1 << (s - 1)), s) : v;
}

static inline int16_t clip16(int32_t v, int32_t c)
{
    if (v > c) return (int16_t)c;
    if (v < -c) return (int16_t)-c;
    return (int16_t)v;
}

static inline int8_t clamp8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

bl_hparams bl_default_hparams(void)
{
    /* vyladeno sweepem na korpusu R.U.R. (docs/bitlm_poc.md) */
    bl_hparams hp = {
        .s1 = 0, .s2 = 4, .g1 = 4, .g2 = 0,
        .lr1 = 14, .lr2 = 8, .lre = 8, .clip = 32767,
        .lrb = 12, .flags = 0,
    };
    return hp;
}

uint32_t bl_rng_next(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

/* Prah 1-bit vahy radku: prumer radku (floor) nebo 0 bez centrovani. */
static inline int32_t mu1(const bl_model *m, int j)
{
    return (m->hp.flags & BL_F_CENTER) ? (int32_t)asr(m->rs1[j], BL_LOG2_D) : 0;
}

static inline int32_t mu2(const bl_model *m, int i)
{
    return (m->hp.flags & BL_F_CENTER) ? (int32_t)asr(m->rs2[i], BL_LOG2_H) : 0;
}

static void recompute_rowsums(bl_model *m)
{
    for (int j = 0; j < BL_H; j++) {
        int32_t s = 0;
        for (int k = 0; k < BL_D; k++) s += m->w1[j][k];
        m->rs1[j] = s;
    }
    for (int i = 0; i < BL_V; i++) {
        int32_t s = 0;
        for (int j = 0; j < BL_H; j++) s += m->w2[i][j];
        m->rs2[i] = s;
    }
}

void bl_init(bl_model *m, uint32_t seed, bl_hparams hp)
{
    memset(m, 0, sizeof(*m));
    m->rng = seed ? seed : 1u;
    m->hp = hp;
    /* poradi spotreby RNG: W1 po radcich, W2 po radcich, embedding */
    for (int j = 0; j < BL_H; j++)
        for (int k = 0; k < BL_D; k++)
            m->w1[j][k] = (int16_t)((int32_t)((bl_rng_next(&m->rng) >> 16) & 0x1FF) - 256);
    for (int i = 0; i < BL_V; i++)
        for (int j = 0; j < BL_H; j++)
            m->w2[i][j] = (int16_t)((int32_t)((bl_rng_next(&m->rng) >> 16) & 0x1FF) - 256);
    for (int c = 0; c < BL_V; c++)
        for (int e = 0; e < BL_E; e++)
            m->emb[c][e] = (int16_t)(((int32_t)((bl_rng_next(&m->rng) >> 24) & 0x3F) - 32) * 256);
    recompute_rowsums(m); /* bias b2 zacina na 0 */
}

/* Rozlozeni slotu: slot s pokryva ctx[slot_start(s) .. +2^slot_lg(s)-1]. */
static inline int slot_lg(int s)
{
#ifdef BL_MULTISCALE
    static const uint8_t lg[16] = { 5, 5, 4, 4, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0 };
    return lg[s];
#else
    (void)s;
    return 0;
#endif
}

static inline int slot_start(int s)
{
#ifdef BL_MULTISCALE
    static const uint8_t st[16] = { 0, 32, 64, 80, 96, 104, 112, 116, 120, 121, 122, 123, 124, 125, 126, 127 };
    return st[s];
#else
    return s;
#endif
}

void bl_forward(const bl_model *m, const uint8_t ctx[BL_N], bl_fwd *f)
{
    const bl_hparams *hp = &m->hp;

    /* x slotu = clamp((soucet stinovych embeddingu v okne) >>> (8 + log2 okna)) */
    for (int s = 0; s < BL_SLOTS; s++) {
        int lg = slot_lg(s), st = slot_start(s);
        for (int e = 0; e < BL_E; e++) {
            int32_t sum = 0;
            for (int n = st; n < st + (1 << lg); n++)
                sum += m->emb[ctx[n] & (BL_V - 1)][e];
            f->x[s * BL_E + e] = clamp8((int32_t)asr(sum, 8 + lg));
        }
    }

    /* BitLinear 1: pouze scitani/odcitani */
    int32_t hmax = 0;
#pragma omp parallel for schedule(static) reduction(max : hmax) if (BL_H * BL_D >= 262144)
    for (int j = 0; j < BL_H; j++) {
        int32_t acc = 0, mu = mu1(m, j);
        for (int k = 0; k < BL_D; k++)
            acc += (m->w1[j][k] >= mu) ? f->x[k] : -f->x[k];
        f->h[j] = acc;
        if (acc > hmax) hmax = acc;
    }
    /* kvantizace aktivaci na uint7 s dynamickou skalou 2^-s (absmax per token,
     * jako BitNet, ale mocnina dvou -> v HW jen detekce nejvyssiho bitu):
     * s = nejmensi posun, aby max(h) >> s <= 127; minimalne hp->s1. */
    unsigned s = hp->s1;
    while ((hmax >> s) > 127) s++;
    f->s1dyn = (uint8_t)s;
    for (int j = 0; j < BL_H; j++) {
        int32_t t = (int32_t)asr(f->h[j], s);
        f->a[j] = (uint8_t)(t < 0 ? 0 : (t > 127 ? 127 : t));
    }

    /* BitLinear 2 + argmax (prvni maximum) */
    int32_t zmax = 0;
    f->argmax = 0;
    for (int i = 0; i < BL_V; i++) {
        int32_t acc = (hp->flags & BL_F_BIAS) ? m->b2[i] : 0, mu = mu2(m, i);
        for (int j = 0; j < BL_H; j++)
            acc += (m->w2[i][j] >= mu) ? f->a[j] : -(int32_t)f->a[j];
        f->z[i] = (int32_t)asr(acc, hp->s2);
        if (i == 0 || f->z[i] > zmax) {
            zmax = f->z[i];
            f->argmax = i;
        }
    }

    /* softmax: exp z LUT, jedno deleni na vzorek (reciprocni hodnota) */
    uint32_t sum = 0;
    for (int i = 0; i < BL_V; i++) {
        int32_t d = zmax - f->z[i];
        if (d > 255) d = 255;
        f->e[i] = BL_EXP_LUT[d];
        sum += f->e[i];
    }
    f->sum = sum;                                   /* >= 65536 */
    f->r = (uint32_t)((1ull << 32) / sum);          /* <= 65536 */
    for (int i = 0; i < BL_V; i++)
        f->p[i] = (uint32_t)(((uint64_t)f->e[i] * f->r) >> 16);
}

void bl_train_step(bl_model *m, const uint8_t ctx[BL_N], uint8_t target)
{
    const bl_hparams *hp = &m->hp;
    static bl_fwd f;
    bl_forward(m, ctx, &f);
    target &= (BL_V - 1);

    m->steps++;
    m->correct += (f.argmax == target);
    m->sum_pt += f.p[target];

    /* gradient logitu: p - onehot (Q16) */
    int32_t gz[BL_V];
    for (int i = 0; i < BL_V; i++)
        gz[i] = (int32_t)rsr((int32_t)f.p[i] - (i == target ? BL_Q16 : 0), hp->g1);

    /* W2: da pocita se STARYMI 1-bit vahami, pak update stinovych vah */
    static int32_t da[BL_H];
    memset(da, 0, sizeof(da));
    for (int i = 0; i < BL_V; i++) {
        int32_t mu = mu2(m, i), rs = 0;
        for (int j = 0; j < BL_H; j++) {
            int16_t w = m->w2[i][j];
            da[j] += (w >= mu) ? gz[i] : -gz[i];
            w = clip16(w - (int32_t)rsr((int64_t)gz[i] * f.a[j], hp->lr2), hp->clip);
            m->w2[i][j] = w;
            rs += w;
        }
        m->rs2[i] = rs;
        if (hp->flags & BL_F_BIAS)
            m->b2[i] = clip16(m->b2[i] - (int32_t)rsr(gz[i], hp->lrb), 32767);
    }

    /* derivace clamp: propousti jen 0 < a < 127 (STE) */
    static int32_t dh[BL_H];
    for (int j = 0; j < BL_H; j++)
        dh[j] = (f.a[j] != 0 && f.a[j] != 127) ? (int32_t)rsr(da[j], hp->g2) : 0;

    /* W1: dx se STARYMI 1-bit vahami, pak update. Radky jsou nezavisle; dx je
     * celociselny soucet, takze paralelni scitani dava bitove stejny vysledek. */
    static int32_t dx[BL_D];
    memset(dx, 0, sizeof(dx));
#pragma omp parallel if (BL_H * BL_D >= 262144)
    {
        int32_t dxl[BL_D];
        memset(dxl, 0, sizeof(dxl));
#pragma omp for schedule(dynamic, 16)
        for (int j = 0; j < BL_H; j++) {
            if (dh[j] == 0)
                continue; /* HW radek preskakuje stejne - bez vlivu na vysledek */
            int32_t mu = mu1(m, j), rs = 0;
            for (int k = 0; k < BL_D; k++) {
                int16_t w = m->w1[j][k];
                dxl[k] += (w >= mu) ? dh[j] : -dh[j];
                w = clip16(w - (int32_t)rsr((int64_t)dh[j] * f.x[k], hp->lr1), hp->clip);
                m->w1[j][k] = w;
                rs += w;
            }
            m->rs1[j] = rs;
        }
#pragma omp critical
        for (int k = 0; k < BL_D; k++) dx[k] += dxl[k];
    }

    /* embedding: pozice n vzestupne, opakovany znak se aktualizuje postupne */
    /* gradient slotu se deli rovnym dilem mezi znaky okna: posun lre + log2(okno);
     * poradi: sloty vzestupne, uvnitr okna pozice vzestupne */
    for (int s = 0; s < BL_SLOTS; s++) {
        int lg = slot_lg(s), st = slot_start(s);
        for (int n = st; n < st + (1 << lg); n++) {
            int c = ctx[n] & (BL_V - 1);
            for (int e = 0; e < BL_E; e++)
                m->emb[c][e] = clip16(m->emb[c][e] - (int32_t)rsr(dx[s * BL_E + e], hp->lre + lg), 32767);
        }
    }
}

void bl_train_random(bl_model *m, const uint8_t *corpus, uint32_t len)
{
    uint32_t r = bl_rng_next(&m->rng);
    /* plnych 32 bitu RNG: kazda pozice korpusu je dosazitelna
     * (v1 pouzivala jen r >> 16, tj. max. 65536 ruznych pozic) */
    uint32_t pos = BL_N + (uint32_t)(((uint64_t)r * (len - BL_N)) >> 32);
    bl_train_step(m, corpus + pos - BL_N, corpus[pos]);
}

uint8_t bl_sample(bl_model *m, const uint8_t ctx[BL_N])
{
    static bl_fwd f;
    bl_forward(m, ctx, &f);
    uint32_t u = bl_rng_next(&m->rng) & 0xFFFF;
    uint32_t cum = 0;
    for (int i = 0; i < BL_V; i++) {
        cum += f.p[i];
        if (cum > u)
            return (uint8_t)i;
    }
    return BL_V - 1;
}

void bl_generate(bl_model *m, uint8_t ctx[BL_N], uint32_t count, uint8_t *out)
{
    for (uint32_t g = 0; g < count; g++) {
        uint8_t t = bl_sample(m, ctx);
        out[g] = t;
        memmove(ctx, ctx + 1, BL_N - 1);
        ctx[BL_N - 1] = t;
    }
}
