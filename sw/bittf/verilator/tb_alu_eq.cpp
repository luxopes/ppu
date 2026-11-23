// Ekvivalence bittf_mini_alu vs bittf_alu: nahodne a krajni vstupy pro vsechny funkce EW.
#include "Valu_eq.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

static uint64_t s = 88172645463325252ull;
static uint64_t rnd64() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static Valu_eq *t;
static void tick() { t->clk = 0; t->eval(); t->clk = 1; t->eval(); }

static int64_t val() {
    switch (rnd64() % 10) {
    case 0: return (int64_t)rnd64();
    case 1: return (int64_t)(rnd64() % 200) - 100;
    case 2: { int k = rnd64() % 64; int64_t v = (int64_t)1 << k; return (rnd64() & 1) ? -v : v; }
    case 3: { int k = rnd64() % 64; int64_t v = ((int64_t)1 << k) + (int64_t)(rnd64() % 3) - 1; return (rnd64() & 1) ? -v : v; }
    case 4: { const int64_t e[] = {0, 1, -1, INT64_MAX, INT64_MIN, INT64_MIN + 1, 255, 256, 511, 512, 32767, -32768};
              return e[rnd64() % 12]; }
    case 5: return (int64_t)(rnd64() >> (rnd64() % 64)) * ((rnd64() & 1) ? -1 : 1);
    default: return (int64_t)(int32_t)rnd64() >> (rnd64() % 32);
    }
}
static int64_t shv() { return (rnd64() % 4) ? (int64_t)(rnd64() % 150) - 75 : val(); }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    t = new Valu_eq;
    t->rst = 1; tick(); tick(); t->rst = 0;
    long bad = 0, cnt[32] = {0};
    for (long i = 0; i < n && bad < 10; i++) {
        int fn = rnd64() % 29;
        t->fn = fn; t->tc = rnd64() % 5; t->clamp_en = rnd64() % 3 == 0; t->clamp_sym = rnd64() & 1;
        t->cbits = 1 + rnd64() % 64;
        t->a = val(); t->b = val(); t->c = val();
        t->r1 = (rnd64() % 2) ? shv() : val(); t->r2 = (rnd64() % 2) ? shv() : val();
        t->imm = (uint32_t)((rnd64() % 3) ? shv() : val()); t->i0 = (uint32_t)rnd64() % 300; t->i1 = (uint32_t)rnd64() % 300;
        if (rnd64() % 4 == 0) t->i0 = (uint32_t)t->b;   // OHSUB/ONEHOT zasahy
        t->rnd = (uint32_t)rnd64();
        t->start = 1; tick(); t->start = 0;
        int dr = 0, dm = 0; uint64_t yr = 0, ym = 0;
        for (int k = 0; k < 200 && !(dr && dm); k++) {
            t->eval();
            if (t->done_ref && !dr) { dr = 1; yr = t->y_ref; }
            if (t->done_mini && !dm) { dm = 1; ym = t->y_mini; }
            if (!(dr && dm)) tick();
        }
        // y se zapisuje na hrane spolu s done -> cist po tiku
        if (!dr || !dm) { printf("fn %d: chybi done (ref %d mini %d)\n", fn, dr, dm); bad++; continue; }
        cnt[fn]++;
        if (yr != ym) {
            printf("NESHODA fn %d tc %d clamp %d/%d cb %d a %lld b %lld c %lld r1 %lld r2 %lld imm %d rnd %u: ref %lld mini %lld\n",
                   fn, t->tc, t->clamp_en, t->clamp_sym, t->cbits, (long long)t->a, (long long)t->b, (long long)t->c,
                   (long long)t->r1, (long long)t->r2, (int)t->imm, t->rnd, (long long)yr, (long long)ym);
            bad++;
        }
        tick();
    }
    printf("%s (%ld vektoru)\n", bad ? "VYSLEDEK: NESHODA" : "VYSLEDEK: SHODA", n);
    return bad != 0;
}
