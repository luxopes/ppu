// Testbench RTL jadra BitTF v2 (5 radkovych portu, Verilator): RTL a simulator ISA bezi v lockstepu - po kazde instrukci
// se porovna cela pamet, registry a citac RNG; po kazdem kroku navic hash stavu s golden modelem.
//
// tb_bittf <prefix programu> <korpus.tok> [--steps 2] [--lat 2] [--stall 0] [--lockstep 1] [--seed 12345]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>

#include "Vbittf_core2.h"
#include "verilated.h"

extern "C" {
#include "../isa/bt_layout.h"
}

static bt_layout L;
static Vbittf_core2 *top;
static uint8_t *rmem;          // pamet RTL
static uint64_t memsize;
static uint64_t cycles;

#define NP 5
#define LB 64
struct rsp_t { uint64_t t; uint8_t d[LB]; bool wack; };
static std::deque<rsp_t> rq[NP];
static int lat = 2, stall = 0;
static uint32_t lfsr = 12345;
static uint64_t nreq[NP];

static uint32_t rnd32() { lfsr ^= lfsr << 13; lfsr ^= lfsr >> 17; lfsr ^= lfsr << 5; return lfsr; }

static uint32_t getbits(const uint32_t *w, int lo, int n)
{
    uint64_t v = 0;
    for (int b = 0; b < n; b++)
        v |= (uint64_t)((w[(lo + b) >> 5] >> ((lo + b) & 31)) & 1) << b;
    return (uint32_t)v;
}

static void tick()
{
    uint32_t ready = 0, rv = 0, wa = 0;
    for (int p = 0; p < NP; p++)
        if (!stall || rnd32() % 100 >= (uint32_t)stall) ready |= 1u << p;
    top->p_rq_ready = ready;
    for (int p = 0; p < NP; p++) {
        // odpoved na cteni / potvrzeni zapisu (v poradi na portu)
        if (!rq[p].empty() && rq[p].front().t <= cycles) {
            rsp_t &r = rq[p].front();
            if (r.wack) wa |= 1u << p;
            else {
                rv |= 1u << p;
                memcpy((uint8_t *)&top->p_rs_data[p * LB / 4], r.d, LB);
            }
            rq[p].pop_front();
        }
    }
    top->p_rs_valid = rv;
    top->p_wack = wa;
    top->clk = 0;
    top->eval();
    for (int p = 0; p < NP; p++) {
        if (((top->p_rq_valid >> p) & 1) && ((ready >> p) & 1)) {
            nreq[p]++;
            uint64_t line = getbits(&top->p_rq_line[0], 26 * p, 26);
            uint64_t a = line * LB;
            if (a + LB > memsize) {
                fprintf(stderr, "RTL: port %d pristup mimo pamet radek 0x%lx\n", p, (unsigned long)line);
                exit(1);
            }
            rsp_t r;
            r.t = cycles + lat + (stall ? rnd32() % 3 : 0);
            if (!rq[p].empty() && rq[p].back().t > r.t) r.t = rq[p].back().t;
            if ((top->p_rq_we >> p) & 1) {
                const uint8_t *wd = (const uint8_t *)&top->p_rq_wdata[p * LB / 4];
                for (int b = 0; b < LB; b++)
                    if (getbits(&top->p_rq_wstrb[0], LB * p + b, 1)) rmem[a + b] = wd[b];
                r.wack = true;
            } else {
                memcpy(r.d, rmem + a, LB);
                r.wack = false;
            }
            rq[p].push_back(r);
        }
    }
    top->clk = 1;
    top->eval();
    cycles++;
}

static int compare(const bt_cpu *c, uint32_t pc)
{
    if (memcmp(c->mem, rmem, memsize)) {
        for (uint64_t a = 0; a < memsize; a++)
            if (c->mem[a] != rmem[a]) {
                int64_t e;
                const char *nm = bt_addr_name(&L, a, &e);
                const uint32_t *w = (const uint32_t *)(c->mem + L.prog_addr + (uint64_t)pc * BT_IWORDS * 4);
                fprintf(stderr, "NESHODA pameti po instrukci pc %u (op %u fn %u): adresa 0x%lx (%s[%ld]) ISA %02x RTL %02x\n",
                        pc, w[0] & 0xFF, (w[0] >> 8) & 0xFF, (unsigned long)a, nm, (long)e, c->mem[a], rmem[a]);
                return 1;
            }
    }
    for (int r = 0; r < BT_NREG; r++) {
        top->dbg_ridx = r;
        top->eval();
        if ((int64_t)top->dbg_rdata != c->R[r]) {
            fprintf(stderr, "NESHODA registru R%d po pc %u: ISA %lld RTL %lld\n", r, pc, (long long)c->R[r],
                    (long long)top->dbg_rdata);
            return 1;
        }
    }
    top->csr_addr = 1;
    top->eval();
    if (top->csr_rdata != c->csr[BT_CSR_CTR]) {
        fprintf(stderr, "NESHODA CTR po pc %u: ISA %u RTL %u\n", pc, c->csr[BT_CSR_CTR], top->csr_rdata);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "pouziti: tb_bittf <prefix> <korpus> [--steps N] [--lat N] [--stall %%] [--lockstep 0/1]\n");
        return 2;
    }
    int steps = 2, lockstep = 1, total = 8000;
    unsigned seed = 12345;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lat")) lat = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--stall")) stall = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lockstep")) lockstep = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = (unsigned)strtoul(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--total-steps")) total = atoi(argv[i + 1]);
    }
    if (bt_layout_load(&L, argv[1]))
        return 1;
    bt_cpu c;
    bt_rng rng;
    bt_model m;
    uint8_t *data = bt_dev_init(&c, &L, argv[1], argv[2], &m, &rng, seed, 4);
    c.threads = 1;
    memsize = L.memsize;
    rmem = (uint8_t *)malloc(memsize);
    memcpy(rmem, c.mem, memsize);

    top = new Vbittf_core2;
    top->rst = 1;
    for (int i = 0; i < 4; i++) tick();
    top->rst = 0;
    // CSR: seed, ctr
    top->csr_we = 1; top->csr_addr = 0; top->csr_wdata = c.csr[BT_CSR_SEED]; tick();
    top->csr_addr = 1; top->csr_wdata = c.csr[BT_CSR_CTR]; tick();
    top->csr_we = 0;

    long split = L.corpus_len - L.val_len;
    int B = L.B, Tn = L.T, N = B * Tn;
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = (uint8_t *)malloc(N), *y = (uint8_t *)malloc(N);
    uint32_t *r = (uint32_t *)malloc(sizeof(uint32_t) * B);
    int bad = 0;
    uint64_t ninstr = 0;
    printf("RTL tb: program %d instrukci, pamet %.1f MiB, lat %d, stall %d%%, lockstep %d\n", L.prog_n,
           memsize / 1048576.0, lat, stall, lockstep);
    for (int step = 1; step <= steps && !bad; step++) {
        if (step == total / 2 || step == total * 3 / 4 || step == total * 7 / 8)
            m.lr_shift--;
        bt_store(&c, bt_t(&L, "lr")->addr, 3, m.lr_shift);
        int64_t lrv = m.lr_shift;
        memcpy(rmem + bt_t(&L, "lr")->addr, &lrv, 8);
        // golden krok
        bt_rng_draw(&rng, B, r);
        for (int b = 0; b < B; b++) {
            long ix = Tn + (long)(r[b] % (uint32_t)(split - 1 - Tn));
            memcpy(x + (long)b * Tn, data + ix - Tn, Tn);
            memcpy(y + (long)b * Tn, data + ix - Tn + 1, Tn);
        }
        int64_t *lg;
        int *el;
        bt_forward(&m, w, x, 1, &lg, &el);
        bt_backward_update(&m, w, y, &rng, 0, step);
        // RTL krok
        uint64_t c0 = cycles;
        c.pc = 0;
        top->prog_base = (uint32_t)L.prog_addr;
        top->start = 1;
        tick();
        top->start = 0;
        int halted = 0;
        while (!halted) {
            tick();
            if (cycles - c0 > 4000000000ull) {
                fprintf(stderr, "RTL: casovy limit\n");
                return 1;
            }
            if (top->dbg_instr_done) {
                uint32_t pc = top->dbg_pc;
                ninstr++;
                if (lockstep) {
                    if (pc != c.pc) {
                        fprintf(stderr, "NESHODA pc: RTL %u ISA %u\n", pc, c.pc);
                        bad = 1;
                        break;
                    }
                    int rr = bt_step(&c, (uint32_t)L.prog_addr);
                    if (rr < 0) {
                        fprintf(stderr, "ISA chyba %s na pc %u\n", bt_err_name(-rr), pc);
                        bad = 1;
                        break;
                    }
                    if (compare(&c, pc)) {
                        bad = 1;
                        break;
                    }
                    if (rr == 1) halted = 1;
                }
            }
            if (top->done) {
                if (top->err) {
                    fprintf(stderr, "RTL chyba %d na pc %u\n", top->err, top->dbg_pc);
                    bad = 1;
                }
                halted = 1;
            }
        }
        if (bad) break;
        if (!lockstep) {
            int rr = bt_run(&c, (uint32_t)L.prog_addr, ~0ull);
            if (rr || memcmp(c.mem, rmem, memsize)) {
                fprintf(stderr, "NESHODA pameti po kroku (bez lockstepu)\n");
                bad = 1;
                break;
            }
        }
        // stav RTL vs golden
        bt_cpu rc = c;
        rc.mem = rmem;
        top->csr_addr = 1;
        top->eval();
        rc.csr[BT_CSR_CTR] = top->csr_rdata;
        uint32_t h1 = bt_dev_hash(&rc, &L, &m), h2 = bt_state_hash(&m, &rng);
        printf("krok %d RTL vs golden %s (%08x/%08x), %lu taktu, %lu instrukci, radku na portech %lu/%lu/%lu/%lu/%lu\n",
               step, h1 == h2 ? "SHODA" : "NESHODA", h1, h2, (unsigned long)(cycles - c0), (unsigned long)ninstr,
               (unsigned long)nreq[0], (unsigned long)nreq[1], (unsigned long)nreq[2], (unsigned long)nreq[3],
               (unsigned long)nreq[4]);
        if (h1 != h2) bad = 1;
        fflush(stdout);
    }
    printf(bad ? "VYSLEDEK: NESHODA\n" : "VYSLEDEK: SHODA\n");
    delete top;
    return bad;
}
