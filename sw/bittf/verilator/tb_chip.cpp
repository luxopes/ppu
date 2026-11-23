// Testbench celeho cipu BitTF mini (Verilator): jadro + radic QSPI + ridici SPI.
// PSRAM je emulovana na urovni vyvodu (QPI 0xEB/0x38) nad stejnou pameti jako v tb_bittf; cip se ovlada
// pres SPI jako z mikrokontroleru. Lockstep se simulatorem ISA po kazde instrukci + hash s golden modelem.
//
// tb_chip <prefix programu> <korpus.tok> [--steps 2] [--lockstep 1] [--seed 12345] [--spidiv 8]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Vbittf_chip.h"
#include "verilated.h"

extern "C" {
#include "../isa/bt_layout.h"
}

static bt_layout L;
static Vbittf_chip *top;
static uint8_t *rmem;          // obsah PSRAM
static uint64_t memsize;
static uint64_t cycles;
static int lat = 0, stall = 0, spidiv = 8;
static const int WAIT = 6;

// ---------------------------------------------------------------- model PSRAM (QPI)
enum { P_IDLE, P_CMD, P_ADDR, P_WDATA, P_RWAIT, P_RDATA };
static int pst = P_IDLE, pn = 0, pcs = 1, psck = 0;
static uint32_t pcmd = 0, paddr = 0, pwb = 0;
static int pnib_lo = 0;
static uint64_t nreads = 0, nwrites = 0;

static void psram_edge()
{
    int cs = top->psram_cs_n, sck = top->psram_sck;
    if (!top->psram_ctl_oe) { pcs = 1; psck = 0; pst = P_IDLE; return; }
    if (cs) { pst = P_IDLE; pcs = 1; psck = sck; return; }
    if (pcs) { pst = P_CMD; pn = 0; pcmd = 0; paddr = 0; pnib_lo = 0; }
    pcs = 0;
    int nib_in = top->psram_io_oe ? (top->psram_io_o & 15) : 0;
    if (sck && !psck) {                     // nabezna hrana: PSRAM vzorkuje
        switch (pst) {
        case P_CMD:
            pcmd = (pcmd << 4) | nib_in;
            if (++pn == 2) {
                if (pcmd != 0xEB && pcmd != 0x38) { fprintf(stderr, "PSRAM: neznamy prikaz %02x\n", pcmd); exit(1); }
                pst = P_ADDR; pn = 0;
            }
            break;
        case P_ADDR:
            paddr = (paddr << 4) | nib_in;
            if (++pn == 6) { pst = pcmd == 0x38 ? P_WDATA : P_RWAIT; pn = 0; pnib_lo = 0; }
            break;
        case P_WDATA:
            if (!pnib_lo) { pwb = nib_in << 4; pnib_lo = 1; }
            else {
                if (paddr >= memsize) { fprintf(stderr, "PSRAM: zapis mimo 0x%x\n", paddr); exit(1); }
                rmem[paddr++] = (uint8_t)(pwb | nib_in); pnib_lo = 0; nwrites++;
            }
            break;
        case P_RWAIT:
            if (++pn == WAIT) { pst = P_RDATA; pnib_lo = 0; }
            break;
        default: break;
        }
    } else if (!sck && psck && pst == P_RDATA) {   // sestupna hrana: PSRAM vystavi nibl
        if (paddr >= memsize) { fprintf(stderr, "PSRAM: cteni mimo 0x%x\n", paddr); exit(1); }
        uint8_t b = rmem[paddr];
        top->psram_io_i = pnib_lo ? (b & 15) : (b >> 4);
        if (pnib_lo) { paddr++; nreads++; }
        pnib_lo ^= 1;
    }
    psck = sck;
}
static int psram_writing() { return !pcs && (pst == P_CMD || pst == P_ADDR || pst == P_WDATA); }

static void tick()
{
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    psram_edge();
    cycles++;
}

// ---------------------------------------------------------------- ridici SPI (rezim 0, MSB prvni)
static uint32_t spi(uint8_t cmd, uint32_t data)
{
    uint64_t out = ((uint64_t)cmd << 32) | data, in = 0;
    top->spi_cs_n = 0;
    for (int i = 0; i < spidiv; i++) tick();
    for (int b = 39; b >= 0; b--) {
        top->spi_mosi = (out >> b) & 1;
        for (int i = 0; i < spidiv; i++) tick();
        top->spi_sck = 1;
        in = (in << 1) | top->spi_miso;
        for (int i = 0; i < spidiv; i++) tick();
        top->spi_sck = 0;
    }
    for (int i = 0; i < spidiv; i++) tick();
    top->spi_cs_n = 1;
    for (int i = 0; i < spidiv; i++) tick();
    return (uint32_t)in;
}
static void spi_wr(int reg, uint32_t v) { spi(0x80 | reg, v); }
static uint32_t spi_rd(int reg) { return spi(reg, 0); }

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
    if (top->dbg_ctr != c->csr[BT_CSR_CTR]) {
        fprintf(stderr, "NESHODA CTR po pc %u: ISA %u RTL %u\n", pc, c->csr[BT_CSR_CTR], top->dbg_ctr);
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
        else if (!strcmp(argv[i], "--spidiv")) spidiv = atoi(argv[i + 1]);
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

    top = new Vbittf_chip;
    top->spi_cs_n = 1; top->spi_sck = 0; top->host_own = 0;
    top->rst_n = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1;
    for (int i = 0; i < 80; i++) tick();     // nulovani registrove sady (64 taktu)
    uint32_t id = spi_rd(7);
    if (id != 0x42544631) { fprintf(stderr, "SPI: spatna identifikace %08x\n", id); return 1; }
    // CSR pres SPI: seed, ctr, adresa programu
    spi_wr(0, c.csr[BT_CSR_SEED]);
    spi_wr(1, c.csr[BT_CSR_CTR]);
    spi_wr(4, (uint32_t)L.prog_addr);
    if (spi_rd(1) != c.csr[BT_CSR_CTR] || spi_rd(4) != (uint32_t)L.prog_addr) { fprintf(stderr, "SPI: zpetne cteni\n"); return 1; }

    long split = L.corpus_len - L.val_len;
    int B = L.B, Tn = L.T, N = B * Tn;
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = (uint8_t *)malloc(N), *y = (uint8_t *)malloc(N);
    uint32_t *r = (uint32_t *)malloc(sizeof(uint32_t) * B);
    int bad = 0;
    uint64_t ninstr = 0;
    printf("tb_chip: program %d instrukci, pamet %.1f MiB, SPI clk/%d, lockstep %d\n", L.prog_n,
           memsize / 1048576.0, 2 * spidiv, lockstep);
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
        spi_wr(5, 1);                        // start
        int halted = 0;
        while (!halted) {
            tick();
            if (cycles - c0 > 4000000000ull) {
                fprintf(stderr, "RTL: casovy limit\n");
                return 1;
            }
            if (top->dbg_instr_done) {
                uint32_t pc = top->dbg_pc;
                // posledni zapis instrukce muze byt jeste na sbernici PSRAM: dobehnout
                while (psram_writing()) tick();
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
            if (top->done_o) {
                while (psram_writing()) tick();
                uint32_t st = spi_rd(5);
                if (st & 0xFF) {
                    fprintf(stderr, "RTL chyba %u na pc %u\n", st & 0xFF, top->dbg_pc);
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
        rc.csr[BT_CSR_CTR] = spi_rd(1);
        uint32_t h1 = bt_dev_hash(&rc, &L, &m), h2 = bt_state_hash(&m, &rng);
        printf("krok %d RTL vs golden %s (%08x/%08x), %lu taktu, %lu instrukci, PSRAM %lu cteni / %lu zapisu bajtu\n",
               step, h1 == h2 ? "SHODA" : "NESHODA", h1, h2, (unsigned long)(cycles - c0), (unsigned long)ninstr,
               (unsigned long)nreads, (unsigned long)nwrites);
        if (h1 != h2) bad = 1;
        fflush(stdout);
    }
    printf(bad ? "VYSLEDEK: NESHODA\n" : "VYSLEDEK: SHODA\n");
    delete top;
    return bad;
}
