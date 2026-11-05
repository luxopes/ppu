// Testbench FPGA obalu BitTF (bittf_fpga_top): host pres AXI4-Lite nahraje registry, pamet je AXI4 model
// s nahodnou latenci a nahodnym ready, pamet jadra lezi od MEM_BASE (> 4 GB, test 64bit adres).
// Po kazdem kroku hash stavu proti golden modelu.
//
// tb_fpga <prefix> <korpus> [--steps 2] [--stall 20]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>

#include "Vbittf_fpga_top.h"
#include "verilated.h"

extern "C" {
#include "../isa/bt_layout.h"
}

#define LB 64
static Vbittf_fpga_top *top;
static uint8_t *mem;
static uint64_t memsize, cycles;
static const uint64_t MEM_BASE = 0x100000000ull;
static int stall = 20;
static uint32_t lfsr = 777;
static uint32_t rnd32() { lfsr ^= lfsr << 13; lfsr ^= lfsr >> 17; lfsr ^= lfsr << 5; return lfsr; }

struct rr { uint64_t t; uint8_t d[LB]; };
static std::deque<rr> rq;
static std::deque<uint64_t> bq;
static bool aw_pend = false, w_pend = false;
static uint64_t aw_addr;
static uint8_t w_data[LB];
static uint64_t w_strb;

static uint64_t off_of(uint64_t a)
{
    if (a < MEM_BASE || a - MEM_BASE + LB > memsize) {
        fprintf(stderr, "AXI: adresa mimo pamet 0x%lx\n", (unsigned long)a);
        exit(1);
    }
    return a - MEM_BASE;
}

static void tick()
{
    bool st = stall && (rnd32() % 100 < (uint32_t)stall);
    top->m_axi_arready = !st;
    top->m_axi_awready = !aw_pend && !(stall && rnd32() % 100 < (uint32_t)stall);
    top->m_axi_wready = !w_pend && !(stall && rnd32() % 100 < (uint32_t)stall);
    top->m_axi_rvalid = 0;
    if (!rq.empty() && rq.front().t <= cycles && top->m_axi_rready) {
        top->m_axi_rvalid = 1;
        memcpy((uint8_t *)&top->m_axi_rdata[0], rq.front().d, LB);
        rq.pop_front();
    }
    top->m_axi_bvalid = 0;
    if (!bq.empty() && bq.front() <= cycles) {
        top->m_axi_bvalid = 1;
        bq.pop_front();
    }
    top->clk = 0;
    top->eval();
    if (top->m_axi_arvalid && top->m_axi_arready) {
        rr r;
        r.t = cycles + 10 + rnd32() % 20;
        if (!rq.empty() && rq.back().t > r.t) r.t = rq.back().t;
        memcpy(r.d, mem + off_of(top->m_axi_araddr), LB);
        rq.push_back(r);
    }
    if (top->m_axi_awvalid && top->m_axi_awready) { aw_pend = true; aw_addr = top->m_axi_awaddr; }
    if (top->m_axi_wvalid && top->m_axi_wready) {
        w_pend = true;
        memcpy(w_data, (uint8_t *)&top->m_axi_wdata[0], LB);
        w_strb = top->m_axi_wstrb;
    }
    if (aw_pend && w_pend) {
        uint64_t o = off_of(aw_addr);
        for (int b = 0; b < LB; b++)
            if ((w_strb >> b) & 1) mem[o + b] = w_data[b];
        uint64_t t = cycles + 5 + rnd32() % 10;
        if (!bq.empty() && bq.back() > t) t = bq.back();
        bq.push_back(t);
        aw_pend = w_pend = false;
    }
    top->clk = 1;
    top->eval();
    cycles++;
}

static void ocl_write(uint32_t a, uint32_t v)
{
    top->s_axi_awaddr = a; top->s_axi_awvalid = 1;
    top->s_axi_wdata = v; top->s_axi_wstrb = 0xF; top->s_axi_wvalid = 1;
    top->s_axi_bready = 1;
    for (int i = 0; i < 100; i++) {
        tick();
        if (top->s_axi_awready) top->s_axi_awvalid = 0;
        if (top->s_axi_wready) top->s_axi_wvalid = 0;
        if (top->s_axi_bvalid) break;
    }
    tick();
    top->s_axi_awvalid = top->s_axi_wvalid = 0;
}

static uint32_t ocl_read(uint32_t a)
{
    top->s_axi_araddr = a; top->s_axi_arvalid = 1; top->s_axi_rready = 1;
    uint32_t v = 0;
    for (int i = 0; i < 100; i++) {
        tick();
        if (top->s_axi_arready) top->s_axi_arvalid = 0;
        if (top->s_axi_rvalid) { v = top->s_axi_rdata; break; }
    }
    tick();
    top->s_axi_arvalid = 0;
    return v;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    int steps = 2, total = 8000;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--stall")) stall = atoi(argv[i + 1]);
    }
    bt_layout L;
    if (bt_layout_load(&L, argv[1])) return 1;
    bt_cpu c;
    bt_rng rng;
    bt_model m;
    uint8_t *data = bt_dev_init(&c, &L, argv[1], argv[2], &m, &rng, 12345, 4);
    memsize = L.memsize;
    mem = (uint8_t *)malloc(memsize);
    memcpy(mem, c.mem, memsize);
    top = new Vbittf_fpga_top;
    top->rst_n = 0;
    for (int i = 0; i < 5; i++) tick();
    top->rst_n = 1;
    tick();
    uint32_t id = ocl_read(0x28);
    printf("FPGA tb: ID %08x, MEM_BASE 0x%lx, stall %d%%\n", id, (unsigned long)MEM_BASE, stall);
    ocl_write(0x20, (uint32_t)MEM_BASE);
    ocl_write(0x24, (uint32_t)(MEM_BASE >> 32));
    ocl_write(0x08, (uint32_t)L.prog_addr);
    ocl_write(0x10, c.csr[BT_CSR_SEED]);
    ocl_write(0x14, c.csr[BT_CSR_CTR]);
    long split = L.corpus_len - L.val_len;
    int B = L.B, Tn = L.T, N = B * Tn;
    bt_work *w = bt_work_new(&m, B);
    uint8_t *x = (uint8_t *)malloc(N), *y = (uint8_t *)malloc(N);
    uint32_t *r = (uint32_t *)malloc(sizeof(uint32_t) * B);
    int bad = id != 0x42544632;
    for (int step = 1; step <= steps && !bad; step++) {
        if (step == total / 2 || step == total * 3 / 4 || step == total * 7 / 8) m.lr_shift--;
        int64_t lrv = m.lr_shift;
        memcpy(mem + bt_t(&L, "lr")->addr, &lrv, 8);
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
        uint64_t c0 = cycles;
        ocl_write(0x00, 1);
        uint32_t s;
        do {
            for (int i = 0; i < 200; i++) tick();
            s = ocl_read(0x04);
        } while (!(s & 2));
        if (s >> 8 & 0xFF) {
            printf("chyba jadra %u\n", s >> 8 & 0xFF);
            bad = 1;
            break;
        }
        bt_cpu rc = c;
        rc.mem = mem;
        rc.csr[BT_CSR_CTR] = ocl_read(0x14);
        uint32_t h1 = bt_dev_hash(&rc, &L, &m), h2 = bt_state_hash(&m, &rng);
        printf("krok %d FPGA obal vs golden %s (%08x/%08x), %lu taktu, CYCLES %u\n", step, h1 == h2 ? "SHODA" : "NESHODA",
               h1, h2, (unsigned long)(cycles - c0), ocl_read(0x1C));
        if (h1 != h2) bad = 1;
    }
    printf(bad ? "VYSLEDEK: NESHODA\n" : "VYSLEDEK: SHODA\n");
    delete top;
    return bad;
}
