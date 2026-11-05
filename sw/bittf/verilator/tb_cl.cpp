// Testbench cesty jako na F2 (bittf_cl_sim = bittf_fpga_top + axi_own_mux) s OPRAVDOVYM ovladacem hosta
// (sw/bittf/host, stejny kod jako na F2): host pise/cte pamet pres AXI4 bursty (PCIS, az 64 beatu, nahodna ID,
// castecne masky), pamet je AXI4 model s bursty, nahodnou latenci a ready. Navic: kdyz jadro prevezme
// pamet, host zkusi cist - odpoved smi prijit az po skonceni jadra.
//
// tb_cl <prefix> <korpus> [volby bt_host_main] [--stall 20]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

#include "Vbittf_cl_sim.h"
#include "verilated.h"

extern "C" {
#include "../host/bt_host.h"
}

#define LB 64
static Vbittf_cl_sim *top;
static uint8_t *mem;
static uint64_t memsize, cycles;
static int stall = 20;
static uint32_t lfsr = 4242;
static uint32_t rnd32() { lfsr ^= lfsr << 13; lfsr ^= lfsr >> 17; lfsr ^= lfsr << 5; return lfsr; }
static bool coin() { return stall && rnd32() % 100 < (uint32_t)stall; }

// --- model pameti (DDR): AW/W fronty, odpovedi v poradi
struct burst { uint64_t a; int len, id; };
static std::deque<burst> awq;
struct wbeat { uint8_t d[LB]; uint64_t strb; bool last; };
static std::deque<wbeat> wq;
struct rbeat { uint64_t t; uint8_t d[LB]; int id; bool last; };
static std::deque<rbeat> rq;
struct bresp { uint64_t t; int id; };
static std::deque<bresp> bq;

static uint64_t chk(uint64_t a)
{
    if (a + LB > memsize) {
        fprintf(stderr, "DDR: adresa mimo pamet 0x%lx\n", (unsigned long)a);
        exit(1);
    }
    return a;
}

// --- host (PCIS): fronty pozadavku a prijatych odpovedi
static std::deque<burst> h_aw, h_ar;
static std::deque<wbeat> h_w;
static int h_bpend = 0, h_rpend = 0;
static std::vector<uint8_t> h_rbuf;     // prijata data cteni v poradi
static int h_rid_exp = -1;
static bool h_id_bad = false;
static std::deque<int> h_rids;

static void tick()
{
    // pamet: vstupy pro tento takt
    top->m_awready = !coin();
    top->m_wready = !coin();
    top->m_arready = !coin();
    top->m_rvalid = 0;
    if (!rq.empty() && rq.front().t <= cycles) {
        top->m_rvalid = 1;
        memcpy((uint8_t *)&top->m_rdata[0], rq.front().d, LB);
        top->m_rid = rq.front().id;
        top->m_rlast = rq.front().last;
    }
    top->m_bvalid = 0;
    if (!bq.empty() && bq.front().t <= cycles) {
        top->m_bvalid = 1;
        top->m_bid = bq.front().id;
    }
    // host: vystupy pro tento takt
    top->h_awvalid = !h_aw.empty();
    if (!h_aw.empty()) { top->h_awaddr = h_aw.front().a; top->h_awlen = h_aw.front().len - 1; top->h_awid = h_aw.front().id; }
    top->h_wvalid = !h_w.empty();
    if (!h_w.empty()) {
        memcpy((uint8_t *)&top->h_wdata[0], h_w.front().d, LB);
        top->h_wstrb = h_w.front().strb;
        top->h_wlast = h_w.front().last;
    }
    top->h_arvalid = !h_ar.empty();
    if (!h_ar.empty()) { top->h_araddr = h_ar.front().a; top->h_arlen = h_ar.front().len - 1; top->h_arid = h_ar.front().id; }
    top->h_bready = !coin();
    top->h_rready = !coin();

    top->clk = 0;
    top->eval();
    // handshaky (vzorkovano pred hranou)
    if (top->m_awvalid && top->m_awready) awq.push_back({top->m_awaddr, top->m_awlen + 1, (int)top->m_awid});
    if (top->m_wvalid && top->m_wready) {
        wbeat b;
        memcpy(b.d, (uint8_t *)&top->m_wdata[0], LB);
        b.strb = top->m_wstrb;
        b.last = top->m_wlast;
        wq.push_back(b);
    }
    if (top->m_arvalid && top->m_arready) {
        uint64_t a = top->m_araddr;
        int n = top->m_arlen + 1, id = top->m_arid;
        uint64_t t = cycles + 10 + rnd32() % 20;
        if (!rq.empty() && rq.back().t > t) t = rq.back().t;
        for (int i = 0; i < n; i++) {
            rbeat r;
            r.t = t + i;
            memcpy(r.d, mem + chk(a + (uint64_t)i * LB), LB);
            r.id = id;
            r.last = i == n - 1;
            rq.push_back(r);
        }
    }
    if (top->m_rvalid && top->m_rready) rq.pop_front();
    if (top->m_bvalid && top->m_bready) bq.pop_front();
    // zapis do pameti, kdyz je cely burst (adresa i vsechny beaty)
    if (!awq.empty() && (int)wq.size() >= awq.front().len) {
        burst b = awq.front();
        awq.pop_front();
        for (int i = 0; i < b.len; i++) {
            wbeat w = wq.front();
            wq.pop_front();
            if (w.last != (i == b.len - 1)) { fprintf(stderr, "DDR: spatne WLAST\n"); exit(1); }
            uint64_t o = chk(b.a + (uint64_t)i * LB);
            for (int k = 0; k < LB; k++)
                if ((w.strb >> k) & 1) mem[o + k] = w.d[k];
        }
        uint64_t t = cycles + 5 + rnd32() % 10;
        if (!bq.empty() && bq.back().t > t) t = bq.back().t;
        bq.push_back({t, b.id});
    }
    // host strana
    if (top->h_awvalid && top->h_awready) { h_aw.pop_front(); }
    if (top->h_wvalid && top->h_wready) h_w.pop_front();
    if (top->h_arvalid && top->h_arready) { h_ar.pop_front(); }
    if (top->h_bvalid && top->h_bready) h_bpend--;
    if (top->h_rvalid && top->h_rready) {
        h_rbuf.insert(h_rbuf.end(), (uint8_t *)&top->h_rdata[0], (uint8_t *)&top->h_rdata[0] + LB);
        if (h_rids.empty() || h_rids.front() != (int)top->h_rid) h_id_bad = true;
        if (top->h_rlast) { h_rids.pop_front(); h_rpend--; }
    }
    top->clk = 1;
    top->eval();
    cycles++;
}

// host zapise data (adresa i delka libovolne, castecne radky pres strobe), bursty po 4 KB
static void host_write(uint64_t a, const uint8_t *d, uint64_t n)
{
    uint64_t end = a + n, la = a & ~(uint64_t)(LB - 1);
    while (la < end) {
        uint64_t bend = (la & ~(uint64_t)4095) + 4096;
        int len = 0;
        uint64_t l0 = la;
        while (la < end && la < bend) {
            wbeat w;
            memset(w.d, 0, LB);
            w.strb = 0;
            for (int k = 0; k < LB; k++)
                if (la + k >= a && la + k < end) { w.d[k] = d[la + k - a]; w.strb |= 1ull << k; }
            len++;
            la += LB;
            w.last = false;
            h_w.push_back(w);
        }
        h_w.back().last = true;
        h_aw.push_back({l0, len, (int)(rnd32() & 0xFFFF)});
        h_bpend++;
    }
    while (h_bpend || !h_aw.empty() || !h_w.empty()) tick();
}

static void host_read_issue(uint64_t a, uint64_t n)
{
    uint64_t end = a + n, la = a & ~(uint64_t)(LB - 1);
    while (la < end) {
        uint64_t bend = (la & ~(uint64_t)4095) + 4096;
        int len = 0;
        uint64_t l0 = la;
        while (la < end && la < bend) { len++; la += LB; }
        int id = rnd32() & 0xFFFF;
        h_ar.push_back({l0, len, id});
        h_rids.push_back(id);
        h_rpend++;
    }
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


// --- zarizeni pro ovladac hosta
static bool own_tested = false, own_bad = false;
static void d_wr32(void *, uint32_t a, uint32_t v) { ocl_write(a, v); }
static uint32_t d_rd32(void *, uint32_t a) { return ocl_read(a); }
static void d_memwr(void *, uint64_t a, const void *s, uint64_t n) { host_write(a, (const uint8_t *)s, n); }
static void d_memrd(void *, uint64_t a, void *dst, uint64_t n)
{
    h_rbuf.clear();
    host_read_issue(a, n);
    while (h_rpend) tick();
    if (h_id_bad) { fprintf(stderr, "spatna RID\n"); exit(1); }
    memcpy(dst, h_rbuf.data() + (a & 63), n);
}
static void d_idle(void *)
{
    if (!own_tested && top->own_core) {
        own_tested = true;
        h_rbuf.clear();
        host_read_issue(0, 64);
        while (h_rpend) {
            tick();
            if (top->own_core && !h_rbuf.empty()) own_bad = true;
        }
        printf("prepinac: cteni hosta behem behu jadra %s\n", own_bad ? "PROSLO (chyba)" : "pockalo na konec");
    }
    for (int i = 0; i < 200; i++) tick();
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        if (i + 1 < argc && !strcmp(argv[i], "--stall")) { stall = atoi(argv[++i]); continue; }
        args.push_back(argv[i]);
    }
    bt_layout L;
    if (argc < 2 || bt_layout_load(&L, argv[1])) return 1;
    memsize = (L.memsize + 4095) & ~(uint64_t)4095;
    mem = (uint8_t *)malloc(memsize);
    for (uint64_t i = 0; i < memsize; i++) mem[i] = rnd32();   // smeti: obraz musi prijit od hosta
    top = new Vbittf_cl_sim;
    top->rst_n = 0;
    for (int i = 0; i < 5; i++) tick();
    top->rst_n = 1;
    tick();
    printf("CL tb: stall %d%%\n", stall);
    bt_dev d = {nullptr, d_wr32, d_rd32, d_memwr, d_memrd, d_idle};
    int rc = bt_host_main(&d, (int)args.size(), args.data());
    if (!own_tested || own_bad) { printf("prepinac pameti netestovan nebo chybny\n"); rc = 1; }
    printf(rc ? "TB: NESHODA\n" : "TB: SHODA\n");
    delete top;
    return rc;
}
