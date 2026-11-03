// Verilator testbench BitLM: bitlm_fpga_top pres AXI4-Lite (stejne jako F2 OCL).
// Spousti bitlm_verify() - bitove presne porovnani HW s referencnim modelem.
// Pouziti: Vbitlm_fpga_top <korpus.tok> [kroku_na_davku]
//          Vbitlm_fpga_top <korpus.tok> train <kroku> [delka_validace]   (trenink v RTL)
#include "Vbitlm_fpga_top.h"
#include "verilated.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "../../../sw/bitlm/host/bitlm_hw.h"

static Vbitlm_fpga_top *top;
static uint64_t cyc = 0;

static void tick() {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
    cyc++;
    if (cyc > 4000000000ull) { fprintf(stderr, "timeout simulace\n"); exit(2); }
}

// AXI4-Lite master: AW a W soucasne, pak B
static void axi_wr(void *, uint32_t addr, uint32_t val) {
    top->s_axi_awaddr = addr; top->s_axi_awvalid = 1;
    top->s_axi_wdata = val; top->s_axi_wstrb = 0xF; top->s_axi_wvalid = 1;
    top->s_axi_bready = 1;
    bool aw = false, w = false;
    for (int i = 0; i < 1000 && !(aw && w); i++) {
        top->clk = 0; top->eval();
        bool awh = top->s_axi_awvalid && top->s_axi_awready;
        bool wh = top->s_axi_wvalid && top->s_axi_wready;
        top->clk = 1; top->eval(); cyc++;
        if (awh) { aw = true; top->s_axi_awvalid = 0; }
        if (wh) { w = true; top->s_axi_wvalid = 0; }
    }
    for (int i = 0; i < 1000; i++) {
        top->clk = 0; top->eval();
        bool bh = top->s_axi_bvalid && top->s_axi_bready;
        top->clk = 1; top->eval(); cyc++;
        if (bh) break;
    }
    top->s_axi_bready = 0;
}

static uint32_t axi_rd(void *, uint32_t addr) {
    top->s_axi_araddr = addr; top->s_axi_arvalid = 1; top->s_axi_rready = 1;
    for (int i = 0; i < 1000; i++) {
        top->clk = 0; top->eval();
        bool arh = top->s_axi_arvalid && top->s_axi_arready;
        top->clk = 1; top->eval(); cyc++;
        if (arh) { top->s_axi_arvalid = 0; break; }
    }
    uint32_t v = 0xFFFFFFFF;
    for (int i = 0; i < 1000; i++) {
        top->clk = 0; top->eval();
        bool rh = top->s_axi_rvalid && top->s_axi_rready;
        uint32_t d = top->s_axi_rdata;
        top->clk = 1; top->eval(); cyc++;
        if (rh) { v = d; break; }
    }
    top->s_axi_rready = 0;
    return v;
}

// cekani na dokonceni: misto dotazovani pres AXI jen pousti takty
static void idle(void *) { for (int i = 0; i < 64; i++) tick(); }

static double val_ce(const bl_model *m, const uint8_t *c, uint32_t from, uint32_t to, double *acc) {
    bl_fwd f;
    double ce = 0;
    uint32_t ok = 0;
    for (uint32_t pos = from; pos < to; pos++) {
        bl_forward(m, c + pos - BL_N, &f);
        uint32_t p = f.p[c[pos]];
        ce += -log2((p ? p : 1) / 65536.0);
        ok += f.argmax == c[pos];
    }
    *acc = (double)ok / (to - from);
    return ce / (to - from);
}

// Trenink v RTL na celem korpusu (stejne nastaveni jako referencni beh),
// prubezne vyhodnoceni z vah prectenych z HW a porovnani s referenci.
static int rtl_train(bitlm_dev *d, const uint8_t *c, uint32_t len, uint32_t steps, uint32_t val_len) {
    uint32_t split = val_len ? len - val_len : len * 9 / 10, batch = 10000;
    static bl_model ref, hw;
    bl_hparams hp = bl_default_hparams();
    bl_init(&ref, 12345, hp);
    bitlm_set_hparams(d, &hp);
    bitlm_load_corpus(d, c, split);
    if (bitlm_init(d, 12345) != BL_ST_DONE) return 1;
    int fails = 0;
    printf("%8s %10s %9s %10s %12s\n", "krok", "val_CE", "val_acc", "takty/krok", "shoda s ref");
    for (uint32_t done = 0; done < steps; done += batch) {
        uint32_t n = steps - done < batch ? steps - done : batch;
        if (bitlm_train(d, split, n) != BL_ST_DONE) return 1;
        for (uint32_t s = 0; s < n; s++) bl_train_random(&ref, c, split);
        bitlm_stats st;
        bitlm_read_stats(d, &st);
        bitlm_read_model(d, &hw);
        hw.hp = hp;
        int same = !memcmp(hw.w1, ref.w1, sizeof(hw.w1)) && !memcmp(hw.w2, ref.w2, sizeof(hw.w2)) &&
                   !memcmp(hw.emb, ref.emb, sizeof(hw.emb)) && hw.rng == ref.rng;
        fails += !same;
        double acc, ce = val_ce(&hw, c, split, len, &acc);
        printf("%8u %10.4f %9.4f %10.0f %12s\n", done + n, ce, acc, (double)st.cycles / n, same ? "ANO" : "NE");
        fflush(stdout);
    }
    uint8_t ctx[BL_N], out[300];
    memcpy(ctx, c + split, BL_N);
    bitlm_generate(d, ctx, sizeof(out), out);
    printf("--- generovano RTL (po %u krocich):\n", steps);
    for (unsigned i = 0; i < sizeof(out); i++) putchar(BL_VOCAB[out[i]]);
    printf("\n---\n%s\n", fails ? "RTL TRENINK: NESHODA S REFERENCI" : "RTL TRENINK: bitove shodny s referenci");
    return fails;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) { fprintf(stderr, "pouziti: %s korpus.tok [kroku]\n", argv[0]); return 2; }
    bool train_mode = argc > 3 && !strcmp(argv[2], "train");
    uint32_t steps = train_mode ? (uint32_t)strtoul(argv[3], 0, 0)
                                : (argc > 2 ? (uint32_t)strtoul(argv[2], 0, 0) : 500);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    std::vector<uint8_t> c(BL_CORPUS_MAX);
    uint32_t len = (uint32_t)fread(c.data(), 1, c.size(), f);
    fclose(f);
    // pro rychlost overeni staci cast korpusu (trenink pouziva cely)
    if (!train_mode && len > 20000) len = 20000;

    top = new Vbitlm_fpga_top;
    top->rst_n = 0;
    top->s_axi_awvalid = top->s_axi_wvalid = top->s_axi_bready = 0;
    top->s_axi_arvalid = top->s_axi_rready = 0;
    for (int i = 0; i < 10; i++) tick();
    top->rst_n = 1;
    for (int i = 0; i < 5; i++) tick();

    bitlm_dev d = { axi_rd, axi_wr, nullptr, idle };
    uint32_t val_len = (train_mode && argc > 4) ? (uint32_t)strtoul(argv[4], 0, 0) : 0;
    int fails = train_mode ? rtl_train(&d, c.data(), len, steps, val_len)
                           : bitlm_verify(&d, c.data(), len, steps, 1);
    printf("simulovano %llu taktu\n", (unsigned long long)cyc);
    delete top;
    return fails ? 1 : 0;
}
