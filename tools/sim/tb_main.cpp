// Test PPU: program v fpga/mem/imem.mem provede
//   lw x3,0(x0); lw x4,4(x0); add x5,x3,x4; sw x5,16(x0); addi x0,x0,8;
//   ai.dot base1=x0 base2=x0 len=5; jal x0,0
// Ocekavani: dmem = {3,4,5,6,7} -> dot = 9+16+25+36+49 = 135.
// (Spatny LW/SW nebo zapis do x0 da jiny vysledek.)
// Pote druha operace pres MMIO (restart ze stavu "done"): base=8, len=2 -> 25+36 = 61.
#include "Vtb_ppu.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>

static Vtb_ppu *top;
static uint64_t cyc = 0;

static void tick() {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
    cyc++;
}

static void mmio_write(uint8_t a, uint32_t v) {
    top->mmio_we = 1; top->mmio_addr = a; top->mmio_wdata = v;
    tick();
    top->mmio_we = 0;
}

static uint32_t mmio_read(uint8_t a) {
    top->mmio_addr = a;
    tick();               // registrovane cteni
    tick();
    return top->mmio_rdata;
}

static bool wait_done(int max) {
    for (int i = 0; i < max; i++)
        if (mmio_read(0x04) == 2) return true;
    return false;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vtb_ppu;
    int fails = 0;

    top->mmio_we = 0; top->mmio_addr = 0; top->mmio_wdata = 0;
    top->rst = 1;
    for (int i = 0; i < 5; i++) tick();
    top->rst = 0;

    if (!wait_done(1000)) { printf("FAIL: akcelerator nedokoncil 1. operaci\n"); fails++; }
    uint32_t r1 = mmio_read(0x14);
    printf("1) CPU program: vysledek = %u (ocekavano 135), accel_result_out = %u\n", r1, top->accel_result_out);
    if (r1 != 135 || top->accel_result_out != 135) fails++;

    mmio_write(0x08, 8);
    mmio_write(0x0C, 8);
    mmio_write(0x10, 2);
    mmio_write(0x00, 1);
    if (!wait_done(1000)) { printf("FAIL: akcelerator nedokoncil 2. operaci\n"); fails++; }
    uint32_t r2 = mmio_read(0x14);
    printf("2) MMIO restart: vysledek = %u (ocekavano 61)\n", r2);
    if (r2 != 61) fails++;

    mmio_write(0x10, 0);
    mmio_write(0x00, 1);
    if (!wait_done(100)) { printf("FAIL: len=0 nedokoncil\n"); fails++; }
    uint32_t r3 = mmio_read(0x14);
    printf("3) len=0: vysledek = %u (ocekavano 0)\n", r3);
    if (r3 != 0) fails++;

    printf("%s (%llu taktu)\n", fails ? "PPU TEST SELHAL" : "PPU TEST OK", (unsigned long long)cyc);
    delete top;
    return fails ? 1 : 0;
}
