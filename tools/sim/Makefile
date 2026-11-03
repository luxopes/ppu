# Simulace puvodniho PPU (CPU + dot-product akcelerator) ve Verilatoru.
VERILATOR ?= verilator
TOP = tb_ppu
RTL = \
    ../../rtl/top/ppu_top.v \
    ../../rtl/cpu/core.v \
    ../../rtl/cpu/alu.v \
    ../../rtl/cpu/regfile.v \
    ../../rtl/accelerator/ai_accel.v \
    ../../fpga/sim/tb_ppu.v

all: sim

sim: obj_dir/V$(TOP)

obj_dir/V$(TOP): $(RTL) tb_main.cpp
	$(VERILATOR) -Wall -Wno-fatal -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME --cc $(RTL) --exe tb_main.cpp --top-module $(TOP) --build -j 0

run: obj_dir/V$(TOP)
	./obj_dir/V$(TOP)

clean:
	rm -rf obj_dir

.PHONY: all sim run clean
