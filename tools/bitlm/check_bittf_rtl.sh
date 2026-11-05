#!/bin/sh
# Preklad a beh RTL testbenche BitTF (Verilator): lockstep RTL vs simulator ISA + hash vs golden.
# Pouziti: tools/bitlm/check_bittf_rtl.sh [korpus] ; promenne CORE (1 = sekvencni v1, 2 = vykonove v2,
#          fpga = obal s AXI4, cl = cesta jako na F2: host nahrava pres PCIS + prepinac pameti),
#          STEPS, LAT, STALL, CFG (argumenty prekladace)
set -e
CORPUS=${1:-data/cz_mix.tok}
export PATH=$HOME/.local/opt/oss-cad-suite/bin:$PATH
CORE=${CORE:-2}
B=build/vl$CORE
mkdir -p $B build/isa
gcc -O2 -fopenmp -c -o $B/bt_layout.o sw/bittf/isa/bt_layout.c
gcc -O2 -fopenmp -c -o $B/bt_isa.o sw/bittf/isa/bt_isa.c
gcc -O2 -fopenmp -c -o $B/bittf_ref.o sw/bittf/golden/bittf_ref.c
gcc -O2 -c -o $B/bt_host.o sw/bittf/host/bt_host.c
gcc -O2 -c -o $B/bt_host_main.o sw/bittf/host/bt_host_main.c
ar rcs $B/libbt.a $B/bt_layout.o $B/bt_isa.o $B/bittf_ref.o $B/bt_host.o $B/bt_host_main.o
if [ "$CORE" = 1 ]; then
    TOP=bittf_core; SRC="rtl/bittf/bittf_core.v rtl/bittf/bittf_alu.v"; TB=tb_bittf
elif [ "$CORE" = fpga ] || [ "$CORE" = cl ]; then
    if [ "$CORE" = fpga ]; then TOP=bittf_fpga_top; TB=tb_fpga; else TOP=bittf_cl_sim; TB=tb_cl; fi
    SRC="fpga/sim/bittf_cl_sim.v fpga/axi/axi_own_mux.v fpga/top/bittf_fpga_top.v fpga/axi/axi_lite_mmio.v rtl/bittf/bittf_axi.v rtl/bittf/bittf_core2.v rtl/bittf/bittf_stream.v rtl/bittf/bittf_wstream.v rtl/bittf/bittf_mm.v rtl/bittf/bittf_lane.v rtl/bittf/bittf_alu.v"
else
    TOP=bittf_core2; TB=tb_bittf2
    SRC="rtl/bittf/bittf_core2.v rtl/bittf/bittf_stream.v rtl/bittf/bittf_wstream.v rtl/bittf/bittf_mm.v rtl/bittf/bittf_lane.v rtl/bittf/bittf_alu.v"
fi
verilator --cc --exe --build -O2 -j ${JOBS:-4} -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-UNOPTFLAT -Irtl/bittf \
    --top-module $TOP -Mdir $B -CFLAGS "-O2" $SRC sw/bittf/verilator/$TB.cpp \
    -LDFLAGS "$PWD/$B/libbt.a -lm -fopenmp" -o $TB > $B/build.log 2>&1 || { tail -30 $B/build.log; exit 1; }
L=$(stat -c %s "$CORPUS")
CFG=${CFG:---d 32 --layers 1 --heads 2 --ff 64 --ctx 16 --batch 2}
python3 tools/bitlm/bittf_compile.py $CFG --corpus-len $L -o build/isa/rtl_cfg > /dev/null
if [ "$CORE" = cl ]; then
    $B/$TB build/isa/rtl_cfg "$CORPUS" --steps ${STEPS:-2} --check ${CHECK:-1} --stall ${STALL:-20}
elif [ "$CORE" = fpga ]; then
    $B/$TB build/isa/rtl_cfg "$CORPUS" --steps ${STEPS:-2} --stall ${STALL:-20}
else
    $B/$TB build/isa/rtl_cfg "$CORPUS" --steps ${STEPS:-2} --lat ${LAT:-2} --stall ${STALL:-0} --lockstep ${LOCKSTEP:-1}
fi
