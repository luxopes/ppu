# Verilator simulace BitLM jadra + bitove presne porovnani s referencnim modelem.
#   make run                       # maly model (148k parametru, kontext 8), overeni 500 kroku/davku
#   make run CONFIG=big            # velky model (1,12M parametru, viceurovnovy kontext 128)
#   make train [CONFIG=big]        # trenink v RTL na celem korpusu + porovnani s referenci
#   CORPUS=... VAL_LEN=...         # jiny korpus / delka validace (train)
VERILATOR ?= verilator
ROOT = ../../..
TOP = bitlm_fpga_top
STEPS ?= 500
TRAIN_STEPS ?= 50000
CONFIG ?= small
P_MS = 0
ifeq ($(CONFIG),big)
  # viceurovnovy kontext 128 znaku, E=32, H=2048 (1,12M parametru)
  P_N = 7
  P_E = 5
  P_H = 11
  P_MS = 1
  CORPUS ?= $(ROOT)/data/cz_mix.tok
  VAL_LEN ?= 11152
else
  P_N = 3
  P_E = 5
  P_H = 9
  CORPUS ?= $(ROOT)/data/rur.tok
  VAL_LEN ?= 0
endif
OBJ = obj_$(CONFIG)
RTL = \
    $(ROOT)/fpga/top/bitlm_fpga_top.v \
    $(ROOT)/fpga/axi/axi_lite_mmio.v \
    $(ROOT)/rtl/accelerator/bitlm_core.v
CSRC = tb_bitlm.cpp $(ROOT)/sw/bitlm/host/bitlm_hw.c $(ROOT)/sw/bitlm/golden/bitlm_ref.c
DEFS = -DBL_LOG2_N=$(P_N) -DBL_LOG2_E=$(P_E) -DBL_LOG2_H=$(P_H) $(if $(filter 1,$(P_MS)),-DBL_MULTISCALE)

all: $(OBJ)/V$(TOP)

$(OBJ)/V$(TOP): $(RTL) $(CSRC) $(ROOT)/rtl/accelerator/bitlm_exp_lut.vh $(ROOT)/sw/bitlm/golden/bitlm_ref.h $(ROOT)/sw/bitlm/host/bitlm_hw.h
	$(VERILATOR) -Wall -Wno-fatal -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
	    -GLOG2_N=$(P_N) -GLOG2_E=$(P_E) -GLOG2_H=$(P_H) -GMULTISCALE=$(P_MS) \
	    -I$(ROOT)/rtl/accelerator --cc $(RTL) --top-module $(TOP) --Mdir $(OBJ) \
	    --exe $(CSRC) -CFLAGS "-O2 $(DEFS) -I$(abspath $(ROOT))/sw/bitlm/golden" -O3 --build -j 4

run: $(OBJ)/V$(TOP)
	./$(OBJ)/V$(TOP) $(CORPUS) $(STEPS)

train: $(OBJ)/V$(TOP)
	./$(OBJ)/V$(TOP) $(CORPUS) train $(TRAIN_STEPS) $(VAL_LEN)

clean:
	rm -rf obj_small obj_big obj_dir

.PHONY: all run train clean
