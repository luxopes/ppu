# Preklad na F2 instanci (FPGA Developer AMI, po `source sdk_setup.sh` v aws-fpga):  make
SDK_DIR ?= $(HOME)/aws-fpga/sdk
PPU := ../../..
CFLAGS = -O2 -std=gnu11 -Wall -fopenmp -I$(SDK_DIR)/userspace/include
SRC = bt_f2_run.c ../host/bt_host.c ../host/bt_host_main.c ../isa/bt_layout.c ../isa/bt_isa.c ../golden/bittf_ref.c

bt_f2_run: $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) -lfpga_mgmt -lm

clean:
	rm -f bt_f2_run
