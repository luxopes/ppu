#!/bin/bash
# Build CL BitTF pro AWS F2 na build instanci (FPGA Developer AMI, Vivado z AMI).
# Pouziti (z korene repozitare PPU):  bash fpga/aws_f2/build_cl.sh [dalsi volby aws_build_dcp_from_cl.py]
# Vysledek: $AWS_FPGA/hdk/cl/examples/cl_bittf/build/checkpoints/*.Developer_CL.tar (pro create-fpga-image)
set -e
PPU=$(cd "$(dirname "$0")/../.." && pwd)
AWS_FPGA=${AWS_FPGA:-$HOME/aws-fpga}
if [ ! -d "$AWS_FPGA" ]; then
    git clone --depth 1 https://github.com/aws/aws-fpga.git "$AWS_FPGA"
fi
cd "$AWS_FPGA"
source hdk_setup.sh
CL=$AWS_FPGA/hdk/cl/examples/cl_bittf
rm -rf "$CL"
cp -r "$PPU/fpga/aws_f2/cl_bittf" "$CL"
# RTL do jednoho adresare design/ (AWS flow kopiruje design/*.{v,sv,vh})
cp "$PPU"/rtl/bittf/bittf_core2.v "$PPU"/rtl/bittf/bittf_stream.v "$PPU"/rtl/bittf/bittf_wstream.v \
   "$PPU"/rtl/bittf/bittf_mm.v "$PPU"/rtl/bittf/bittf_lane.v "$PPU"/rtl/bittf/bittf_alu.v \
   "$PPU"/rtl/bittf/bittf_axi.v "$PPU"/rtl/bittf/*.vh \
   "$PPU"/fpga/top/bittf_fpga_top.v "$PPU"/fpga/axi/axi_lite_mmio.v "$PPU"/fpga/axi/axi_own_mux.v \
   "$CL/design/"
export CL_DIR=$CL
cd "$CL/build/scripts"
for f in aws_build_dcp_from_cl.py build_all.tcl build_level_1_cl.tcl encrypt.tcl; do
    ln -sf "$HDK_SHELL_DIR/build/scripts/$f" .
done
./aws_build_dcp_from_cl.py -c cl_bittf --mode small_shell "$@"
ls -la "$CL/build/checkpoints/"*.tar
grep -h -A12 "Design Timing Summary" "$CL"/build/reports/*timing_summary*route* 2>/dev/null | head -20 || true
