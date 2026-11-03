#!/bin/bash
# Minimal synthesis/place&route driver for open-source flow
# Requires: yosys, openroad, and a PDK + LEF/.lib files available locally

set -euo pipefail

WORKDIR=$(pwd)
TOP=ppu_top
VERILOG_DIR="${WORKDIR}/../../rtl"

echo "Running synthesis for ${TOP} (primitive driver)"

yosys -p "read_verilog ${VERILOG_DIR}/**/*.v; synth -top ${TOP}; write_verilog ${TOP}_syn.v"

# Note: next steps require LEF and .lib from PDK and are PDK-specific
echo "Place & Route: run OpenROAD scripts (requires PDK LEF/.lib)"
echo "Placeholder - adapt openroad TCL flow to your PDK and foundry rules"

echo "Generated: ${TOP}_syn.v"
