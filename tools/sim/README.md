# Simulation scripts

Verilator simulace (Verilator ≥ 5, např. z OSS CAD Suite).

- `make run`: test původního PPU (CPU + dot-product akcelerátor). Program
  v `fpga/mem/imem.mem`:
  ```
  lw x3,0(x0); lw x4,4(x0); add x5,x3,x4; sw x5,16(x0); addi x0,x0,8
  ai.dot rs1=x0, rs2=x0, len=5   (R-format, opcode 1111011, funct7 = délka)
  jal x0,0
  ```
  Očekává se výsledek 135 (dmem = 3,4,5,6,7), pak restart přes MMIO (61)
  a délka 0 (0).
- `bitlm/`: BitLM jádro přes AXI4-Lite. `make run` (bitová shoda s referencí),
  `make train` (trénink v RTL).
