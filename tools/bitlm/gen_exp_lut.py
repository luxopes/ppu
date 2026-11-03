#!/usr/bin/env python3
"""Generuje tabulku exp(-k/T) v Q16 pro softmax BitLM.

Stejna celociselna tabulka se pouziva v C referencnim modelu
(sw/bitlm/golden/bitlm_exp_lut.h) i v RTL (rtl/accelerator/bitlm_exp_lut.vh),
takze HW a SW jsou bitove shodne bez zavislosti na plovouci carce.
"""
import math
import os

T = 16.0
N = 256
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

vals = [int(round(65536.0 * math.exp(-k / T))) for k in range(N)]
assert vals[0] == 65536 and all(0 <= v <= 65536 for v in vals)

c_path = os.path.join(ROOT, "sw/bitlm/golden/bitlm_exp_lut.h")
v_path = os.path.join(ROOT, "rtl/accelerator/bitlm_exp_lut.vh")

with open(c_path, "w") as f:
    f.write("/* Generovano tools/bitlm/gen_exp_lut.py - neupravovat rucne. */\n")
    f.write(f"/* EXP_LUT[k] = round(65536 * exp(-k / {T:g})), k = 0..{N - 1} */\n")
    f.write("#ifndef BITLM_EXP_LUT_H\n#define BITLM_EXP_LUT_H\n#include <stdint.h>\n")
    f.write(f"static const uint32_t BL_EXP_LUT[{N}] = {{\n")
    for i in range(0, N, 8):
        f.write("    " + ", ".join(f"{v:5d}" for v in vals[i:i + 8]) + ",\n")
    f.write("};\n#endif\n")

with open(v_path, "w") as f:
    f.write("// Generovano tools/bitlm/gen_exp_lut.py - neupravovat rucne.\n")
    f.write(f"// exp_lut(k) = round(65536 * exp(-k / {T:g})), k = 0..{N - 1}\n")
    f.write("function [16:0] exp_lut;\n    input [7:0] k;\n    begin\n        case (k)\n")
    for k, v in enumerate(vals):
        f.write(f"            8'd{k}: exp_lut = 17'd{v};\n")
    f.write("            default: exp_lut = 17'd0;\n        endcase\n    end\nendfunction\n")

print("zapsano:", c_path, v_path)
