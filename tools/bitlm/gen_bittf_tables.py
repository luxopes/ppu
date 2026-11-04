#!/usr/bin/env python3
"""Vygeneruje sw/bittf/golden/bittf_tables.h z tabulek intsim.py (ROM cipu: 1/sqrt, exp)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intsim as S  # noqa: E402


def arr(name, t, ctype):
    v = t.tolist()
    rows = [", ".join(str(x) for x in v[i:i + 16]) for i in range(0, len(v), 16)]
    return f"static const {ctype} {name}[{len(v)}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../sw/bittf/golden/bittf_tables.h")
with open(out, "w") as f:
    f.write("/* Vygenerovano tools/bitlm/gen_bittf_tables.py - neupravovat rucne. */\n")
    f.write("#ifndef BITTF_TABLES_H\n#define BITTF_TABLES_H\n#include <stdint.h>\n")
    f.write("/* 1/sqrt: RSQ[i] = round(2^22 / sqrt((i + 0.5) * 256)), i = 64..255 */\n")
    f.write(arr("BT_RSQ", S.RSQ, "int32_t"))
    f.write("/* EXP8[i] = round(255 exp(-i/32)) */\n" + arr("BT_EXP8", S.EXP8, "int32_t"))
    f.write("/* EXP16[i] = round(65535 exp(-i/32)) */\n" + arr("BT_EXP16", S.EXP16, "int32_t"))
    f.write("#endif\n")
print("zapsano", out)
