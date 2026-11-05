#!/usr/bin/env python3
"""Vygeneruje rtl/bittf/bittf_tables.vh (ROM 1/sqrt a exp) z sw/bittf/golden/bittf_tables.h."""
import os
import re

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
src = open(os.path.join(root, "sw/bittf/golden/bittf_tables.h")).read()


def table(name):
    m = re.search(r"%s\[\d+\] = \{([^}]*)\}" % name, src)
    return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def vfun(name, v, ibits, obits):
    s = [f"function [{obits - 1}:0] {name};\n    input [{ibits - 1}:0] i;\n    begin\n        case (i)\n"]
    for k, x in enumerate(v):
        s.append(f"        {ibits}'d{k}: {name} = {obits}'d{x};\n")
    s.append(f"        default: {name} = {obits}'d0;\n        endcase\n    end\nendfunction\n")
    return "".join(s)


rsq, e8, e16 = table("BT_RSQ"), table("BT_EXP8"), table("BT_EXP16")
assert len(rsq) == 256 and len(e8) == 256 and len(e16) == 512
assert max(rsq) < 1 << 24 and max(e8) < 1 << 16 and max(e16) < 1 << 17
out = os.path.join(root, "rtl/bittf/bittf_tables.vh")
with open(out, "w") as f:
    f.write("// Vygenerovano tools/bitlm/gen_bittf_rtl_tables.py - neupravovat rucne.\n")
    f.write(vfun("rsq_lut", rsq, 8, 24) + vfun("exp8_lut", e8, 8, 16) + vfun("exp16_lut", e16, 9, 17))
print("zapsano", out, max(rsq), max(e8), max(e16))
