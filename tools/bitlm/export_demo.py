#!/usr/bin/env python3
"""Export snimku vah BitLM pro webove demo (inference v prohlizeci).

Pro inference staci znamenka vah (1 bit) a tabulka x = clamp(emb >> 8, -127, 127).
Vystup: JSON {step: base64(W1 bity | W2 bity | x int8)} + testovaci vektory z C
modelu (sw/bitlm/golden), aby slo overit, ze JS pocita totez.

Pouziti: export_demo.py vystup.json snap_0.bin snap_2000.bin ...
"""
import base64
import json
import os
import re
import struct
import sys

H, D, V, E = 512, 256, 32, 32


def load(path):
    raw = open(path, "rb").read()
    n1, n2, ne = H * D, V * H, V * E
    vals = struct.unpack("<%dh" % (n1 + n2 + ne), raw[:2 * (n1 + n2 + ne)])
    return vals[:n1], vals[n1:n1 + n2], vals[n1 + n2:]


def pack_bits(ws):
    out = bytearray((len(ws) + 7) // 8)
    for i, w in enumerate(ws):
        if w >= 0:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    snaps = {}
    for p in paths:
        step = int(re.search(r"(\d+)\.bin$", p).group(1))
        w1, w2, emb = load(p)
        x = bytes(((max(-127, e >> 8)) & 0xFF) for e in emb)   # >> je v Pythonu floor = asr
        blob = pack_bits(w1) + pack_bits(w2) + x
        snaps[step] = base64.b64encode(blob).decode()
    json.dump({"H": H, "D": D, "V": V, "E": E, "N": 8, "s2": 4,
               "snaps": dict(sorted(snaps.items()))}, open(out, "w"))
    print(out, os.path.getsize(out), "B,", len(snaps), "snimku")


if __name__ == "__main__":
    main()
