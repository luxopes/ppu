# PPU — Performance Processor Unit

Projekt pro návrh hybridního PPU (CPU + GPU + ultra‑light cores). Cíle:

- FPGA prototyp pro ověření funkcí
- Linux support (bootloader, kernel, drivers)
- Fyzický návrh a tape‑out na SkyWater `sky130`

## Aktuální milník: BitLM, 1-bitový jazykový model trénovaný na FPGA (PoC)

Celý trénink 1-bitového jazykového modelu (výběr vzorku, dopředný a zpětný
průchod, softmax, update vah) běží v akcelerátoru `rtl/accelerator/bitlm_core.v`.
Výsledky jsou bitově shodné s referenčním modelem v C. Cílová platforma je
AWS EC2 F2 (VU47P). Podrobnosti: **[docs/bitlm_poc.md](docs/bitlm_poc.md)**,
cloud: **[fpga/aws_f2/README.md](fpga/aws_f2/README.md)**.

Rychlý start (Linux, gcc, Verilator ≥ 5):

```sh
python3 tools/bitlm/prepare_corpus.py data/rur_pg59112.txt data/rur.tok
gcc -O2 -o build/bitlm_train sw/bitlm/golden/bitlm_train.c sw/bitlm/golden/bitlm_ref.c -lm
./build/bitlm_train -c data/rur.tok -n 2000000 -d 500000 -D 3   # referenční trénink na CPU
make -C tools/sim/bitlm run      # RTL vs. reference, bit po bitu
make -C tools/sim/bitlm train    # 50k kroků tréninku přímo v RTL
make -C tools/sim run            # test původního PPU (CPU + dot-product)
```

## Struktura

- `docs/` — dokumentace (`bitlm_poc.md` = specifikace PoC)
- `fpga/` — FPGA wrappery (`top/bitlm_fpga_top.v`), AXI (`axi/`), AWS F2 (`aws_f2/`), simulace
- `rtl/` — Verilog moduly (cores, accelerator, interconnect, memory)
- `sw/` — bootloader, kernel, ovladače, příklady; `sw/bitlm/` = referenční model a host ovladač
- `tapeout/` — PDK & signoff artefakty
- `tools/` — skripty pro synth/sim/pci; `tools/bitlm/` = příprava dat, sweep
- `data/` — trénovací korpus; `results/` — logy z tréninku a ověření
