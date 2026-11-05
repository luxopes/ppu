# BitTF ISA v1: instrukce jednotky pro trenink

Cip (RTL v `rtl/bittf`) neprovadi pevne zadratovany model, ale **program** z nekolika obecnych
instrukci. Program jednoho kroku treninku (vyber davky, dopredu, zpet, Lion) vytvori prekladac
`tools/bitlm/bittf_compile.py` z konfigurace modelu; m1, m5 i m14 tak bezi na stejnem cipu.

Retez overeni:

```
golden model (C, bittf_ref.c)  ==  program na simulatoru ISA (bt_isa.c)  ==  RTL (Verilator)  ==  FPGA
         bt_isa_run: logity, gradienty a stav po kazdem kroku        RTL vs ISA: pamet po instrukcich
```

Definice cisel instrukci je jen v `tools/bitlm/bittf_isa.py` (generuje `sw/bittf/isa/bt_isa_ops.h`
a `rtl/bittf/bittf_isa.vh`). Referencni chovani = `sw/bittf/isa/bt_isa.c`.

## 1. Stav stroje

- **Pamet**: bajtove adresovana, little endian, 32bitove adresy. Obsahuje program, parametry, korpus
  a vsechny mezivysledky. Kazdy operand ma sirku prvku 1/2/4/8 B a znamenko.
- **R0..R63**: skalarni registry int64, R0 = 0.
- **CSR**: `SEED`, `CTR` (citac proudu nahodnych cisel), `ERR` (kod chyby), `CYCLES`.
- **PC**: index instrukce od zacatku programu (instrukce = 48 slov po 32 bitech).

Zapis hodnoty, ktera se nevejde do sirky ciloveho prvku, je **chyba** (`ERR = 1`, stop) - tim se
odhali spatne zvolene sirky tenzoru.

## 2. Format instrukce

| Slovo | Obsah |
|---|---|
| 0 | `op` (8) \| `fn` (8) \| priznaky (16) |
| 1-4 | pocty smycek `n0..n3` (`n0` nejvnitrnejsi) |
| 5-28 | operandy D, A, B, C, kazdy 6 slov: adresa, konfigurace, kroky `s0..s3` (v prvcich) |
| 29-35 | gather: adresa indexu, konfigurace, kroky `g0..g3`, meritko |
| 36 | `rs1` \| `rs2` \| `rd` \| `rs3` |
| 37-39 | `imm`, `imm2`, `imm3` (se znamenkem) |

Adresa prvku operandu pro indexy smycek `i0..i3`:
`base + sirka * (s0*i0 + s1*i1 + s2*i2 + s3*i3 [+ idx * meritko])`, kde u operandu s priznakem gather
je `idx = pamet[gbase + sirka_idx * (g0*i0 + ... + g3*i3)]` (vyber tokenu z embeddingu, okna z korpusu).
Kroky 0 = broadcast (napr. hodnota na radek). Transpozice a preskladani hlav jsou jen jine kroky.

## 3. Instrukce

### Skalarni (`SCALAR`) a rizeni (`BRANCH`, `HALT`)

`LI` (64bitova konstanta), `ADD/SUB` (`rd = rs1 +- rs2 + imm`), `MAX`, `MIN`, `BITLEN` (`bitlen(rs1) + imm`),
`SEL` (`rs3 ? rs1 : rs2`), `SHRA`, `NE`, `LD/ST` (skalar z/do pameti), `CSRR/CSRW`.
Vetveni `BEQZ`, `BNEZ`, `JMP` na absolutni index instrukce. Pouziti: exponenty blokove plovouci carky,
preskoceni nuloveho gradientu (stejne jako golden).

### Po prvcich (`EW`)

Pro kazdy prvek 4-D smycky: `D = f(A, B, C)`. Posun `SH = imm + R[rs1] + tc(C)`,
`tc` je 0, `+C`, `-C`, `+(C>>1)` nebo `-(C>>1)`.

| fn | vysledek | pouziti |
|---|---|---|
| `COPY ADD SUB MUL` | | preskladani, exponenty, `y = x^ * g` |
| `RSR` | `rsr(A, SH)` | posun se zaokrouhlenim, kvantizace |
| `ADDRSR` | `A + rsr(B, SH)` | pricteni do rezidua |
| `MULRSR` | `rsr(A*B, SH)` | norma, softmax |
| `MULRELURSR` | `rsr(A*max(B,0), SH)` | zpetny ReLU² |
| `MULSUB` | `A*B - C` | zpetna norma |
| `SRD` | `srd(A, SH, nahoda)` | stochasticke zaokrouhleni |
| `POW2Q` | znamenko x 2^k, nestranne | gradient pro `dW` |
| `BITLENI`, `SEXPR` | `bitlen(A) + imm`, `SH` | exponenty |
| `NRMS`, `NRMR` | posun a 1/sqrt z tabulky RSQ | RMS norma |
| `EXP8`, `EXP16`, `RCP` | tabulky exp, `floor((2^imm + A/2) / A)` | softmax attention a CE |
| `OHSUB`, `ONEHOT` | `A - [i0 == B] 2^imm`, `[i0 == A]` | gradient CE, embedding |
| `RELU2`, `MEANDIV` | `max(A,0)²`, `floor((2A + d) / 2d)` | MLP, norma |
| `RSR2ADD` | `rsr(A, R1) + rsr(B, R2)` | soucet BFP |
| `LIONQ`, `MULADDI` | `A - sgn(7B + C) 2^R1`, `imm*A + B` | Lion |
| `RNG`, `MODU` | nahodne cislo, `A mod imm` | vyber davky |

Priznaky: orezani na `imm3` bitu (se znamenkem, nebo symetricky), **kauzalni maska** (`i0 > i1` ->
`R[rs2]`), **RNG** (fn `SRD/POW2Q/RNG`: novy proud `key = fmix32(SEED ^ fmix32(CTR))`, `CTR++`, prvek
s plochym indexem `f` dostane `fmix32(f * 0x9E3779B1 + key)` - stejne jako golden).

### Redukce (`RED`)

`D[i1,i2,i3] = kombinace pres i0 z p(A,B,C)`, kombinace `SUM`/`MAX`/`AMAX`, predzpracovani
`A`, `A*A`, `A*B`, `A*B*C`, `SH`. S priznakem `TOREG` jde vysledek do `R[rd]` (globalni maximum).

### Maticove nasobeni (`MM`)

`D[i2][i1][i0] = sum_{k < n3} A(i2, i1, k) * B(i2, i0, k)`, akumulator 48 bitu (preteceni = chyba).
Rezim `SIGN` operandu B: hodnota `q >= 0 ? +1 : -1` - 1bitove vahy cte primo ze stinu int16.
Kroky urcuji transpozici, takze stejna instrukce dela `x·Wᵀ`, `g·W`, `gᵀ·x` i vsechny matice attention.

## 4. Program kroku

`bittf_compile.py` generuje program po funkcich golden modelu (stejne nazvy: `norm_fwd`, `quant_tok`,
`lin_bwd`, `attn_bwd`, `add_bfp`, `lion`). m1 (d128, L5, B32): 2 493 instrukci, 16,3 G MAC a 0,52 G
operaci po prvcich na krok. Host pred krokem zapise jen `lr` (rozvrh); vyber davky z korpusu,
cely trenink a nahodna cisla bezi na cipu.

## 5. Overeni

`sw/bittf/isa/bt_isa_run <prefix> <korpus>`: po doprednem pruchodu porovna logity, pred Lionem
gradienty a exponenty vsech parametru, po kroku hash stavu (stejny jako golden `bt_state_hash`).

| konfigurace | kroku | vysledek |
|---|---|---|
| d32 L1 H2 ff64 T16 B2 | 300 | shoda |
| d64 L2 H2 ff128 T128 B4 | 5 | shoda |
| m1: d128 L5 H4 ff648 T128 B32 | 4 | shoda (16,3 G MAC + 0,52 G prvku na krok) |

## 6. RTL a stav (2026-11-05)

| jadro | popis | krok m1 (d128 L5 B32) |
|---|---|---|
| v1 `bittf_core.v` | sekvencni reference, 1 pristup do pameti naraz | (jen male konfigurace) |
| v2 `bittf_core2.v` | 5 radkovych portu (64 B), EW/RED 8 drah, MM pole 32x32 (DSP), dvojite buffery | **109 M taktu, bitove shodne s golden** |

Profil v2 na m1 (latence pameti 20 taktu): EW 47 %, RED 26 %, MM 27 %; 139 M radku = 8,9 GB na krok.
Pri 250 MHz ~0,44 s/krok (golden C na 4 vlaknech 2,8-3,9 s). Na F2 s jednim kanalem DDR4 (~16 GB/s)
omezi pasmo na ~0,6 s/krok; HBM (32 kanalu) a uzsi mezivysledky jsou dalsi krok.

FPGA obal `fpga/top/bittf_fpga_top.v` (AXI4-Lite registry + most na AXI4) je overen testbenchem
`sw/bittf/verilator/tb_fpga.cpp` s AXI modelem pameti (nahodna latence a ready).
