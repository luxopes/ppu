# BitTF: jednotka pro trenink 1-bit transformeru v cipu (navrh)

Navazuje na `docs/bitlm_poc.md` (kap. 9 a 10). Cil: trenovat primo na FPGA (AWS F2, VU47P)
1-bit transformer ve tvaru, ktery uz je overeny v PyTorch simulaci cipu
(`tools/bitlm/concept_lab.py --chip`, `tools/bitlm/train_big.py`):

- vahy linearnich vrstev 1 bit: `sign(w) * 2^-round(log2 sqrt(fan_in))`, int16 stin,
- aktivace int8 s dynamickym posunem (per-token absmax zaokrouhleny na mocninu dvou),
- zpetny pruchod int8 se stochastickym zaokrouhlenim,
- optimalizator Lion s int8 momentem, krok `+-2^k` LSB stinu, stochasticke zaokrouhleni.

Kvalita teto varianty (1,18M parametru, kontext 128, 8000 x 32 sekvenci): 2,530 bit/znak
proti 2,451 u AdamW ve float (kap. 10).

## 1. Cilove velikosti

| model | d | vrstev | hlav | ff | parametru | 1-bit vahy | stin int16 + moment int8 |
|---|---|---|---|---|---|---|---|
| m1 (overeni) | 128 | 5 | 4 | 648 | 1,18 M | 145 KiB | 3,4 MiB |
| m5 | 256 | 6 | 8 | 1024 | 4,77 M | 0,6 MiB | 13,6 MiB |
| m14 | 384 | 8 | 8 | 1536 | 14,2 M | 1,7 MiB | 40,7 MiB |

VU47P: ~1,3 M LUT, 9 024 DSP, ~34 MiB URAM + BRAM, 16 GiB HBM2 (~460 GB/s).
m1 a m5 drzi vahy, stin i moment v URAM; m14 drzi stin a moment v HBM (1-bit vahy v URAM).

## 2. Krok treninku a tok dat

Jeden krok = B sekvenci x T=128 tokenu (B=32 -> 4096 tokenu), pak jeden update vah.

1. **Dopredny pruchod** vrstvu po vrstve pro vsech 4096 tokenu. Vstupy vsech linearnich vrstev
   (int8) a reziduum se ukladaji do HBM (~17 KiB na token u m5, ~70 MiB na krok).
   Pravdepodobnosti attention se neukladaji (prepocitaji se ve zpetnem pruchodu, jako FlashAttention).
2. **Ztrata**: softmax logitu (tabulka exp) - one-hot -> delta int8.
3. **Zpetny pruchod** vrstvu po vrstve:
   - `dx = W^T * delta` - binarni matice, stejna jednotka jako dopredu (transpozice zdarma, viz 3.1),
   - `dW = sum_t delta_t (x) x_t` - int8 x int8 pres vsech 4096 tokenu (DSP pole),
   - **Lion update je slouceny s vypoctem dW**: jakmile je dlazdice dW hotova, hned se precte
     stin + moment, spocita update a zapise zpet. Gradient se nikde neuklada -> zadny buffer
     velikosti modelu.
4. Embedding (svazany s vystupem) a jeho gradient pres vektorovou jednotku.

Proto neni potreba pamet na gradienty a velikost davky je omezena jen kapacitou HBM.

## 3. Bloky

```
            AXI-Lite (registry, mikrokod)      PCIe DMA (korpus, vahy)
                    |                                 |
              +-----------+    +----------------------------------------+
              | Sekvencer |--->|        Sbernice scratchpadu (URAM)      |
              | mikrokodu |    +----+-----------+-----------+-----------+
              +-----------+         |           |           |           |
                               +---------+ +---------+ +---------+ +---------+
                               |  BMM    | |  MAC    | |  VEC    | |  LION   |
                               | 1-bit   | | int8xint8| | SIMD 32 | | update  |
                               | matice  | | DSP pole | | lanes   | | vah     |
                               +---------+ +---------+ +---------+ +---------+
                                    |           |           |           |
                               +-----------------------------------------+
                               |  Vahy: znamenka (URAM), stin+moment     |
                               |  (URAM/HBM), aktivace kroku (HBM)       |
                               +-----------------------------------------+
```

### 3.1 BMM - binarni nasobeni matic (dopredu i `dx` zpet)

`y[t,o] = sum_i x[t,i] * s[o,i]`, `s = +-1`: jen scitani/odcitani int8, strom scitacek.
Weight-stationary: dlazdice znamenek 64 x 512 bitu je v registrech, tokeny tecou po jednom za takt
-> 32 k operaci za takt (8 Tops/s pri 250 MHz). Znamenka se nactou jednou na 4096 tokenu, takze
pasmo pameti vah je zanedbatelne. Pro `dx = W^T delta` se pouzije stejna dlazdice transponovane
(scitaci strom pres druhou osu) - zadna kopie transponovanych vah.
Odhad: ~300 k LUT.

### 3.2 MAC - int8 x int8 pole na DSP (`dW`, attention)

Output-stationary dlazdice 64 x 64 akumulatoru int32, 2 nasobeni int8 na DSP (2 048 DSP).
Tokeny tecou po jednom za takt; po 4096 taktech je dlazdice `dW` hotova a jde do LION.
Stejne pole pocita attention: `S = Q K^T`, `O = P V` a jejich gradienty (T=128, d_h=32..48).
Odhad: 2 048 DSP, 1 TMAC/s.

### 3.3 VEC - vektorova jednotka (32 lane)

Dynamicky posun (absmax -> mocnina dvou), kvantizace int8 se stochastickym zaokrouhlenim
(fmix32 hash jako v BitLM), reziduum, ReLU (nebo ReLU^2), normalizace, softmax po radcich
(exp tabulka Q16 z BitLM, jedno deleni na radek), ztrata, embedding gather/scatter.

### 3.4 LION - update vah

Pro kazdou vahu: `c = sign(b1*m + (1-b1)*g)`, `stin -= SR(c * 2^k)`, `m = b2*m + (1-b2)*g`
(m int8 s mocninou dvou na tenzor, stochasticke zaokrouhleni). `b1 = 0.9`, `b2 = 0.99` se
v cipu nahradi posuny (`m - m>>3 ...`) - presne konstanty overi golden model.
64 vah za takt, dlazdice 64 x 64 za 64 taktu (skryto za 4096 takty vypoctu dalsi dlazdice).

### 3.5 Sekvencer

Program kroku je seznam instrukci (BMM, MAC, VEC, LION, DMA) v BRAM, nahrava ho host. Rozmery
modelu jsou jen parametry instrukci -> m1, m5 i m14 bezi na stejnem cipu bez nove syntezy.

## 4. Odhad vykonu (m5)

Na krok (4096 tokenu): dopredu 2N*4096 = 39 G binarnich operaci, `dx` 39 G, `dW` 19,5 G MAC.
BMM 8 Tops/s -> ~10 ms, MAC 1 TMAC/s -> ~20 ms; s prekryvem a rezii attention/VEC odhad
**20-30 ms na krok, tj. 140-200 tisic tokenu/s**. GitHub CPU (4 vCPU): ~3 400 tokenu/s
-> **zhruba 40-60x rychleji**; zdvojnasobeni MAC pole (4 096 DSP) posouva odhad k ~80x.
Cely dnesni trenink m5 (64 000 kroku) by trval ~25-35 minut misto ~1 dne.
Jde o odhad z parametru cipu; skutecne cislo da az RTL simulace a mereni na F2.

## 5. Postup (stejny jako u BitLM)

1. **Zjednoduseni pro HW overit v PyTorch** (`--hw-norm/--hw-act/--hw-attn/--hw-pow2`):
   LayerNorm -> normalizace absmax (posun), GELU -> ReLU, attention celociselne. [bezi]
2. **Celociselna specifikace + golden model v C** (`sw/bittf/golden/`): kompletni integer
   forward/backward/Lion, overit kvalitu proti PyTorch (cil: do ~0,05 bitu od 2,53).
3. **RTL po blocich** (BMM, MAC, VEC, LION, sekvencer), kazdy blok bitove presne proti golden
   modelu ve Verilatoru; pak cela jednotka na m1, mutacni testy.
4. **AWS F2**: HBM/DMA, Vivado build, mereni rychlosti a treninku na skutecnem HW.

## 6. Cesta ke kremiku (ASIC)

Cil do budoucna: vyrobit cip na levnem akademickem MPW (sdileny wafer), typicky 130 nm
s otevrenym PDK (IHP SG13G2, SkyWater 130, GF180) a otevrenymi nastroji (OpenROAD/LibreLane),
pripadne pres Europractice. Pozn.: Efabless (drive bezplatne shuttly OpenMPW/chipIgnite) v roce
2025 skoncil; aktualni moznosti a ceny je nutne overit v dobe tapeoutu.

Dusledky pro navrh uz ted:

- **RTL bez vendor primitiv**: zadne DSP/URAM instance, jen genericky Verilog; pameti za vlastnimi
  wrappery (`bittf_ram_*`), ktere na FPGA mapuji na BRAM/URAM a na ASIC na SRAM makra (OpenRAM,
  makra foundry). Nasobicky obecne `*` s parametrizovatelnym poctem.
- **Paralelismus jako parametr** (BMM lanes, MAC dlazdice, VEC lanes): FPGA 32k/2048/32,
  ASIC demonstrator radove 256-1k / 16-64 / 8 - stejny mikrokod, jen pomalejsi.
- **Pamet je hlavni omezeni kremiku**: SRAM na 130 nm ~0,1 MiB/mm^2. 1-bit vahy m1 (145 KiB)
  se na cip vejdou (~1,5 mm^2), stin int16 + moment int8 (3,4 MiB) ne -> u ASIC jsou stin a moment
  v externi pameti (PSRAM/LPDDR pres jednoduche rozhrani) nebo se pouzije optimalizator **bop**
  bez stinu (1+8 bitu na vahu, kap. 10 bitlm_poc.md: 2,70 bit/znak) - pro kremik je proto
  dulezitejsi nez pro FPGA.
- **Hodiny** ~100-200 MHz na 130 nm, synchronni reset, jedna hodinova domena v jadru.
- Prvni kremik realisticky jako **demonstrator**: inference + trenink malych modelu (m1) s externi
  pameti; vykon nizsi nez FPGA, ale vlastni cip.
