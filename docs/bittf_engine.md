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

**Rozhodnuti (2026-11-04):** cil je 130 nm (levne akademicke MPW, otevrene PDK IHP SG13G2 /
SkyWater 130 / GF180, nastroje OpenROAD/LibreLane), pri schvaleni financi 65 nm (pres Europractice,
PDK foundry pod NDA, komercni nebo akademicke nastroje). **Pamet je externi na PCB.**
Efabless (drive bezplatne OpenMPW/chipIgnite) v roce 2025 skoncil; aktualni shuttly a ceny overit
v dobe tapeoutu.

### 6.1 Pravidla pro RTL uz ted

- Genericky Verilog bez vendor primitiv. Nasobicky a pameti za wrappery (`bittf_mul`, `bittf_sram`),
  na FPGA -> DSP/BRAM/URAM, na ASIC -> standardni bunky / SRAM makra (OpenRAM nebo foundry).
- Jadro mluvi s pameti pres **jeden genericky port** (burst req/valid, sirka parametrem).
  FPGA: most na AXI4 do HBM/DDR. ASIC: radic externi pameti (6.3). Jadro se tim nemeni.
- Paralelismus parametrem (BMM lanes, MAC dlazdice, VEC lanes); mikrokod stejny pro vsechny cile.
- Jedna hodinova domena v jadru, synchronni reset, radic pameti ve vlastni domene s CDC FIFO.

### 6.2 Co je na cipu a co venku

| | na cipu (SRAM) | externi pamet na PCB |
|---|---|---|
| 1-bit znamenka vah | ano (m1 145 KiB; m5 0,6 MiB jen na 65 nm) | kopie |
| stin int16 + moment int8 | - | ano (m1 3,4 MiB, m5 13,6 MiB) |
| aktivace kroku | dlazdice (desitky KiB) | ano (~7 KiB/token u m1) |
| korpus | - | ano (nebo stream z hostitele) |

### 6.3 Externi pamet a pasmo

Na 130/65 nm je cip vypocetne omezeny, ne pametove: krok m1 s 4096 tokeny ma ~4,8 G MAC pro dW;
pri 64 MAC x 150 MHz je to ~0,5 s, zatimco prenos stinu+momentu (7 MiB) a aktivaci (~56 MiB)
trva pri ~1,6 GB/s ~40 ms. Staci proto jednoducha pamet bez drahe PHY:

| pamet | rozhrani | pasmo | poznamka |
|---|---|---|---|
| **SDR SDRAM** 32 bit (napr. 2x 16bit cip) | bezne I/O pady, 133 MHz | ~0,5 GB/s | otevrene radice, kapacita 32-64 MiB |
| **HyperRAM / OctoSPI PSRAM** (4 kanaly) | 8 bit DDR, 12 pinu/kanal | ~1,6 GB/s | malo pinu, 8-64 MiB/cip |
| LPDDR4 | vyzaduje licencovanou PHY | > 10 GB/s | jen pro 65 nm s rozpoctem; pro tento vykon zbytecne |

Doporuceni: **130 nm -> 2-4 kanaly HyperRAM** (nejmene pinu, nejjednodussi PCB);
**65 nm -> 4-8 kanalu HyperRAM nebo SDRAM 32 bit**. Spojeni s PC pres SPI/USB most
(mikrokontroler na desce) - registry a mikrokod jako dnes pres AXI-Lite.

### 6.4 Odhad velikosti

| | 130 nm | 65 nm |
|---|---|---|
| hodiny | 100-150 MHz | 300-400 MHz |
| BMM lanes / MAC / VEC | 1 k / 64 / 8 | 4 k / 256 / 16 |
| SRAM na cipu | ~0,25 MiB (~2-3 mm^2) | ~1 MiB |
| model pro trenink | m1 (1,2 M) | m1-m5 |
| krok m1 (4096 tokenu) | ~0,5 s (~8 k tokenu/s) | ~0,05 s (~80 k tokenu/s) |

Optimalizator **bop** (bez stinu, 1+8 bitu na vahu) zmensi externi pamet a prenosy o ~60 %;
pro kremik ma smysl ho doladit (dnes 2,70 vs 2,53 u Lionu).

### 6.5 Dvouurovnova pamet a vic cipu na desce (rozhodnuti 2026-11-04)

- **Rychla pamet**: SRAM na cipu (0,25-1 MiB) + 2-4x HyperRAM na desce (16-256 MiB, ~1,6 GB/s):
  stin, moment a aktivace pri treninku modelu do ~10-20 M parametru.
- **Velka pamet (GB)**: DDR3/DDR4 2-4 GiB pres **pametovy most** na desce (male FPGA, napr. ECP5
  s otevrenym radicem LiteDRAM), protoze DDR PHY na 130 nm je licencovana/narocna; na 65 nm pripadne
  vlastni PHY. Pouziti: inference velkych 1-bit modelu (1 GiB ~ 8 G 1-bit vah), uloziste korpusu
  a checkpointu. Velikost *trenovaneho* modelu na 130 nm omezuje vypocet (~150 G binarnich op/s),
  ne pamet.
- **Vic cipu na jedne desce**: meziciopove spojeni (synchronni paralelni sbernice nebo seriovy link)
  a deleni modelu po vrstvach (pipeline) nebo po davce (data parallel; Lion pak potrebuje secteni
  gradientu mezi cipy). V RTL: genericky port pro link ve stejnem stylu jako pametovy port.
- IHP Free/Open MPW dodava typicky 10-40 netestovanych holych cipu (podle behu) -> nekolik desek
  po 2-4 cipech + rezerva; pouzdreni (QFN/COB) se plati zvlast.

### 6.6 Cilova deska (zamer uzivatele 2026-11-04, k dopracovani po dobehnuti treninku)

- **4 cipy BitTF na jedne desce**, deska = **PCIe karta do PC**; pokud by PCIe narazilo na licence,
  pak **externi USB modul**. Veskere vypocty (trenink i generovani) bezi na karte, PC jen nahraje
  korpus/program a cte vysledky.
- Pamet: **HyperRAM** u kazdeho cipu (rychla) + bezna DDR (GB) pro celou kartu.
- Licence rozhrani (DDR PHY, PCIe PHY) vyresit **ridicim FPGA na desce** misto vlastnich PHY na cipu:
  napr. Xilinx Artix-7 (PCIe Gen2 hard blok a DDR3 radic MIG jsou ve volnem Vivadu bez licence),
  nebo Lattice ECP5 + otevrene LiteDRAM/LitePCIe. FPGA = PCIe/USB endpoint + radic DDR + rozvod dat
  ke 4 cipum. Cipy BitTF maji jen jednoduche digitalni rozhrani (HyperRAM, link k FPGA/sousedum)
  -> zadna licencovana IP na kremiku.
- Otevrene body: PCI vendor ID (pro prototyp ID vyrobce FPGA, pro produkt clenstvi v PCI-SIG),
  USB varianta (USB 3 pres FPGA nebo most FT601), topologie linku mezi 4 cipy (kruh / hvezda pres FPGA).

### 6.7 Ovladac a software (zamer)

Vrstvy (shora): nastroje `bittf train/generate` + Python API -> knihovna `libbittf` (registry,
mikrokod, DMA; stejna abstrakce rd/wr jako dnes `sw/bitlm/host/bitlm_hw.c`, takze bezi nad simulaci,
AWS F2 i vlastni kartou) -> jadrovy ovladac.
- **Linux, PCIe**: ridici FPGA s Xilinx XDMA -> hotovy otevreny ovladac `dma_ip_drivers` (GPL),
  nebo zcela bez vlastniho modulu pres **VFIO** (mapovani BAR + DMA z user space, IOMMU).
- **Windows, PCIe**: vlastni ovladac musi byt podepsany (EV certifikat + Microsoft attestation) -
  narocnejsi, az pozdeji.
- **USB modul**: WinUSB/libusb - **zadny vlastni ovladac** na Windows ani Linuxu.

### 6.8 Maximalni vykon ze stareho procesu (zamer 2026-11-04)

Zadani uzivatele: vytezit z 130/65 nm maximum. Princip: **cely cip = jedno velke pole scitacek + SRAM**,
zadne vyhrazene int8 nasobicky.

- **Jedno sdilene pole pro vsechny tri matice kroku.** Dopredu `x·W` a zpet `g·Wᵀ` jsou s 1-bit vahami
  jen scitani (XNOR/popcount, pricist/odecist); stejne pole cte W po radcich i po sloupcich.
- **`dW = xᵀ·g` po bitovych rovinach:** `dW = Σ_b 2^b · (xᵀ · g_b)`, kazda rovina `g_b` je binarni
  -> dW pocita totez pole scitacek, jen v tolika pruchodech, kolik bitu ma gradient. Blok MAC
  (3.2) v ASIC odpada; plochu dostane pole a SRAM.
- **Mene bitu gradientu pro dW = mene pruchodu** (Lion bere jen znamenko momentu, takze presnost
  gradientu muze byt nizka). Overuje se `--dwq` (int8 = 8 pruchodu, int4 = 4, ternarni = 1-2,
  per-tensor nebo per-token mocnina dvou, vse se stochastickym zaokrouhlenim -> nestranne).
- **Uzsi aktivace (`--abits` 6/4)** = uzsi scitacky -> vic drah na stejne plose.
- **Attention** (maly podil prace pri ctx 128) na stejnem poli: `q·kᵀ` a `p·v` take po bitovych
  rovinach (int8 = 8 pruchodu), softmax ve VEC tabulkou.
- **Pamet:** 1-bit vahy v SRAM (512 KB = 4 Mi vah = cela matice 1024x4096, weight-stationary),
  velka davka, prepocet attention; 2 kanaly HyperRAM / Octal xSPI na cip, pokud dovoli piny.
- **Hodiny:** hluboke pipelinovani stromu scitacek (cil ~150-200 MHz na 130 nm).

Ocekavany zisk proti verzi s MAC: 2-4x rychlejsi krok na stejne plose (podle vysledku `--dwq`).
