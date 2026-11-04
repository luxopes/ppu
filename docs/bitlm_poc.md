# BitLM-nano: 1-bitový jazykový model trénovaný na FPGA (PoC)

Stav k 2026-11-03. Technická specifikace PoC v rámci projektu PPU.

## 1. Cíl a vymezení

Cílem PoC je ukázat, že **celý trénink** 1-bitového jazykového modelu může běžet
v hardwaru PPU na FPGA, tedy nejen inference. Celý trénovací krok běží na čipu:
výběr vzorku z korpusu, dopředný průchod, softmax, zpětný průchod a aktualizace
vah. Host jen nahraje text, nastaví hyperparametry a spustí N kroků.

Model je záměrně malý (PoC). Nejde o konkurenci velkým modelům. Ověřuje se
**trénovací smyčka v hardwaru**, která se pak dá škálovat (kap. 8).

## 2. Model

Model je znakový: slovník má 32 symbolů (mezera, a–z, `.,?!-`).

```
kontext 8 znaků ─► embedding 32 (int8; stínový int16)  ─► x ∈ int8^256
x ─► BitLinear W1 (512×256, 1 bit) ─► h ─► a = clamp(h >>> s, 0, 127)   (s dynamický)
a ─► BitLinear W2 (32×512, 1 bit)  ─► z = (W2·a) >>> s2 ─► softmax (LUT, Q16) ─► p
```

* **1-bitové váhy:** používá se jen znaménko stínové váhy (+1/−1). Dopředný
  i zpětný průchod přes váhy je tedy jen sčítání a odčítání, bez násobiček.
* **Stínové (latentní) váhy:** int16. Gradient se k nim propouští metodou
  straight-through estimator, stejně jako v BitNetu.
* **Aktivace:** uint7 s dynamickou škálou 2^-s. Jde o per-token absmax
  kvantizaci jako v BitNetu, ale s mocninou dvou, takže v HW stačí detekovat
  nejvyšší bit (bez dělení).
* **Softmax:** exp z tabulky (256 položek, Q16) a jedno dělení na vzorek
  (převrácená hodnota sumy, sekvenční dělička, 33 taktů).
* **Optimalizace:** online SGD (batch 1). Učicí kroky jsou bitové posuny.
  Plán učení (posun +1 každých 500k kroků) nastavuje host přepsáním registrů.
* **Velikost:** 147 456 jednobitových vah (W1 + W2), tedy 18 KiB, plus
  embedding 1024 × int8.

Aritmetika je celočíselná a **bitově přesně specifikovaná** referenčním modelem
`sw/bitlm/golden/bitlm_ref.c`. RTL ji implementuje 1:1 a testbench obě
implementace porovnává bit po bitu (kap. 5).

### Ladění, které bylo potřeba (záznam)

| Problém | Příčina | Řešení |
|---|---|---|
| Model se neučil (loss nad baseline) | malý rozsah logitů (s2=6), softmax téměř plochý | s2=4 |
| Všechny neurony saturované, W1 z 97 % kladné | floor-zaokrouhlení gradientů posouvá Σg; chybí normalizace | zaokrouhlující posuny + dynamická absmax škála aktivací |
| W1 se přestala měnit | gradienty zaokrouhlené na 0 (malé hodnoty, velký posun) | přesnější gradient (g1=4) |
| Řádky W2 se nafukují (95 % +1) | stínové váhy jen 11 bitů (clip 1023), embedding jen int8 | plný rozsah int16 (clip 32767), stínový embedding int16 |
| Strop kapacity ~3.17 bit/znak | malý model | E=32, H=512 → 2.87 bit/znak |

Varianty z BitNetu (centrování `sign(w − mean)` a bias logitů) jsou
v referenčním modelu jako `hparams.flags`. Na tomto korpusu nepomohly, proto
HW v1 implementuje `flags = 0`.

## 3. Výsledky referenčního modelu (CPU, bitově shodný s HW)

Korpus: Karel Čapek, *R.U.R.* (anglický překlad, Project Gutenberg #59112),
121 516 tokenů. 90 % trénink, 10 % validace. 2 000 000 kroků, seed 12345.

| Model | Validační entropie [bit/znak] | Přesnost top-1 |
|---|---|---|
| rovnoměrné rozdělení | 5.000 | 3 % |
| unigram | 4.294 | – |
| bigram | 3.521 | – |
| trigram | 2.944 | – |
| **BitLM-nano (1-bit, 18 KiB vah)** | **2.871** | **43.8 %** |

Text generovaný naučeným modelem (prompt „of the e“):

> l themownths all thans. helena. ohins crecfings to feal tommek mean thatk a
> pibwownatais is oll the raraid tarked yau. a to d. madame. hallemeier. ...
> what youlens you. helena. a domin. you. dr. gall. thats you thinkzat?

Model se naučil jména postav (helena, domin, hallemeier, dr. gall), strukturu
replik („jméno. text“) a běžná anglická slova. Na CPU (1 jádro, referenční C
kód) jde trénink rychlostí ~7 000 kroků/s.

Data: `results/golden_train_log.csv`, `results/golden_train.txt`.

## 4. Hardware: `rtl/accelerator/bitlm_core.v`

* Váhy W1 (2048 slov × 64 lane × 16 bit) a W2 (256 slov) jsou v blokové paměti.
  Každý průchod vahami je proud jednoho slova za takt (64 operací ±/MAC
  paralelně) s pevnou latencí 2 takty.
* Zpětný průchod W1 přeskakuje řádky s nulovým gradientem (neaktivní ReLU,
  ~68 % řádků).
* Krok treninku má fáze T_POS, FETCH, F_EMB, F_W1, F_ACT, F_W2, EXP, DIV,
  PROB, B_W2, B_DH, B_W1 a B_EMB.
* **Změřeno v RTL simulaci:** 3 430 taktů na trénovací krok (po zahřátí).
  To je **~73 000 kroků/s při 250 MHz**, tedy ~10× víc než referenční C kód
  na jednom jádře CPU (7 000 kroků/s). Celý trénink (2M kroků) zabere na FPGA
  ~27 s, na CPU 285 s. Na reálném čipu se počet taktů změří v registru CYCLES.
* Výkon škáluje lineárně s počtem lane (`LOG2_L`).
* Generování textu běží také v HW (vzorkování z p pomocí RNG xorshift32).

### Mapa registrů (AXI4-Lite, bajtové adresy)

| Adresa | Název | Popis |
|---|---|---|
| 0x000 | ID | `0x424C4D31` („BLM1“) |
| 0x004 | PARAMS | verze, log2 L, H, V, E, N |
| 0x008 | CTRL | 1 INIT (váhy z RNG), 2 TRAIN, 3 GEN, 4 CLEAR_STATS |
| 0x00C | STATUS | bit0 busy, bit1 done, bit2 chyba, [13:8] stav |
| 0x010 | SEED | seed RNG pro INIT |
| 0x014 | NSTEPS | počet trénovacích kroků |
| 0x018 | CORPUS_LEN | délka korpusu v tokenech (N < len ≤ 131072) |
| 0x01C | GEN_LEN | počet generovaných znaků (≤ 4096) |
| 0x020 | HP0 | g2, g1, s2, s1 (po 5 bitech v bajtech) |
| 0x024 | HP1 | lrb, lre, lr2, lr1 |
| 0x028 | HP2 | flags (HW v1: 0), clip |
| 0x02C–0x038 | statistiky | počet kroků, správné predikce, Σ p(cíl) (64 bit) |
| 0x03C | RNG | stav RNG (čtení/zápis) |
| 0x040–0x044 | CYCLES | takty posledního příkazu (64 bit) |
| 0x080– | CTX | kontext pro GEN (4 tokeny/slovo) |
| 0x010000 | GEN | vygenerovaný text (4 tokeny/slovo) |
| 0x020000 | CORPUS | korpus (4 tokeny/slovo) |
| 0x080000 | W1 | stínové váhy, index j·256+k, 1 váha/slovo |
| 0x100000 | W2 | index i·512+j |
| 0x110000 | EMB | index c·32+e |

Paměťová okna jsou z hosta přístupná jen v klidu (busy = 0).

## 5. Ověřování

* `tools/sim/bitlm`: Verilator simulace `bitlm_fpga_top` přes AXI4-Lite
  (stejné rozhraní jako AWS F2). Test `bitlm_verify()` (sdílený s host
  programem pro F2) porovnává po každé fázi **všechny váhy, embedding, stav RNG
  a statistiky** s referenčním modelem:
  * INIT;
  * 5 dávek tréninku s různými hyperparametry, včetně krajních: posuny 0, malý
    clip, saturace, velké posuny;
  * korpus délky N+1;
  * chybový vstup (len ≤ N → chyba bez změny vah);
  * generování 200 znaků (tokeny, RNG i kontext);
  * zápis a zpětné čtení vah z hosta.
* `tools/sim`: test původního PPU (CPU + dot-product akcelerátor) po opravách.
* Stejný `verify` běží i na reálném F2 (`sudo ./bitlm_f2 verify`), takže
  bitovou shodu lze doložit i na skutečném čipu.

**Stav ověření k 2026-11-03** (Verilator 5.053, OSS CAD Suite 2026-10-20):

| Test | Výsledek |
|---|---|
| `make -C tools/sim/bitlm run`: verify, 5 × 500 kroků + krajní případy + GEN + zápis vah | **0 neshod** (`results/rtl_verify_500.txt`) |
| `make -C tools/sim/bitlm train`: 50 000 kroků na celém korpusu v RTL | váhy bitově shodné s referencí po každých 10k krocích; val. entropie 3.5101 bit/znak = referenční běh (`results/rtl_train_50k.txt`) |
| Mutační test (3 záměrně vložené chyby do RTL) | všechny 3 odhaleny: zaokrouhlení (10 kroků), maska STE a shoda maxima v argmax (500 kroků). Proto je výchozí hodnota 500 kroků na dávku. |
| `make -C tools/sim run`: původní PPU po opravách | OK: 135 / 61 / 0 |
| Stejný test PPU s **původním** `ai_accel.v` | selže (62 místo 135, nejde restartovat) → chyba reprodukována a opravena |

**Neověřeno:** syntéza a časování ve Vivadu na VU47P a běh na reálném F2.
Pokus o orientační syntézu v Yosysu (`synth_xilinx -family xcup`) na
pracovní stanici skončil bez výsledku, nejspíš pro nedostatek paměti
(14 GB) při zploštění návrhu. Skutečné využití zdrojů ukáže až Vivado na AWS.
Konstanty SDK pro F2 viz `fpga/aws_f2/README.md`.

## 6. Opravy původního PPU (2026-11-03)

| Modul | Chyba | Oprava |
|---|---|---|
| `ai_accel.v` | po 1. operaci uvízl v `S_DONE`, další příkazy se neprovedly | návrat do `S_IDLE` |
| `ai_accel.v` | výsledek bez posledního součinu | `acc + prod` |
| `ai_accel.v` | čtení paměti o takt dřív, než byla data platná | čekací stavy (latence 2) |
| `core.v` | LW zapisoval data z předchozí instrukce (registrované rozhraní u jednocyklového jádra) | kombinační paměťové rozhraní |
| `core.v` | zápis do x0 při rd=0 | zápis jen pro rd≠0 |
| `core.v` | vlastní instrukce: rs2 kolidovalo s okamžitou hodnotou I-formátu | R-formát, délka ve funct7 |
| `axi_lite_slave.v` | čtení ignorovalo ARADDR | obal nad novým `axi_lite_mmio.v` |
| `ppu_top.v` | zápisy CPU do MMIO nevedly nikam; `integer` v nepojmenovaném bloku | směrování na akcelerátor; pojmenovaný blok |
| `tb_ppu.v` | nezapojené MMIO vstupy; testbench závislý na interních jménech starého Verilatoru | porty ven, test jen přes rozhraní |
| `fpga/fpga_top.v` | duplicitní modul `fpga_top` (kolize s `fpga/top/fpga_top.v`) | odstraněn (záloha mimo projekt) |

## 7. Cloud FPGA: AWS EC2 F2

Viz `fpga/aws_f2/README.md`: výběr, rozpočet a postup buildu.

## 8. Cesta ke škálování (za PoC)

1. **Transformer místo MLP:** blok attention s 1-bitovými (ternárními)
   projekcemi Q/K/V/O. Na FPGA se přidá softmax po řádcích (stejná LUT)
   a buffer klíčů a hodnot.
2. **Ternární váhy {−1, 0, +1} (BitNet b1.58):** podle publikovaných
   výsledků Microsoftu dosahují při stejné velikosti kvality FP16
   transformerů. V HW to stojí jen 1 bit masky navíc.
3. **Paměť HBM** (F2 má 16 GiB) pro modely nad kapacitu on-chip pamětí a pro
   delší kontext.
4. **Více lane / více jader:** výkon roste lineárně s počtem lane (64 → 512)
   a s replikací jader přes SLR.
5. **Mini-batch s akumulací gradientu** místo online SGD: stabilnější trénink
   větších modelů.

## 9. Diagnostika konceptu (2026-11-03): proč je BitLM horší než transformer

Všechny modely mají ~1,1–1,2 M parametrů a kontext 128 znaků. Trénink na
smíšeném českém korpusu (2,2 M znaků, 8 knih z Project Gutenberg + R.U.R.),
validace na posledních 10 % R.U.R. Data jsou v `results/cz_mix/`.

| Model | Val. entropie [bit/znak] | Přesnost | Paměť vah pro inferenci |
|---|---|---|---|
| trigram | 3,17 | – | – |
| BitLM, deterministické zaokrouhlení, nejlepší stav (4M kroků se rozpadl na 3,73) | 3,14 | 35 % | ~145 KiB |
| **BitLM + stochastické zaokrouhlení (RTL v3), 4M kroků** | **2,97** | **38,4 %** | ~145 KiB |
| stejná architektura, 1 bit, Adam + mini-batch (`concept_lab.py mlp1`) | 2,92 | 39,9 % | ~145 KiB |
| transformer, 4 bity po tréninku (PTQ) | 2,82 | 39,2 % | 545 KiB |
| **transformer 1 bit, trénovaný v 1 bitu (QAT, `bitnet1`)** | **2,58** | **45,0 %** | ~145 KiB |
| transformer ternární {−1,0,+1} (`bitnet158`) | 2,49 | 46,0 % | ~290 KiB |
| transformer float32 | 2,41 | 48,9 % | 4,3 MiB |

Závěry:
1. **Způsob učení v HW** byl největší brzdou. Deterministické zaokrouhlení
   ztrácelo malé opravy vah, takže výstupní vrstva ujela (96 % +1) a dlouhý
   trénink se rozpadl. Stochastické zaokrouhlení (RTL v3, bitově ověřeno)
   mezeru proti Adamu prakticky uzavřelo: 2,97 vs. 2,92.
2. **Zbývající rozdíl ~0,4 bitu je architektura.** Transformer s attention je
   při stejném počtu bitů o 0,4 lepší.
3. 1bitový transformer trénovaný přímo v 1 bitu je lepší než 4bitový
   transformer zmenšený po tréninku (2,58 vs. 2,82) při čtvrtinové paměti.

Další krok: trénink 1bitového (ternárního) transformeru v HW. Otevřená otázka
je, zda transformer zvládne celočíselné SGD se stochastickým zaokrouhlením
místo Adamu. Ověřit nejdřív v referenčním modelu.

## 10. Optimalizator pro trenink 1-bit transformeru v cipu

`tools/bitlm/concept_lab.py --chip --opt ...`, 1-bit transformer (d=128, 5 vrstev, 1,18M parametru),
kontext 128, cz_mix, 8000 kroku x 32 sekvenci, workflow `concept_scan.yml`, vysledky `results/cz_mix/gh/scan*`.
Vsechny parametry celociselne (int16 stin, krok = mocnina dvou), update se stochastickym zaokrouhlenim.

| optimalizator | bit/znak | presnost | pamet navic na 1-bit vahu |
|---|---|---|---|
| AdamW (float, reference) | 2,451 | 47,8 % | 2x fp32 |
| **Lion, moment int8, zpetny pruchod int8** | **2,530** | **45,8 %** | int16 stin + int8 moment |
| Lion (float moment) | 2,538 | 45,3 % | |
| afac (Lion + faktorizovana adaptivita radky/sloupce) | 2,540 | 44,1 % | + O(radky+sloupce) |
| bop (preklapeni bitu, bez stinu) | 2,700 | 41,5 % | jen int8 moment |
| signSGD (Lion bez momentu) | 2,760 | 40,3 % | int16 stin |
| Lion, moment int4 | 3,359 | 28,8 % | |
| SGD / normalizovane SGD (2000 kroku) | 3,13-3,60 | <=34 % | |

Zavery: prosty SGD (dnesni BitLM) na transformer nestaci; Lion s int8 momentem a int8 zpetnym
pruchodem je jen 0,08 bitu za Adamem -> do HW jde Lion (update = +-2^k, bez nasobeni/deleni).
Faktorizovana adaptivita nic neprinesla. Bop setri 16bitovy stin (1+8 bitu/vahu, ~2,5x vetsi model
ve stejne pameti) za 0,17 bitu - kandidat na pametove usporny rezim pri skalovani.

## 11. Zjednoduseni pro cip a pevna specifikace (2026-11-04)

Scany `concept_scan.yml` (8000 kroku, d=128-ish concept_lab, cz_mix, Lion int8, `--gq 8`);
vysledky v `results/cz_mix/gh/scan_hw`, `scan_hw2`, `scan_bits`, `scan_combo`.
Sum mezi seedy: vychozi model 2,530 / 2,562 (seed 1) -> rozdily do ~0,03 jsou sum.

| Varianta | CE (bit/znak) | Zaver |
|---|---|---|
| vychozi (LayerNorm, GELU, float attention, int8 vse) | 2,530 / 2,562 | reference |
| ReLU² / ReLU | 2,500 / 2,530 | ReLU² |
| int attention (int8 q,k,v, softmax 2^x tabulkou, p uint8) | 2,561 | ano |
| RMS bez zisku / RMS + uceny zisk | 2,601 / 2,522 | RMS + zisk |
| amax norma (i x4, i se ziskem) | 2,84-3,61 | ne |
| dW gradient int4 (tensor/token), ternarni | 2,63 / 2,585, 2,69-2,94 | ne |
| **dW gradient = znamenko x 2^e (8 urovni)** | 2,537 | **ano - posun misto nasobeni** |
| dW gradient 2^e jen 4 urovne | 2,63-2,65 | ne |
| aktivace 6 / 4 bity | 2,491 / 2,649 | 6 bitu |
| **cela sestava** (RMS+zisk, pow2 skaly, ReLU², int attention, dW 2^e, aktivace int6) | **2,549** | **= reference v ramci sumu** |
| totez bez int6 (2 seedy) | 2,553 / 2,558 | |

**Pevna specifikace pro golden C model a RTL:**
1-bit vahy `sign(w)*2^-k` (int16 stin), aktivace na vstupu BitLinear int6 per-token se skalou 2^n,
RMS norma (1/sqrt tabulkou ve VEC) + int zisk po kanalech, ReLU², attention celociselne,
zpetny pruchod: gradient pro `dx` int8 (pow2 absmax + SR), gradient pro `dW` znamenko x 2^e
(3bitovy exponent, nestranne SR) -> dW = akumulace posunutych aktivaci, zadne nasobicky;
optimalizator Lion s int8 momentem, update +-2^lr LSB se SR.
