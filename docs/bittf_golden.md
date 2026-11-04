# BitTF golden model: celociselna aritmetika treninku

Referencni definice toho, co presne pocita cip pri treninku 1-bit transformeru. Dve shodne implementace:

- `tools/bitlm/intsim.py` - Python (torch int64), rychla zmena specifikace a mereni kvality;
- `sw/bittf/golden/bittf_ref.c` - C golden model pro RTL (Verilator) a pozdeji ovladac.

Shodu obou overuje `tools/bitlm/check_bittf_golden.sh`: po kazdem kroku kontrolni soucty logitu,
gradientu vsech tenzoru a stavu (vahy, moment, RNG). RTL bude porovnavano s C stejne.

Specifikace modelu (vysledek scanu, `docs/bitlm_poc.md` kap. 11): 1-bit vahy, aktivace int6,
RMS norma + zisk, ReLU², celociselna attention, Lion s int8 momentem, gradient pro `dW` znamenko x 2^k.

## 1. Zapis cisel

Kazdy tenzor = **mantisy (cela cisla) + exponent** (mocnina dvou), hodnota = `m * 2^e`. Exponent je
bud jeden pro cely tenzor, nebo jeden na token (radek). V cipu je exponent jen posun.

Zakladni operace (vsechny v `int64`):

| Operace | Definice |
|---|---|
| `rsr(a, s)` | `s > 0`: `(a + 2^(s-1)) >> s` (zaokrouhleni pul nahoru, aritmeticky posun); `s <= 0`: `a << -s` |
| `sr(a, s, r)` | stochasticke zaokrouhleni: `(a + (r mod 2^s)) >> s`, `r` = 32bitove nahodne cislo |
| `bitlen(x)` | pocet bitu `x >= 0` (0 pro 0) |
| `quant_tok(y, b)` | po radcich: `s = bitlen(max|y|) - (b-1)`, `clamp(rsr(y, s), -2^(b-1), 2^(b-1)-1)` |
| `quant_sr(m, b)` | po tenzoru: totez s `sr` misto `rsr` (nestranne) |
| `normalize(m, b)` | po tenzoru: posun tak, aby `max|m|` melo `b` bitu (`rsr`) |
| `pow2q(m)` | `m` normalizovane na 16 bitu, `lo = 2^(bitlen(max)-7)`; prvek -> `znamenko * 2^k`, k = 0..7, nestranne mezi sousednimi mocninami (pod `lo`: 0 nebo `lo`) |

**Nahodna cisla** (stochasticke zaokrouhleni, inicializace, vyber davky): kazde volani = novy proud
s citacem `ctr`: `key = fmix32(seed ^ fmix32(ctr))`, prvek `i` dostane `fmix32(i * 0x9E3779B1 + key)`
(vse mod 2^32, `fmix32` z MurmurHash3). Poradi volani je pevne (viz kod) - stejne v C i RTL.

## 2. Parametry

Vsechny jsou **int16 stin** `q` (clamp +-32767):

| Parametr | Pocatecni hodnota | Pouziti dopredu |
|---|---|---|
| vahy BitLinear | uniformne -4096..4095 | `sign(q)` (q >= 0 -> +1), vystup * 2^-k, `k = round(log2(fan_in)/2)` |
| embedding tok, pos | ~N(0; 0,02) v LSB 2^-16 (soucet 4 uniformnich) | primo int16 (Q16) |
| zisk normy | 4096 (= 1,0 v LSB 2^-12) | nasobeni po kanalech |

## 3. Dopredny pruchod (na token, B sekvenci po T)

- **Rezidualni proud** `x`: int32, pevne Q16 (`x = tok[id] + pos[t]`).
- **RMS norma:**
  1. `ss = sum x²`; `sh = bitlen(ss) - 16` zaokrouhleno nahoru na sude.
  2. Index `(ss >> sh) >> 8` (64..255) do tabulky `RSQ` (1/sqrt, 16 bitu).
  3. `rr = rsr(sqd * RSQ, 16)` (`sqd = round(sqrt(d) * 4096)`), `x^ = rsr(x * rr, 6 + sh/2)` (Q12).
  4. Vystup `x^ * g` (Q24) -> `quant_tok(., 6)`.
- **BitLinear:** `y = sum a * sign(W)` (jen scitani), exponent vystupu = exponent vstupu - k.
- **Attention** (`dh = d/H`, skore * 2^-ks, `ks = (bitlen(dh)-1)/2`):
  1. q po tokenech int8; k, v po (sekvence, hlava) int8 (zarovnani tokenu na max exponent, pak `quant`).
  2. Skore `q·k` -> pevna radova carka Q5: `rsr(sc, -(eq + ek - ks) - 5)`, kauzalni maska.
  3. `e = EXP8[min(max - s, 255)]` (tabulka `round(255 e^(-i/32))`).
  4. `p = (e * round(2^24 / sum e) + 2^15) >> 16`, clamp 255 (uint8, skala 2^-8).
  5. `y = p·v`, hlavy na spolecny exponent sekvence, `quant_tok(., 6)` -> proj.
- **MLP:** `fc1` -> `ReLU²` (`r = max(o, 0)`, `r*r`) -> `quant_tok(., 6)` -> `fc2`.
- **Pricteni do proudu:** `x += rsr(vystup, -(exponent) - 16)`, kontrola rozsahu int32.
- **Hlava:** `ln_f` -> `quant_tok(., 8)` -> logity = `n8 · tok^T` (svazane s embeddingem).

## 4. Zpetny pruchod

- **Softmax + CE:**
  1. Logity do Q8, rozdil od maxima -> Q5 index do `EXP16` (16 bitu).
  2. `p = e * round(2^31/sum e) >> 16` (Q15).
  3. `D = p - onehot * 2^15`, exponent `-15 - log2(B*T)`.
- **BitLinear zpet:** gradient vystupu `normalize(., 16)`, pak:
  - `dx`: `quant_sr(., 8)` -> `sum g * sign(W)` (jen scitani, pole cte W transponovane);
  - `dW`: `pow2q` -> `sum_t (znamenko * 2^k) * a_t`, kde `a_t` jsou int6 aktivace zarovnane na `max(ea) - 8` -> **jen posun a pricteni, zadne nasobicky**.
- **RMS norma zpet:**
  1. `dg = sum dy·x^`.
  2. `t = dy·g - x^ · round(mean(dy·g·x^))`.
  3. `dx = t * rr`, exponent po tokenech `e - 14 - sh/2`, zarovnani na maximum.
- **ReLU² zpet:** `dfo = dh * r` (exponent +1 za dvojku).
- **Attention zpet:**
  1. `dy` int8 (SR).
  2. `dv = p^T dy`, `dp = dy v^T`, `ds = p (dp - rsr(sum p dp, 8))`.
  3. `ds` int8 (SR).
  4. `dq = ds k`, `dk = ds^T q` (q zarovnane na max - 8). Vse na spolecny exponent -> gradient `qkv`.
- **Rezidualni proud zpet:** soucet BFP s 40 bity rezervy pod vetsim vrcholem.
- **Embedding:** gradient hlavy + soucet radku podle tokenu; pozice = soucet pres davku.

## 5. Lion (int8 moment)

Pro kazdy parametr:
1. Zarovnat gradient `g` a moment `m` na `max(vrchol) - 30`.
2. Aktualizace vahy: `u = sign(7m + g)` (b1 = 7/8), `q -= u * 2^lr` (lr = 4, pulen v 50/75/88 %).
3. Novy moment: `m = 127m + g` (b2 = 127/128, exponent -7) -> `quant_sr(., 8)`.

## 6. Overeni a vysledky

- `check_bittf_golden.sh`: maly model po 3 krocich (vsechny kontrolni soucty) a plna velikost
  40 kroku (hash stavu kazdych 5 kroku) - **C a Python bitove shodne**.
- Kvalita (8000 kroku, cz_mix, stejne jako scany): workflow `intsim.yml`, vysledek v `results/cz_mix/gh/intsim`.
