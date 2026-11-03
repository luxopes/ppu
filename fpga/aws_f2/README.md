# BitLM na AWS EC2 F2: výběr, rozpočet a postup

## Proč AWS F2

| Možnost | FPGA | Cena | Stav / poznámka |
|---|---|---|---|
| **AWS EC2 F2 `f2.6xlarge`** | 1× AMD Virtex UltraScale+ **VU47P** + 16 GiB HBM | **$1.98/h** on-demand (us-east-1); spot cca $0.6/h | aktivně podporováno; Vivado v FPGA Developer AMI; vybráno |
| AWS EC2 F1 | VU9P | – | ukončeno 20. 12. 2025 |
| Azure NP-series | Alveo U250 | – | ukončení 31. 5. 2027; nové rezervace od 2. 4. 2026 nejdou; nevhodné pro nový projekt |
| Alibaba Cloud f3 | 2× VU9P | cenu se nepodařilo dohledat | orientace na čínské regiony; zbytečné riziko |

Hlavní důvody pro F2:

* **Platí se jen za běh na FPGA.** Syntéza a implementace (Vivado) běží na
  levné instanci bez FPGA s FPGA Developer AMI. Vlastní licence Vivada není
  potřeba.
* Platforma má budoucnost: VU47P a HBM stačí i pro další fáze (transformer,
  větší modely).
* Shell (rozhraní OCL AXI4-Lite, 250 MHz `clk_main_a0`) přesně odpovídá
  rozhraní `bitlm_fpga_top`.

### Rozpočet PoC (odhad)

| Položka | Množství | Cena |
|---|---|---|
| Build instance (AWS doporučuje 4 vCPU / 32 GiB, např. r6i.xlarge, cca $0.25/h) | 3–4 buildy × 1,5 h | ≈ $2 |
| F2 `f2.6xlarge` on-demand | 4–6 h (nahrání AFI, verify, trénink, demo) | ≈ $8–12 |
| S3 (DCP, logy) | < 1 GB | < $1 |
| **Celkem** | | **≈ $15–20**, s rezervou do $50 |

S instancí F2 ve spotu je cena zhruba třetinová. Ceny je potřeba před
objednáním ověřit v AWS kalkulačce. Ceny build instancí jsou orientační.

**Předem nutné:** požádat o navýšení kvóty vCPU pro F-instance (výchozí
kvóta bývá 0) v regionu us-east-1 nebo us-west-2. Schválení může trvat
několik dní, proto je dobré zažádat hned.

## Postup (aws-fpga, větev `f2`)

Ověřeno proti dokumentaci F2 HDK (Vivado 2025.2, Small Shell, VU47P).

1. Na build instanci s **FPGA Developer AMI** (1.19.x, Vivado 2025.2):
   ```sh
   git clone https://github.com/aws/aws-fpga.git && cd aws-fpga && git checkout f2
   source hdk_setup.sh
   cd hdk/cl/examples && ./create_new_cl.py --new_cl_name cl_ppu_bitlm
   cd cl_ppu_bitlm && export CL_DIR=$PWD
   ```
2. Zkopírovat RTL do `$CL_DIR/design`:
   `rtl/accelerator/bitlm_core.v`, `rtl/accelerator/bitlm_exp_lut.vh`,
   `fpga/axi/axi_lite_mmio.v`, `fpga/top/bitlm_fpga_top.v`.
3. V `$CL_DIR/design/cl_ppu_bitlm.sv` (kopie CL_TEMPLATE) **odstranit tie-off
   OCL** (`cl_ocl_awready = 'b0; ... cl_ocl_rdata = 'b0;`) a místo něj vložit
   obsah `design/cl_ppu_bitlm_ocl.svh`. Pokud by tie-off zůstal, čtení
   z hosta nikdy neskončí (`cl_ocl_rvalid = 0`). Ostatní rozhraní a `sh_ddr`
   nechat podle šablony.
4. Doplnit nové soubory do seznamu zdrojů v build skriptech šablony
   (`$CL_DIR/build/scripts`, synth skript). Konkrétní soubor se má zkontrolovat
   v aktuální šabloně.
5. Build:
   ```sh
   cd $CL_DIR/build/scripts && ./aws_build_dcp_from_cl.py -c cl_ppu_bitlm
   ```
   Výstup: `$CL_DIR/build/checkpoints/<tag>.Developer_CL.tar`. Přípona
   `.VIOLATED` znamená nesplněné časování. Pak se sníží `LOG2_L` nebo
   přidá pipeline stupeň do sčítacího stromu.
6. AFI:
   ```sh
   aws s3 mb s3://<bucket> && aws s3 cp <tag>.Developer_CL.tar s3://<bucket>/dcp/
   aws ec2 create-fpga-image --name cl_ppu_bitlm --input-storage-location Bucket=<bucket>,Key=dcp/<tag>.Developer_CL.tar \
       --logs-storage-location Bucket=<bucket>,Key=logs/ --region us-east-1
   aws ec2 describe-fpga-images --fpga-image-ids afi-... --region us-east-1   # čekat na "available"
   ```
7. Na F2 instanci:
   ```sh
   cd aws-fpga && source sdk_setup.sh
   sudo fpga-clear-local-image -S 0
   sudo fpga-load-local-image -S 0 -I agfi-...
   cd <PPU>/sw/bitlm/host && make -f Makefile.f2
   sudo ./bitlm_f2 verify ../../../data/rur.tok      # bitová shoda HW vs. reference
   sudo ./bitlm_f2 train  ../../../data/rur.tok      # trénink 2M kroků + generování
   ```

## Ještě neověřeno

* Konstanty `FPGA_APP_PF` a `APP_PF_BAR0` (`fpga_pci.h`): názvy pocházejí ze
  SDK, ale z aktuální hlavičky nebyly přečteny. Ověřit při prvním překladu.
* Velikost OCL BAR: dokumentace uvádí 32 i 64 MiB. Mapa BitLM potřebuje jen
  1,1 MiB, takže na tom nezáleží.
* Využití zdrojů a časování na 250 MHz: ukáže až build ve Vivadu. Kritická je
  pravděpodobně sčítací stromová logika přes 64 lane a výběr `dh` z 512
  registrů.
