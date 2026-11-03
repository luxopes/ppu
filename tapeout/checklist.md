# Tapeout checklist (minimal starter)

Tento soubor obsahuje minimální seznam artefaktů a kroků potřebných pro přípravu na tapeout.

- PDK artefakty: LEF (stdcells + IO), Liberty (.lib), SPICE/RC corner models
- Padframe a IO LEF, pinout (pin map)
- UPF/CPF pro power intent (viz `tapeout/power.upf`)
- SDC timing constraints (viz `tapeout/constraints.sdc`)
- Synthesis netlist (gate-level) a SDF pro back-annotation
- DFT: scan insertion report, ATPG vectors, JTAG/BIST hooks
- P&R outputs: LEF/DEF, placed netlist, routing database
- Signoff: LVS/DRC runlists, DRC/LVS reports, DFM checks
- Power signoff: IR drop and EM reports
- Final GDS/OASIS, mask set spec, signoff report bundle
- Release package README (pins, die size, reticle info)

Poznámka: nahraďte šablony a placeholdery skutečnými soubory získanými z PDK a týmových EDA skriptů.
