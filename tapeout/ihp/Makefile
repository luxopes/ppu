# BitTF mini na IHP SG13G2 (LibreLane). Spoustet v nix shellu: nix develop ./tapeout/ihp
MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PDK_ROOT ?= $(MAKEFILE_DIR)/IHP-Open-PDK
PDK ?= ihp-sg13g2
PDK_COMMIT ?= 89d42ee6d2fd9b50b86a1aada6fee6673c3e6008
DIE ?= 1414
MARGIN ?= 330
DENS ?= 70

$(PDK_ROOT)/$(PDK):
	ciel enable $(PDK_COMMIT) --pdk-root $(PDK_ROOT) --pdk-family ihp-sg13

clone-pdk: $(PDK_ROOT)/$(PDK)

# konfigurace s rozmery (DIE = strana cipu v um vcetne tesnici obruby, MARGIN = pady + obruba + prstenec napajeni)
librelane/config.gen.yaml: librelane/config.yaml
	python3 -c "import sys; d=$(DIE); m=$(MARGIN); c1=d-m; x=m+15; \
	s=open('librelane/config.yaml').read(); \
	r={'@DIE@':d,'@C0@':m,'@C1@':c1,'@DENS@':$(DENS),'@M0X@':x,'@M0Y@':x,'@M1Y@':x+72,'@M2Y@':x+144,'@M3Y@':x+216}; \
	[s:=s.replace(k,str(v)) for k,v in r.items()]; open('librelane/config.gen.yaml','w').write(s)"

librelane: $(PDK_ROOT)/$(PDK) librelane/config.gen.yaml
	librelane librelane/config.gen.yaml --pdk $(PDK) --pdk-root $(PDK_ROOT) --manual-pdk --save-views-to final/

librelane-nodrc: $(PDK_ROOT)/$(PDK) librelane/config.gen.yaml
	librelane librelane/config.gen.yaml --pdk $(PDK) --pdk-root $(PDK_ROOT) --manual-pdk --save-views-to final/ \
	  --skip KLayout.DRC --skip Magic.DRC --skip KLayout.Antenna --skip KLayout.Density

.PHONY: clone-pdk librelane librelane-nodrc librelane/config.gen.yaml
