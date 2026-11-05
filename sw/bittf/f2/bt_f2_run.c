/*
 * bt_f2_run - trénink BitTF na AWS F2: registry jadra pres BAR0 (OCL), pamet jadra = DDR pres BAR4
 * (PCIS, write-combining). Logika behu je spolecna se simulaci (sw/bittf/host/bt_host_main.c).
 *
 * sudo ./bt_f2_run <prefix programu> <korpus.tok> [--steps N] [--check K] [--dump soubor] [+slot S]
 * (AFI musi byt nahrany: fpga-load-local-image -S 0 -I agfi-...)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fpga_mgmt.h>
#include <fpga_pci.h>

#include "../host/bt_host.h"

typedef struct {
    pci_bar_handle_t ocl, mem;
    uint8_t *mp;      /* mapovani BAR4 (write-combining) */
    uint64_t mlen;
} f2_dev;

static void die(const char *m)
{
    fprintf(stderr, "F2: %s\n", m);
    exit(1);
}

static void f2_wr32(void *c, uint32_t off, uint32_t v)
{
    if (fpga_pci_poke(((f2_dev *)c)->ocl, off, v))
        die("zapis registru selhal");
}

static uint32_t f2_rd32(void *c, uint32_t off)
{
    uint32_t v;
    if (fpga_pci_peek(((f2_dev *)c)->ocl, off, &v))
        die("cteni registru selhalo");
    return v;
}

static void f2_memwr(void *c, uint64_t a, const void *s, uint64_t n)
{
    f2_dev *d = c;
    if (a + n > d->mlen)
        die("zapis mimo mapovanou pamet");
    memcpy(d->mp + a, s, n);
    __sync_synchronize();
    /* Pojistka poradi: start jde pres BAR0 (jina cesta v shellu) a AXI nezarucuje poradi cteni a zapisu.
     * Posledni zapsane bajty proto cteme zpet, dokud nesedi - pak jsou zapisy v DDR. Host behem behu
     * jadra na BAR4 nesmi sahat (PCIS ma timeout 8 us a pak je potreba AFI nahrat znovu). */
    uint64_t t = (a + n) >= 8 && n >= 8 ? a + n - 8 : a;
    uint64_t m = (a + n) - t;
    for (int i = 0;; i++) {
        uint8_t b[8];
        for (uint64_t k = 0; k < m; k++)
            b[k] = ((volatile uint8_t *)d->mp)[t + k];
        if (!memcmp(b, (const uint8_t *)s + (t - a), m))
            break;
        if (i > 100000)
            die("zapis do DDR se neprojevil");
    }
}

static void f2_memrd(void *c, uint64_t a, void *dst, uint64_t n)
{
    f2_dev *d = c;
    if (a + n > d->mlen)
        die("cteni mimo mapovanou pamet");
    /* po 8 bajtech (cteni z BAR jsou necachovana) */
    uint64_t i = 0;
    uint8_t *o = dst;
    for (; i < n && ((a + i) & 7); i++)
        o[i] = ((volatile uint8_t *)d->mp)[a + i];
    for (; i + 8 <= n; i += 8) {
        uint64_t v = *(volatile uint64_t *)(d->mp + a + i);
        memcpy(o + i, &v, 8);
    }
    for (; i < n; i++)
        o[i] = ((volatile uint8_t *)d->mp)[a + i];
}

static void f2_idle(void *c)
{
    (void)c;
    usleep(200);
}

int main(int argc, char **argv)
{
    int slot = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "+slot"))
            slot = atoi(argv[i + 1]);
    if (fpga_mgmt_init())
        die("fpga_mgmt_init selhal (bezi jako root? je nainstalovane SDK?)");
    f2_dev d;
    memset(&d, 0, sizeof d);
    d.ocl = PCI_BAR_HANDLE_INIT;
    d.mem = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &d.ocl))
        die("pripojeni BAR0 selhalo");
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &d.mem))
        die("pripojeni BAR4 selhalo");
    /* pamet jadra: obraz z layoutu (+ rezerva), mapujeme od 0 */
    bt_layout L;
    if (argc < 2 || bt_layout_load(&L, argv[1]))
        die("nelze nacist layout");
    d.mlen = (L.memsize + 4095) & ~(uint64_t)4095;
    void *p;
    if (fpga_pci_get_address(d.mem, 0, d.mlen / 4, &p))
        die("mapovani BAR4 selhalo");
    d.mp = p;
    bt_dev dev = {&d, f2_wr32, f2_rd32, f2_memwr, f2_memrd, f2_idle};
    int rc = bt_host_main(&dev, argc, argv);
    fpga_pci_detach(d.mem);
    fpga_pci_detach(d.ocl);
    return rc;
}
