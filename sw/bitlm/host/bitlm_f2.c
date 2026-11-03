/*
 * BitLM na AWS EC2 F2: host program (bezi na F2 instanci, FPGA ve slotu 0).
 *
 *   sudo ./bitlm_f2 verify corpus.tok          # bitove presne overeni HW vs. reference
 *   sudo ./bitlm_f2 train  corpus.tok [kroku]  # trenink na FPGA, eval, generovani
 *
 * Preklad (po `source sdk_setup.sh` v aws-fpga):
 *   make -f Makefile.f2
 * Pristup: OCL AXI-Lite pres AppPF BAR0 (fpga_pci_peek/poke).
 * POZN.: konstanty FPGA_APP_PF a APP_PF_BAR0 pochazi z fpga_pci.h v aws-fpga
 * SDK; pred prvnim behem overit v aktualni verzi SDK (viz fpga/aws_f2/README.md).
 */
#include "bitlm_hw.h"

#include <fpga_mgmt.h>
#include <fpga_pci.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t f2_rd(void *ctx, uint32_t addr)
{
    uint32_t v = 0xFFFFFFFF;
    if (fpga_pci_peek(*(pci_bar_handle_t *)ctx, addr, &v))
        fprintf(stderr, "fpga_pci_peek 0x%x selhal\n", addr);
    return v;
}

static void f2_wr(void *ctx, uint32_t addr, uint32_t val)
{
    if (fpga_pci_poke(*(pci_bar_handle_t *)ctx, addr, val))
        fprintf(stderr, "fpga_pci_poke 0x%x selhal\n", addr);
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void evaluate(const bl_model *m, const uint8_t *c, uint32_t from, uint32_t to,
                     double *ce, double *acc)
{
    bl_fwd f;
    double s = 0;
    uint32_t ok = 0;
    for (uint32_t pos = from; pos < to; pos++) {
        bl_forward(m, c + pos - BL_N, &f);
        uint32_t p = f.p[c[pos]];
        s += -log2((p ? p : 1) / 65536.0);
        ok += f.argmax == c[pos];
    }
    *ce = s / (to - from);
    *acc = (double)ok / (to - from);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "pouziti: %s verify|train korpus.tok [kroku]\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 2; }
    static uint8_t c[BL_CORPUS_MAX];
    uint32_t len = (uint32_t)fread(c, 1, sizeof(c), f);
    fclose(f);

    if (fpga_mgmt_init()) { fprintf(stderr, "fpga_mgmt_init selhal\n"); return 1; }
    pci_bar_handle_t bar = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(0, FPGA_APP_PF, APP_PF_BAR0, 0, &bar)) {
        fprintf(stderr, "fpga_pci_attach selhal (je AFI nahrany? sudo fpga-describe-local-image -S 0)\n");
        return 1;
    }
    bitlm_dev d = { f2_rd, f2_wr, &bar, NULL };
    if (bitlm_probe(&d)) return 1;

    int rc = 0;
    if (!strcmp(argv[1], "verify")) {
        rc = bitlm_verify(&d, c, len, 1000, 1) ? 1 : 0;
    } else if (!strcmp(argv[1], "train")) {
        uint32_t steps = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 0) : 2000000;
        uint32_t split = len * 9 / 10, batch = 50000, decay_every = 500000;
        bl_hparams hp = bl_default_hparams();
        static bl_model m;
        bl_init(&m, 12345, hp);       /* jen kvuli hp a pro porovnani */

        bitlm_set_hparams(&d, &hp);
        bitlm_load_corpus(&d, c, split);
        if (bitlm_init(&d, 12345) != BL_ST_DONE) { fprintf(stderr, "INIT selhal\n"); return 1; }
        printf("%9s %9s %9s %12s %10s\n", "krok", "tr_acc", "tr_p", "kroku/s(HW)", "s(host)");
        double t0 = now();
        uint64_t total_cycles = 0;
        for (uint32_t done = 0; done < steps; done += batch) {
            bitlm_cmd(&d, BL_CMD_CLEAR_STATS, 1000);
            uint32_t n = steps - done < batch ? steps - done : batch;
            if (bitlm_train(&d, split, n) != BL_ST_DONE) { fprintf(stderr, "TRAIN selhal\n"); return 1; }
            bitlm_stats st;
            bitlm_read_stats(&d, &st);
            total_cycles += st.cycles;
            printf("%9u %9.4f %9.4f %12.0f %10.1f\n", done + n, (double)st.correct / st.steps,
                   (double)st.sum_pt / st.steps / 65536.0, st.steps / (st.cycles / 250e6), now() - t0);
            /* plan uceni stejny jako referencni beh: kazdych 500k kroku lr +1 (max 3x) */
            if ((done + n) % decay_every == 0 && (done + n) / decay_every <= 3) {
                hp.lr1++; hp.lr2++; hp.lre++; hp.lrb++;
                bitlm_set_hparams(&d, &hp);
            }
        }
        printf("trenink: %u kroku, %.2f s cisteho HW casu pri 250 MHz, celkem %.1f s vcetne hosta\n",
               steps, total_cycles / 250e6, now() - t0);
        bitlm_read_model(&d, &m);
        m.hp = hp;
        double ce, acc;
        evaluate(&m, c, split, len, &ce, &acc);
        printf("validace (vahy z FPGA, vyhodnoceno na CPU): CE %.4f bit/znak, presnost %.4f\n", ce, acc);
        uint8_t ctx[BL_N], out[600];
        memcpy(ctx, c + split, BL_N);
        bitlm_generate(&d, ctx, sizeof(out), out);
        printf("--- generovano na FPGA:\n");
        for (unsigned i = 0; i < sizeof(out); i++) putchar(BL_VOCAB[out[i]]);
        printf("\n---\n");
        FILE *wf = fopen("bitlm_f2_weights.bin", "wb");
        if (wf) {
            fwrite(m.w1, sizeof(m.w1), 1, wf);
            fwrite(m.w2, sizeof(m.w2), 1, wf);
            fwrite(m.emb, sizeof(m.emb), 1, wf);
            fclose(wf);
        }
    } else {
        fprintf(stderr, "neznamy prikaz %s\n", argv[1]);
        rc = 2;
    }
    fpga_pci_detach(bar);
    return rc;
}
