/* Vyhodnoceni snimku vah BitLM na validacni casti korpusu (posledni 10 %).
 * Pouziti: eval_snapshot korpus.tok snap.bin  ->  "ce acc" */
#include "../../sw/bitlm/golden/bitlm_ref.h"
#include <math.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    static bl_model m;
    static uint8_t c[BL_CORPUS_MAX];
    if (argc != 3) { fprintf(stderr, "pouziti: %s korpus.tok snap.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    uint32_t len = (uint32_t)fread(c, 1, sizeof(c), f);
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 2; }
    if (fread(m.w1, sizeof(m.w1), 1, f) != 1 || fread(m.w2, sizeof(m.w2), 1, f) != 1 ||
        fread(m.emb, sizeof(m.emb), 1, f) != 1) { fprintf(stderr, "kratky soubor\n"); return 2; }
    fclose(f);
    m.hp = bl_default_hparams();
    uint32_t split = len * 9 / 10, ok = 0;
    double ce = 0;
    bl_fwd fw;
    for (uint32_t pos = split; pos < len; pos++) {
        bl_forward(&m, c + pos - BL_N, &fw);
        uint32_t p = fw.p[c[pos] & 31];
        ce += -log2((p ? p : 1) / 65536.0);
        ok += fw.argmax == (c[pos] & 31);
    }
    printf("%.5f %.5f\n", ce / (len - split), (double)ok / (len - split));
    return 0;
}
