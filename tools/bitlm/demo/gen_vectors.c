/* Testovaci vektory pro JS inferenci (test_bitlm_js.cjs): z[] a argmax z C modelu
 * pro nekolik kontextu z korpusu, vahy ze snimku (bitlm_train -S).
 * Pouziti: gen_vectors snap.bin korpus.tok > vectors_<krok>.json */
#include "../../../sw/bitlm/golden/bitlm_ref.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "pouziti: %s snap.bin korpus.tok\n", argv[0]); return 2; }
    static bl_model m; static uint8_t c[BL_CORPUS_MAX];
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; }
    if (fread(m.w1, sizeof m.w1, 1, f) != 1 || fread(m.w2, sizeof m.w2, 1, f) != 1 || fread(m.emb, sizeof m.emb, 1, f) != 1) { fprintf(stderr, "kratky soubor\n"); return 2; }
    fclose(f);
    m.hp = bl_default_hparams();
    f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 2; }
    size_t len = fread(c, 1, sizeof c, f); fclose(f);
    printf("[");
    for (int t = 0; t < 6; t++) {
        size_t pos = 1000 + t * 17000; if (pos >= len) pos = len - 1;
        bl_fwd fw; bl_forward(&m, c + pos - BL_N, &fw);
        printf("%s{\"ctx\":[", t ? "," : "");
        for (int n = 0; n < BL_N; n++) printf("%s%d", n ? "," : "", c[pos - BL_N + n]);
        printf("],\"z\":[");
        for (int i = 0; i < BL_V; i++) printf("%s%d", i ? "," : "", fw.z[i]);
        printf("],\"argmax\":%d}", fw.argmax);
    }
    printf("]\n");
}
