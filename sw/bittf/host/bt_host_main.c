/* Spolecny beh tréninku na FPGA: nahrani obrazu, kroky, soubezna kontrola proti golden modelu. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bt_host.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int bt_host_main(bt_dev *d, int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "pouziti: %s <prefix> <korpus.tok> [--steps N] [--total-steps 8000] [--seed 12345] "
                        "[--lr-shift 4] [--check K] [--dump soubor]\n", argv[0]);
        return 2;
    }
    int steps = 3, total = 8000, lr = 4, check = 1;
    unsigned seed = 12345;
    const char *dump = NULL;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--total-steps")) total = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = (unsigned)strtoul(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--lr-shift")) lr = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--check")) check = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--dump")) dump = argv[i + 1];
        else if (strncmp(argv[i], "+", 1)) {
            fprintf(stderr, "neznama volba %s\n", argv[i]);
            return 2;
        }
    }
    int rc = bt_host_probe(d, 1);
    if (rc) {
        fprintf(stderr, rc == 1 ? "spatne ID zarizeni (nahrany jiny AFI?)\n" : "DDR neni pripravena\n");
        return 1;
    }
    bt_layout L;
    if (bt_layout_load(&L, argv[1]))
        return 1;
    bt_cpu c;
    bt_rng rng;
    bt_model m;
    uint8_t *data = bt_dev_init(&c, &L, argv[1], argv[2], &m, &rng, seed, lr);
    double t0 = now();
    bt_host_load(d, &L, &c);
    printf("obraz %.1f MB nahran za %.2f s\n", L.memsize / 1e6, now() - t0);
    long split = L.corpus_len - L.val_len;
    int B = L.B, Tn = L.T, N = B * Tn;
    bt_work *w = check ? bt_work_new(&m, B) : NULL;
    uint8_t *x = malloc((size_t)N), *y = malloc((size_t)N);
    uint32_t *r = malloc(sizeof(uint32_t) * (size_t)B);
    int64_t lr_shift = lr;
    double tdev = 0, tgold = 0;
    uint64_t cyc = 0;
    int bad = 0;
    for (int step = 1; step <= steps && !bad; step++) {
        if (step == total / 2 || step == total * 3 / 4 || step == total * 7 / 8)
            lr_shift--;
        double ts = now();
        uint32_t cy = 0;
        int err = bt_host_step(d, &L, lr_shift, &cy);
        tdev += now() - ts;
        cyc += cy;
        if (err) {
            fprintf(stderr, "krok %d: chyba jadra %d\n", step, err);
            bad = 1;
            break;
        }
        if (check) {
            /* golden: stejny vyber davky (RNG) a stejny krok */
            ts = now();
            m.lr_shift = (int)lr_shift;
            bt_rng_draw(&rng, B, r);
            for (int b = 0; b < B; b++) {
                long ix = Tn + (long)(r[b] % (uint32_t)(split - 1 - Tn));
                memcpy(x + (long)b * Tn, data + ix - Tn, (size_t)Tn);
                memcpy(y + (long)b * Tn, data + ix - Tn + 1, (size_t)Tn);
            }
            int64_t *lg;
            int *el;
            bt_forward(&m, w, x, 1, &lg, &el);
            bt_backward_update(&m, w, y, &rng, 0, step);
            tgold += now() - ts;
            if (step % check == 0 || step == steps) {
                uint32_t h1 = bt_host_hash(d, &L, &m), h2 = bt_state_hash(&m, &rng);
                printf("krok %d: FPGA vs golden %s (%08x/%08x), %u taktu\n", step, h1 == h2 ? "SHODA" : "NESHODA", h1,
                       h2, cy);
                if (h1 != h2)
                    bad = 1;
            }
        } else if (step % 100 == 0 || step == steps) {
            printf("krok %d, %.3f s/krok, %.1f M taktu/krok\n", step, tdev / step, cyc / 1e6 / step);
        }
        fflush(stdout);
    }
    printf("FPGA %.3f s/krok (%.1f M taktu)%s", tdev / steps, cyc / 1e6 / steps, check ? "" : "\n");
    if (check)
        printf(", golden CPU %.3f s/krok\n", tgold / steps);
    if (dump && !bad) {
        if (bt_host_dump(d, &L, dump)) {
            fprintf(stderr, "zapis %s selhal\n", dump);
            bad = 1;
        } else
            printf("stav ulozen do %s\n", dump);
    }
    printf(bad ? "VYSLEDEK: CHYBA\n" : "VYSLEDEK: OK\n");
    return bad;
}
