#!/bin/sh
# Overeni bitove shody BitTF golden modelu (C) s intsim.py: kontrolni soucty po kazdem kroku.
# Pouziti: tools/bitlm/check_bittf_golden.sh [python] [korpus]
set -e
PY=${1:-python3}
CORPUS=${2:-data/cz_mix.tok}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
gcc -O2 -Wall -Wextra -fopenmp -o "$D/bittf_train" sw/bittf/golden/bittf_ref.c sw/bittf/golden/bittf_train.c -lm
run() {  # $1 = nazev, zbytek = argumenty
    n=$1; shift
    "$D/bittf_train" "$CORPUS" "$@" | grep -E '^(trace|hash)' > "$D/c.$n"
    $PY tools/bitlm/intsim.py "$CORPUS" "$@" --out "$D/p.json" 2>/dev/null | grep -E '^(trace|hash)' > "$D/p.$n"
    if [ -s "$D/c.$n" ] && cmp -s "$D/c.$n" "$D/p.$n"; then
        echo "$n: SHODA ($(wc -l < "$D/c.$n") kontrolnich souctu)"
    else
        echo "$n: NESHODA"; diff "$D/c.$n" "$D/p.$n" | head -20; exit 1
    fi
}
run maly --d 64 --layers 2 --heads 2 --ff 128 --batch 4 --steps 3 --trace 3 --eval-every 1000000
run plny --steps ${STEPS:-40} --hash-every 5 --eval-every 1000000
