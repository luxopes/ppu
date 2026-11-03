#!/bin/sh
# Paralelni sweep hyperparametru referencniho modelu (1 vlakno na beh).
# Pouziti: [JOBS=8] [BIN=build/bitlm_train] [CORPUS=data/rur.tok] [EXTRA="-v 11152 -d 100000"] \
#          tools/bitlm/sweep.sh KROKU "s1,s2,...,flags" ["..."] ...
# Vystup: radek "val_CE val_acc hparams" serazeny podle val_CE.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build/bitlm_train}
CORPUS=${CORPUS:-$ROOT/data/rur.tok}
N=$1; shift
export OMP_NUM_THREADS=1
for p in "$@"; do echo "$p"; done | xargs -P "${JOBS:-$(nproc)}" -I{} sh -c \
  "\"$BIN\" -c \"$CORPUS\" -n $N -e $N -g 0 ${EXTRA:-} -p {} | awk -v p={} '/^ *$N /{print \$4, \$5, p}'" \
  | sort -n
