#!/usr/bin/env bash
# exp-06: three reserved candidates on the full 1319-item test set, byte-identical
# flags, candidate -> inspect log captured as each finishes. The incumbent soup34's
# log from exp-05 is reused for the paired tests (same protocol, same flags).
set -euo pipefail
: > analysis/exp-06_logmap.txt
for spec in "soup234:/home/ben/task/ckpts/soup234/served" \
            "soup37:/home/ben/task/ckpts/soup37/served" \
            "exp04c1200:/home/ben/task/ckpts/exp04c1200/served"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "=== $name ==="
  bash run_eval.sh "$path" "exp-06_${name}_full" -1 || { echo "$name FAILED"; continue; }
  log=$(python -c "import json;print(json.load(open('analysis/exp-06_${name}_full_diag.json'))['log'])") || { echo "$name NOLOG"; continue; }
  echo "${name}=${log}" >> analysis/exp-06_logmap.txt
done
grep -E "^(exp-03|exp-04|soup34)=" analysis/exp-05_logmap.txt >> analysis/exp-06_logmap.txt
python pair_eval.py --pairs $(tr '\n' ' ' < analysis/exp-06_logmap.txt) --out analysis/exp-06_paired.json
