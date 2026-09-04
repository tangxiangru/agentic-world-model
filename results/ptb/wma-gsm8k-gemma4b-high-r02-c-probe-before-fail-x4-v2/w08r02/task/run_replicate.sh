#!/usr/bin/env bash
# exp-07: a second independent full-test read of the two top candidates, so the
# choice between them is made on a two-run mean instead of on one read whose
# batch-composition noise (exp-05) is larger than the gap.
set -euo pipefail
: > analysis/exp-07_logmap.txt
for spec in "soup234r2:/home/ben/task/ckpts/soup234/served" \
            "soup34r2:/home/ben/task/ckpts/soup34/served"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "=== $name ==="
  bash run_eval.sh "$path" "exp-07_${name}_full" -1 || { echo "$name FAILED"; continue; }
  log=$(python -c "import json;print(json.load(open('analysis/exp-07_${name}_full_diag.json'))['log'])") || { echo "$name NOLOG"; continue; }
  echo "${name}=${log}" >> analysis/exp-07_logmap.txt
done
grep -E "^soup34=" analysis/exp-05_logmap.txt >> analysis/exp-07_logmap.txt
grep -E "^soup234=" analysis/exp-06_logmap.txt >> analysis/exp-07_logmap.txt
python pair_eval.py --pairs $(tr '\n' ' ' < analysis/exp-07_logmap.txt) --out analysis/exp-07_paired.json
