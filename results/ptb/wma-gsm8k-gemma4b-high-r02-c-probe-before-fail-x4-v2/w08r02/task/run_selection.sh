#!/usr/bin/env bash
# exp-05: score every surviving candidate on the full 1319-item test set with
# byte-identical flags, recording each candidate's inspect log path as it lands.
set -euo pipefail
: > analysis/exp-05_logmap.txt
for spec in "exp-02:/home/ben/task/ckpts/exp-02/served" \
            "exp-03:/home/ben/task/ckpts/exp-03/served" \
            "exp-04:/home/ben/task/ckpts/exp-04/served" \
            "soup34:/home/ben/task/ckpts/soup34/served"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "=== $name ==="
  bash run_eval.sh "$path" "exp-05_${name}_full" -1
  log=$(python -c "import json;print(json.load(open('analysis/exp-05_${name}_full_diag.json'))['log'])")
  echo "${name}=${log}" >> analysis/exp-05_logmap.txt
done
python pair_eval.py --pairs $(tr '\n' ' ' < analysis/exp-05_logmap.txt) --out analysis/exp-05_paired.json
