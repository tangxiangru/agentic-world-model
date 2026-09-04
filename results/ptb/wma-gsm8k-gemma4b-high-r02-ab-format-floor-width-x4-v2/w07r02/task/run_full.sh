set -u
for pair in "ckpts/exp-04/final exp-04_final" "ckpts/exp-02/final exp-02_final"; do
  set -- $pair
  M=$1; T=$2
  echo "=== FULL $T ==="
  python evaluate.py --model-path "$M" --limit -1 --max-tokens 4000 --max-connections 16 \
    --gpu-memory-utilization 0.85 --json-output-file "eval/${T}_full1319.json" > "logs/evalfull_${T}.log" 2>&1 || echo FAILED
  cat "eval/${T}_full1319.json" 2>/dev/null
  L=$(grep -o 'logs/[0-9T:+-]*_gsm8k_[A-Za-z0-9]*\.json' "logs/evalfull_${T}.log" | tail -1)
  python analyze_eval.py --log "$L" --out "analysis/${T}_full_diag.json" | head -14
done
