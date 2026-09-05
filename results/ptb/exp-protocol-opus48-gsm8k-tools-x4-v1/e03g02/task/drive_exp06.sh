set -uo pipefail
echo "=== EXP06 START $(date) ==="
bash run_exp06.sh > logs/exp-06.run.log 2>&1
RC=$?
echo "=== run_exp06.sh exit=$RC $(date) ==="
grep -E "STAGE|\[sample\]|solved|Contaminated documents|combined_rft|'loss'|epoch|render\]|merge\]|save\]|OutOfMemory|Traceback|DONE|rc=" logs/exp-06.run.log | tail -40
if [ $RC -ne 0 ] || [ ! -f ckpts/exp-06/merged/model.safetensors.index.json ]; then
  echo "PIPELINE FAILED rc=$RC"; tail -25 logs/exp-06.run.log; exit 21
fi
echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-06/merged --limit 150 --json-output-file /home/ben/task/eval/exp-06_dev150.json > logs/exp-06.eval.log 2>&1
echo "=== EVAL exit=$? $(date) ==="
grep -E "accuracy|stderr|total time" logs/exp-06.eval.log | tail -6
echo "=== METRICS ==="; cat eval/exp-06_dev150.json
