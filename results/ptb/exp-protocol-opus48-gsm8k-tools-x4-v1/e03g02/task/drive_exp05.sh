set -uo pipefail
echo "=== RFT PIPELINE START (direct locked argv) $(date) ==="
bash run_exp05.sh > logs/exp-05.run.log 2>&1
RC=$?
echo "=== run_exp05.sh exit=$RC $(date) ==="
echo "--- key lines ---"
grep -E "STAGE|\[sample\]|Contaminat|Documents scanned|combined_rft|'loss'|epoch|render|merge|save\]|OutOfMemory|Error|Traceback|DONE" logs/exp-05.run.log | tail -60
if [ $RC -ne 0 ] || [ ! -f ckpts/exp-05/merged/model.safetensors.index.json ]; then
  echo "FATAL: pipeline failed (rc=$RC) or merged model missing"; tail -30 logs/exp-05.run.log; exit 21
fi
echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-05/merged --limit 150 --json-output-file /home/ben/task/eval/exp-05_dev150.json > logs/exp-05.eval.log 2>&1
echo "=== EVAL exit=$? $(date) ==="
grep -E "accuracy|stderr|total time" logs/exp-05.eval.log | tail -8
echo "=== METRICS ==="; cat eval/exp-05_dev150.json
