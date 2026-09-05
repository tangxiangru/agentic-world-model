set -uo pipefail
echo "=== TRAIN START $(date) ==="
awm exp_protocol run --dir . exp-03 2>&1 | tee logs/exp-03.run.log
echo "=== TRAIN wrapper exit=${PIPESTATUS[0]} $(date) ==="
if [ ! -f ckpts/exp-03/merged/model.safetensors.index.json ]; then
  echo "FATAL: merged model missing"; ls -la ckpts/exp-03/merged 2>/dev/null; exit 21
fi
ls -la ckpts/exp-03/merged
echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-03/merged --limit 150 --json-output-file /home/ben/task/eval/exp-03_dev150.json 2>&1 | tee logs/exp-03.eval.log
echo "=== EVAL exit=${PIPESTATUS[0]} $(date) ==="
echo "=== METRICS ==="; cat eval/exp-03_dev150.json
