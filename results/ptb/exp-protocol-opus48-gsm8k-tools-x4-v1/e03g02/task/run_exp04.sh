set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== TRAIN START $(date) ==="
awm exp_protocol run --dir . exp-04 2>&1 | tee logs/exp-04.run.log | grep -E "loss|epoch|OutOfMemory|Error|render|data|save|merge|Traceback|returncode" | tail -60
echo "=== TRAIN wrapper exit=${PIPESTATUS[0]} $(date) ==="
if [ ! -f ckpts/exp-04/merged/model.safetensors.index.json ]; then
  echo "FATAL: merged model missing"; tail -20 memory/attempts/exp-04/*/stderr.txt 2>/dev/null; exit 21
fi
ls -la ckpts/exp-04/merged
echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-04/merged --limit 150 --json-output-file /home/ben/task/eval/exp-04_dev150.json 2>&1 | tee logs/exp-04.eval.log | grep -E "accuracy|stderr|total time|Error|samples" | tail -20
echo "=== EVAL exit=${PIPESTATUS[0]} $(date) ==="
echo "=== METRICS ==="; cat eval/exp-04_dev150.json
