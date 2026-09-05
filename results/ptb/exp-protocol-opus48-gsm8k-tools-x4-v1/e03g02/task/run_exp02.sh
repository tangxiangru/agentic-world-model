set -uo pipefail
echo "=== TRAIN START $(date) ==="
awm exp_protocol run --dir . exp-02 2>&1 | tee logs/exp-02.run.log
TRAIN_RC=${PIPESTATUS[0]}
echo "=== TRAIN awm-run wrapper exit=$TRAIN_RC $(date) ==="
if [ ! -f ckpts/exp-02/merged/model.safetensors.index.json ] && [ ! -f ckpts/exp-02/merged/model.safetensors ]; then
  echo "FATAL: merged model artifacts missing"; ls -la ckpts/exp-02/merged 2>/dev/null; exit 21
fi
echo "=== merged files ==="; ls -la ckpts/exp-02/merged
echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-02/merged --limit 150 --json-output-file /home/ben/task/eval/exp-02_dev150.json 2>&1 | tee logs/exp-02.eval.log
EVAL_RC=${PIPESTATUS[0]}
echo "=== EVAL exit=$EVAL_RC $(date) ==="
echo "=== METRICS ==="; cat eval/exp-02_dev150.json
