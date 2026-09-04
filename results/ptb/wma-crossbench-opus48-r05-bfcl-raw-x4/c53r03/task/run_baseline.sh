python evaluate.py \
  --model-path /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --limit 20 \
  --max-connections 4 \
  --json-output-file baseline_metrics.json 2>&1 | tail -30
