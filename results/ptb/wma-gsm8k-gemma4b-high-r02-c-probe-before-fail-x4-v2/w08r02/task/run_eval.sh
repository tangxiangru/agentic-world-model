#!/usr/bin/env bash
# run_eval.sh <model_dir> <tag> [limit]
# The locked evaluation protocol (n=200 unless overridden) plus the per-sample
# diagnostic, in one step. Writes eval/<tag>.json and analysis/<tag>_diag.json.
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-200}"
mkdir -p eval analysis logs
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}.json" > "logs/${TAG}_eval.log" 2>&1
python analyze_eval.py --logdir logs --out "analysis/${TAG}_diag.json" > /dev/null
python - "$TAG" <<'PY'
import json, sys
tag = sys.argv[1]
d = json.load(open(f"analysis/{tag}_diag.json"))
keep = ["accuracy","stderr","n_samples","share_with_answer_line","share_max_tokens",
        "wrong_but_first_ANSWER_line_correct","wrong_and_answer_line_also_wrong",
        "format_share_of_failures","garbage_prefix_indices","completion_chars_mean",
        "completion_chars_p95","log"]
print(json.dumps({k: d.get(k) for k in keep}, indent=2))
PY
