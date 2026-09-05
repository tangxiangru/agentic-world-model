#!/usr/bin/env bash
# exp-08: run the deliverable exactly as the grader will -- evaluate.py with every
# flag left at its shipped default (model-path final_model, limit 150,
# max-connections 2, gpu-memory-utilization 0.3, max-tokens 4000), from a fresh
# process, with nothing of this session's tuning applied.
set -euo pipefail
python evaluate.py --json-output-file /home/ben/task/eval/exp-08_default.json
python analyze_eval.py --logdir logs --out analysis/exp-08_default_diag.json > /dev/null
python - <<'PY'
import json
d = json.load(open("analysis/exp-08_default_diag.json"))
print(json.dumps({k: d.get(k) for k in
    ["accuracy","stderr","n_samples","share_with_answer_line","share_max_tokens",
     "garbage_prefix_indices","completion_chars_mean","eval_config","model_args","log"]}, indent=2))
PY
