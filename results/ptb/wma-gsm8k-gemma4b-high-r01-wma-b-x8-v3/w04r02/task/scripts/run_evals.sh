#!/usr/bin/env bash
# usage: run_evals.sh <tag>:<model-path> [<tag>:<model-path> ...]
set -u
cd /home/ben/task
for spec in "$@"; do
  tag="${spec%%:*}"; path="${spec#*:}"
  echo "=== $tag -> $path ==="
  python evaluate.py --model-path "$path" --limit 150 --max-connections 16 \
      --json-output-file "eval/${tag}.json" > "logs/eval_${tag}.log" 2>&1
  echo "exit $?"
  grep -oE "Log: logs/[^ ]+\.json" "logs/eval_${tag}.log" | tail -1
  cat "eval/${tag}.json" 2>/dev/null; echo
done
echo ALL_DONE
