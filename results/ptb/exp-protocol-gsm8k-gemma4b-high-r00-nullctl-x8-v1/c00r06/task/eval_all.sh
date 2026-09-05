#!/bin/bash
# usage: eval_all.sh <limit> <tag> <model dirs...>
LIMIT=$1; shift
TAG=$1; shift
for m in "$@"; do
  name=$(basename "$m")
  echo "=== evaluating $m (limit $LIMIT)"
  python evaluate.py --model-path "$m" --limit "$LIMIT" --max-connections 48 \
    --gpu-memory-utilization 0.85 \
    --json-output-file "runs/${TAG}_${name}.json" > "runs/${TAG}_${name}.log" 2>&1
  echo -n "  $name: "; cat "runs/${TAG}_${name}.json" 2>/dev/null | tr -d '\n '; echo
done
echo ALLDONE
