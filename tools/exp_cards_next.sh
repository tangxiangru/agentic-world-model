#!/bin/bash
# Pop the next run_ref to extract: first queue entry that is neither running
# nor already carded. Appends it to running.txt and prints it. Pass a run_ref
# as $1 to mark it finished (removes it from running.txt) before popping.
set -eu
Q=data/exp-cards/gsm8k-gemma-holdout-v1
R=results/exp-cards/gsm8k-gemma-holdout-v1
touch "$Q/running.txt"
if [ "${1:-}" != "" ]; then
  grep -vxF "$1" "$Q/running.txt" > "$Q/running.tmp" || true
  mv "$Q/running.tmp" "$Q/running.txt"
fi
next=""
while read -r r; do
  grep -qxF "$r" "$Q/running.txt" && continue
  side=$(.venv/bin/python -c "import json;print(json.load(open('$Q/manifest.json'))['$r']['side'])")
  if ls "$R/$side/$r"/exp-*.yaml >/dev/null 2>&1; then continue; fi
  next="$r"; break
done < "$Q/queue.txt"
if [ -n "$next" ]; then echo "$next" >> "$Q/running.txt"; fi
done_n=$( (ls -d "$R"/train/r-*/exp-01.yaml "$R"/test/r-*/exp-01.yaml 2>/dev/null || true) | wc -l | tr -d " ")
echo "next=$next running=$(wc -l < "$Q/running.txt" | tr -d ' ') done=$done_n"
