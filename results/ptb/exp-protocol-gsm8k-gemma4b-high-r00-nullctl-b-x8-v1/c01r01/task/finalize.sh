#!/bin/bash
# Wait for GRPO to stop, then export + evaluate the candidate checkpoints so the
# best one can be picked for final_model.
set -u
PID=$1
LIMIT=${2:-250}
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
sleep 45

LAST=$(ls -d work/grpo_v2/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$LAST" ]; then echo "no grpo checkpoints"; exit 1; fi
ALL=$(ls -d work/grpo_v2/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
MID=$(echo "$ALL" | awk 'NR==int((NR_TOTAL+1)/2)' NR_TOTAL="$(echo "$ALL" | wc -l)")
[ -z "$MID" ] && MID=$(echo "$ALL" | head -1)

echo "candidates: last=$LAST mid=$MID"
bash export_and_eval.sh "work/grpo_v2/checkpoint-$LAST" "work/export_last" "grpo_last" "$LIMIT"
bash export_and_eval.sh "work/grpo_v2/checkpoint-$MID"  "work/export_mid"  "grpo_mid"  "$LIMIT"
echo "=== done ==="
for f in work/grpo_last.json work/grpo_mid.json; do echo "$f: $(cat $f 2>/dev/null)"; done
