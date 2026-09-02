#!/bin/bash
# exp-08: confirm the leading merges on the full 1319-item GSM8K test set.
# soup-w121 is the average of the two n=500 leaders' ingredient sets and is new here.
set -u
cd /home/ben/task
python scripts/soup.py --inputs ckpts/exp-02/final ckpts/exp-03/final ckpts/exp-05/final \
       --weights 1 2 1 --out ckpts/soup-w121 >> logs/exp-08_soup.log 2>&1
for arm in "soup23:ckpts/soup-23" "soup35:ckpts/soup-35" "soupw121:ckpts/soup-w121" "soup235:ckpts/soup-235"; do
  name="${arm%%:*}"; path="${arm#*:}"
  echo "=== $name $path $(date -u)"
  python evaluate.py --model-path "$path" --limit -1 --max-connections 16 \
      --json-output-file "/home/ben/task/eval/exp-08_${name}_full1319.json" \
      > "logs/exp-08_${name}.log" 2>&1
  echo "--- $name exit=$? : $(cat "/home/ben/task/eval/exp-08_${name}_full1319.json" 2>/dev/null | tr -d '\n')"
done
echo "=== done $(date -u)"
