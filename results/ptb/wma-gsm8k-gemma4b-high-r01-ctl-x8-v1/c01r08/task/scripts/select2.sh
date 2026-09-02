#!/bin/bash
# exp-07: build the remaining soup variants over the same three parents and
# score each under the exp-06 protocol (n=500, greedy, max-connections 16).
set -u
cd /home/ben/task

python scripts/soup.py --inputs ckpts/exp-02/final ckpts/exp-03/final --out ckpts/soup-23  >> logs/exp-07_soup.log 2>&1
python scripts/soup.py --inputs ckpts/exp-02/final ckpts/exp-05/final --out ckpts/soup-25  >> logs/exp-07_soup.log 2>&1
python scripts/soup.py --inputs ckpts/exp-03/final ckpts/exp-05/final --out ckpts/soup-35  >> logs/exp-07_soup.log 2>&1
python scripts/soup.py --inputs ckpts/exp-02/final ckpts/exp-03/final ckpts/exp-05/final \
       --weights 2 1 1 --out ckpts/soup-w211 >> logs/exp-07_soup.log 2>&1

for name in soup23 soup25 soup35 soupw211; do
  case $name in
    soup23) path=ckpts/soup-23;; soup25) path=ckpts/soup-25;;
    soup35) path=ckpts/soup-35;; soupw211) path=ckpts/soup-w211;;
  esac
  echo "=== $name $path $(date -u)"
  python evaluate.py --model-path "$path" --limit 500 --max-connections 16 \
      --json-output-file "/home/ben/task/eval/exp-07_${name}_dev500.json" \
      > "logs/exp-07_${name}.log" 2>&1
  echo "--- $name exit=$? : $(cat "/home/ben/task/eval/exp-07_${name}_dev500.json" 2>/dev/null | tr -d '\n')"
done
echo "=== done $(date -u)"
