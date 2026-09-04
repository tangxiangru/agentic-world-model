#!/bin/bash
# wait for training pid to finish, then eval
TPID=$1
MODEL=$2
OUT=$3
LIMIT=${4:-150}
while kill -0 $TPID 2>/dev/null; do sleep 20; done
echo "training pid $TPID finished, starting eval on $MODEL"
python evaluate.py --model-path "$MODEL" --limit $LIMIT --json-output-file "$OUT" --max-connections 4 > eval_${OUT}.log 2>&1
echo "eval done: $OUT"
cat "$OUT"
