#!/usr/bin/env bash
# Set the greedy decode arm on every exp-04 checkpoint (so the read is
# comparable with exp-03's 0.7133) and evaluate them under the locked protocol.
set -u
cd /home/ben/task
for CKPT in "$@"; do
  python scripts/set_decode.py --ckpt "$CKPT" --mode greedy > /dev/null
done
bash scripts/sweep_eval.sh exp-04greedy 150 "$@"
