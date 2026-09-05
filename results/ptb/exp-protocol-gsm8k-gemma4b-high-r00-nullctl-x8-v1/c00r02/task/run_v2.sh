#!/usr/bin/env bash
# Wait for RFT generation to finish, then build v2 data, train v2, eval it.
set -u
cd /home/ben/task
GEN_PID=${1:-0}

if [ "$GEN_PID" != "0" ]; then
  echo "[$(date -u +%T)] waiting for gen_rft pid $GEN_PID"
  while kill -0 "$GEN_PID" 2>/dev/null; do sleep 20; done
fi
echo "[$(date -u +%T)] generation finished"
# let the GPU drain
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
echo "[$(date -u +%T)] gpu free (${used}MiB)"

if [ ! -s work/rft_data.jsonl ]; then
  echo "ERROR: work/rft_data.jsonl missing/empty; aborting v2"
  exit 1
fi
wc -l work/rft_data.jsonl

python mix_v2.py --rft work/rft_data.jsonl --base work/sft_data.jsonl \
  --out work/v2_data.jsonl --n-replay 45000 || exit 1

echo "[$(date -u +%T)] starting v2 training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_sft.py \
  --data work/v2_data.jsonl --out work/sft_v2 --init work/sft_v1 \
  --lr 6e-6 --warmup 20 --epochs 1.0 --fewshot-frac 0.2 \
  --micro-tokens 8192 --accum-tokens 131072 \
  --attn flash_attention_2 --no-grad-ckpt > logs/train_v2.log 2>&1
rc=$?
echo "[$(date -u +%T)] training exit $rc"
[ $rc -ne 0 ] && { tr '\r' '\n' < logs/train_v2.log | tail -20; exit 1; }

python finalize.py work/sft_v2 --temperature 0
sleep 15
bash run_eval.sh work/sft_v2 300 v2_greedy 32
echo "[$(date -u +%T)] v2 pipeline done"
