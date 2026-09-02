#!/bin/bash
# Launcher kept in its own file so no interactive command line contains the
# string "sample_model.py" -- a `pkill -f sample_model.py` earlier matched the
# wrapper shell's own argv and killed the shell that had just spawned the job.
cd /home/ben/task || exit 1
python scripts/sample_model.py \
  --model ckpts/exp-03-greedy \
  --questions data/rft_q_gsm8k.jsonl \
  --out data/rft_samples_gsm8k.jsonl \
  --mode rft --k 6 --temperature 1.0 --max-tokens 512 \
  --fewshot 0 --gpu-mem 0.85 --max-model-len 1536 --max-num-seqs 1024
echo "RFT_SAMPLING_DONE rc=$?"
