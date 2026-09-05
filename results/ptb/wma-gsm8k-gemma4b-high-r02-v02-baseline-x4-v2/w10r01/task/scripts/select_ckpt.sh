#!/usr/bin/env bash
# Score a run's checkpoints on the held-out GSM8K-*train* dev set, greedy, with
# the harness's own 10-shot system prefix. Cheap stand-in for the harness eval,
# used only to pick which checkpoint is worth a real --limit 150 read.
#
# Trainer checkpoints carry weights and config.json but no tokenizer and no
# processor files, and Gemma3ForConditionalGeneration needs both to load in
# vLLM, so the snapshot's (unchanged) auxiliary files are copied in first.
set -u
RUN=${1:?run dir, e.g. ../ckpts/exp-02}
SHOTS=${2:-10}
N=${3:-300}
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
cd "$(dirname "$0")"

for d in "$RUN"/checkpoint-* "$RUN"/final; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  tag="$(basename "$RUN")_${name}_fs${SHOTS}"
  out="../analysis/dev${N}_${tag}.jsonl"
  if [ -f "${out%.jsonl}.score" ]; then
    echo "$name (cached): $(cat "${out%.jsonl}.score")"
    continue
  fi
  for f in tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json \
           added_tokens.json preprocessor_config.json processor_config.json; do
    [ -f "$d/$f" ] || cp "$SNAP/$f" "$d/$f" 2>/dev/null
  done
  echo "=== $name ==="
  python gen.py --model "$d" --tokenizer "$SNAP" --questions ../data/dev_heldout300.jsonl \
    --out "$out" --n 1 --temperature 0.0 --fewshot "$SHOTS" --limit "$N" \
    --max-tokens 768 --gpu-mem 0.85 2>"../logs/gen_${tag}.log" | tee "${out%.jsonl}.score"
done
