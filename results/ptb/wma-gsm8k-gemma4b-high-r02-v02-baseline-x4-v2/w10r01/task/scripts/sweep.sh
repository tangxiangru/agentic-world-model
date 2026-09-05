#!/usr/bin/env bash
# Score explicit model dirs on dev300 at a given shot count. Usage:
#   bash sweep.sh <shots> <dir> [dir ...]
# Trainer checkpoints get the snapshot's tokenizer/processor files copied in;
# decoding is forced greedy through gen.py's SamplingParams, so a checkpoint's
# own (possibly sampled) generation_config does not matter here.
set -u
SHOTS=${1:?shots}; shift
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
cd "$(dirname "$0")"
for d in "$@"; do
  [ -d "$d" ] || { echo "missing $d"; continue; }
  tag=$(echo "$d" | sed 's#.*/ckpts/##; s#/#_#g')_fs${SHOTS}
  out="../analysis/dev_${tag}.jsonl"
  if [ -s "${out%.jsonl}.res" ]; then echo "$tag (cached): $(cat "${out%.jsonl}.res")"; continue; fi
  for f in tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json \
           added_tokens.json preprocessor_config.json processor_config.json; do
    [ -f "$d/$f" ] || cp "$SNAP/$f" "$d/$f" 2>/dev/null
  done
  PYTHONDONTWRITEBYTECODE=1 python gen.py --model "$d" --tokenizer "$SNAP" --questions ${DEVSET:-../data/dev_heldout300.jsonl} \
    --out "$out" --n 1 --temperature 0.0 --fewshot "$SHOTS" --limit ${DEVN:-300} \
    --max-tokens 768 --gpu-mem 0.85 > "../logs/gen_${tag}.log" 2>&1
  grep -o '{"greedy_pass@1.*}' "../logs/gen_${tag}.log" > "${out%.jsonl}.res"
  echo "$tag: $(cat "${out%.jsonl}.res")"
done
