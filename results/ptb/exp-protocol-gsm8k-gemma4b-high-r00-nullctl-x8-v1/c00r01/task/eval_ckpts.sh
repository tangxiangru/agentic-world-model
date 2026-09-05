#!/bin/bash
# usage: eval_ckpts.sh <run_dir> <limit> <ckpt_step...>
set -u
RUN=$1; LIMIT=$2; shift 2
export HF_HOME=/home/ben/hf_cache
for STEP in "$@"; do
  SRC="$RUN/checkpoint-$STEP"
  [ "$STEP" = "final" ] && SRC="$RUN/final"
  if [ ! -d "$SRC" ]; then echo "missing $SRC"; continue; fi
  TAG="$(basename "$RUN")_$STEP"
  python package_model.py "$SRC" "runs/pkg_$TAG" --greedy > /dev/null
  bash run_eval.sh "runs/pkg_$TAG" "$TAG" "$LIMIT"
  rm -rf "runs/pkg_$TAG"
done
echo "EVAL SWEEP DONE"
for f in runs/*.json; do echo -n "$f "; python -c "import json,sys;print(json.load(open('$f')))" 2>/dev/null; done
