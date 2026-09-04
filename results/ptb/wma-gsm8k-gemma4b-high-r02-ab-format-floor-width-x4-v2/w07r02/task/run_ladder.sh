set -u
for c in 281 562 843 final; do
  if [ "$c" = "final" ]; then M=ckpts/exp-02/final; T=exp-02_final; else M=ckpts/exp-02/checkpoint-$c; T=exp-02_ck$c; fi
  echo "=== $T ==="
  bash run_eval.sh "$M" "$T" || echo "FAILED $T"
done
