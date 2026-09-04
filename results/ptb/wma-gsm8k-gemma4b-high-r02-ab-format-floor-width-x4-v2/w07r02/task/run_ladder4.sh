set -u
for c in final 312; do
  if [ "$c" = "final" ]; then M=ckpts/exp-04/final; T=exp-04_final; else M=ckpts/exp-04/checkpoint-$c; T=exp-04_ck$c; fi
  echo "=== $T ==="; bash run_eval.sh "$M" "$T" || echo "FAILED $T"
done
