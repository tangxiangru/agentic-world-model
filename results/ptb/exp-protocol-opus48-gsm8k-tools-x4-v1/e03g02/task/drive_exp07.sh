set -uo pipefail
echo "=== EVAL exp-06/final_model @500 $(date) ==="
python evaluate.py --model-path /home/ben/task/final_model --limit 500 --json-output-file /home/ben/task/eval/exp-07_final_dev500.json > logs/exp-07.log 2>&1
echo "final rc=$? $(date)"; grep -E "accuracy|stderr|total time" logs/exp-07.log | tail -4
echo "=== EVAL exp-04 @500 $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-04/merged --limit 500 --json-output-file /home/ben/task/eval/exp-07_exp04_dev500.json > logs/exp-07b.log 2>&1
echo "exp04 rc=$? $(date)"; grep -E "accuracy|stderr|total time" logs/exp-07b.log | tail -4
echo "=== RESULTS ==="; echo "final_model(exp-06):"; cat eval/exp-07_final_dev500.json; echo; echo "exp-04:"; cat eval/exp-07_exp04_dev500.json
