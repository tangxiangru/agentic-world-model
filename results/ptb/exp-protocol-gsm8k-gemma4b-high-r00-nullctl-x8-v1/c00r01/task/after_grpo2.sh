set -x
PID=$(cat runs/grpo2.pid)
while ps -p "$PID" > /dev/null 2>&1; do sleep 30; done
sleep 30
bash eval_ckpts.sh runs/grpo2 300 90 180 final
echo "SWEEP2 DONE"
