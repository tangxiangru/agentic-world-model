set -x
PID=$(cat runs/grpo3.pid)
while ps -p "$PID" > /dev/null 2>&1; do sleep 20; done
sleep 20
bash eval_ckpts.sh runs/grpo3 300 final
echo "SWEEP3 DONE"
