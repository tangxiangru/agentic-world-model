#!/bin/bash
# Recorder cells driven by a locally served open-weights scientist. This is
# claude_recorder with the CLI swapped: PostTrainBench's opencode scaffold
# talking to a vLLM OpenAI-compatible endpoint (Qwen3.6-27B served on the same
# node). Same contract: no prior information of any kind, every experiment
# registered by `awm wm submit`, PTB owns success semantics.

AWM_ROOT=/home/ben/agent/awm-src
VLLM_BASE_URL="${AWM_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
MIN_REMAINING_MINUTES=30

export BASH_MAX_TIMEOUT_MS="36000000"

# opencode reads provider config from opencode.json in the working directory.
# vLLM without --api-key accepts any bearer; "vllm" is a placeholder, not a secret.
cat > /home/ben/task/opencode.json <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "${VLLM_BASE_URL}", "apiKey": "vllm" },
      "models": { "qwen3.6-27b": { "name": "Qwen3.6-27B" } }
    }
  }
}
JSON

bash /home/ben/update_agent_cli.sh opencode

export PYTHONPATH="${AWM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AWM_SESSION_DIR=/home/ben/task
mkdir -p /home/ben/.local/bin /home/ben/task/wm
printf '#!/bin/bash\nexec python3 -m awm.cli "$@"\n' > /home/ben/.local/bin/awm
chmod +x /home/ben/.local/bin/awm
export PATH="/home/ben/.local/bin:${PATH}"

python3 -m awm.cli wm init \
    --mode record \
    --arm null \
    --base-model "${MODEL_TO_TRAIN:-google/gemma-3-4b-pt}" \
    || printf 'awm init failed\n' > /home/ben/task/wm/init.err

cp "${AWM_ROOT}/input/exp-card.template.yaml" /home/ben/task/exp-card.template.yaml

cd /home/ben/task || exit 1
# The prompt arrives as task/instruction.md (the --env copy is blanked for
# study agents because a long value corrupts the injected environment).
if [ -r /home/ben/task/instruction.md ]; then
    opencode run --model "$AGENT_CONFIG" --format json < /home/ben/task/instruction.md
    scientist_rc=$?
else
    printf '%s' "$PROMPT" | opencode run --model "$AGENT_CONFIG" --format json
    scientist_rc=$?
fi

# Resume the same scientist session while budget remains, as the Claude
# recorder does. Not a model or artifact success gate.
while true; do
    TIMER_OUTPUT="$(bash timer.sh 2>/dev/null)"
    if printf '%s\n' "${TIMER_OUTPUT}" | grep -q "expired"; then
        break
    fi
    REMAINING_HOURS="$(printf '%s\n' "${TIMER_OUTPUT}" | grep -oP '^\d+(?=:)')"
    REMAINING_MINS="$(printf '%s\n' "${TIMER_OUTPUT}" | grep -oP '(?<=:)\d+')"
    if [ -z "${REMAINING_HOURS}" ] || [ -z "${REMAINING_MINS}" ]; then
        break
    fi
    TOTAL_REMAINING_MINS=$((REMAINING_HOURS * 60 + REMAINING_MINS))
    if [ "${TOTAL_REMAINING_MINS}" -lt "${MIN_REMAINING_MINUTES}" ]; then
        break
    fi
    CONTINUATION_PROMPT="You still have ${REMAINING_HOURS}h ${REMAINING_MINS}m remaining. The launcher resumed this scientist conversation because your previous final response ended it. Check any training or evaluation processes and artifacts, recover them if necessary, register anything unregistered with awm wm submit, and continue improving the result."
    printf '%s' "${CONTINUATION_PROMPT}" | opencode run --continue --model "$AGENT_CONFIG" --format json
    scientist_rc=$?
done

exit "${scientist_rc}"
