#!/bin/bash
# Recorder cells: PostTrainBench's Claude invocation, no peer session. The
# scientist gets no prior information of any kind and explores on its own; it
# registers every experiment itself — `awm wm submit <card.yaml>` before each
# launch and again with results — and the command keeps the record: validated
# cards, script snapshots, archived checkpoints for the post-run evaluation.
# The PTB runner remains responsible for deciding whether the scientist run
# and evaluation succeeded; no study-specific artifact or credential gate runs.

IFS=: read -r MODEL _REST <<EOF
${AGENT_CONFIG}
EOF
AWM_ROOT=/home/ben/agent/awm-src

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"
MIN_REMAINING_MINUTES=30

# Keep the same CLI setup used by PostTrainBench's Claude baseline.
bash /home/ben/update_agent_cli.sh claude

# The WMA implementation is packaged by rollout/setup.sh, so a GPU job never
# clones or installs a second checkout from the network.
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

# The card template the prompt tells the scientist to copy from.
cp "${AWM_ROOT}/input/exp-card.template.yaml" /home/ben/task/exp-card.template.yaml

cd /home/ben/task || exit 1
if [ -r /home/ben/task/instruction.md ]; then
    claude --print --verbose --model "$MODEL" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions \
        < /home/ben/task/instruction.md
    scientist_rc=$?
else
    printf '%s' "$PROMPT" | claude --print --verbose --model "$MODEL" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
    scientist_rc=$?
fi

# Match PostTrainBench's claude_reprompt lifecycle: resume the same scientist
# conversation while budget remains. This is intentionally independent of
# model artifacts and PTB evaluation semantics.
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

    CONTINUATION_PROMPT="You still have ${REMAINING_HOURS}h ${REMAINING_MINS}m remaining. The launcher resumed this scientist conversation because your previous final response ended its Claude invocation. Check any training or evaluation processes and artifacts, recover them if necessary, and continue improving the result."
    printf '%s' "${CONTINUATION_PROMPT}" | claude --print --verbose --continue \
        --model "$MODEL" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
    scientist_rc=$?
done

exit "${scientist_rc}"
