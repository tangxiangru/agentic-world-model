#!/bin/bash
# C1 is PostTrainBench's Claude agent with only an alternate prompt and the
# read-only prior-runs mount supplied by wm_pack.sbatch.

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"
MIN_REMAINING_MINUTES=30

bash /home/ben/update_agent_cli.sh claude

if [ -r /home/ben/task/instruction.md ]; then
    claude --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions \
        < /home/ben/task/instruction.md
    scientist_rc=$?
else
    printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
    scientist_rc=$?
fi

# Match PostTrainBench's claude_reprompt lifecycle: a final response ends one
# Claude invocation, so resume the same scientist conversation while useful
# budget remains. This keeps the outer container alive; it is not a model or
# artifact success gate.
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
        --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
    scientist_rc=$?
done

exit "${scientist_rc}"
