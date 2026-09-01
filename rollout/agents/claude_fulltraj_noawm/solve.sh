#!/bin/bash
# C1 is PostTrainBench's Claude agent with only an alternate prompt and the
# read-only prior-runs mount supplied by wm_pack.sbatch.

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

bash /home/ben/update_agent_cli.sh claude

if [ -r /home/ben/task/instruction.md ]; then
    claude --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions \
        < /home/ben/task/instruction.md
else
    printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
fi
