#!/bin/bash
# C2/C3 use PostTrainBench's Claude invocation plus one peer WMA session. The
# PTB runner remains responsible for deciding whether the scientist run and
# evaluation succeeded; no study-specific artifact or credential gate runs.

IFS=: read -r MODEL ARM SIDES <<EOF
${AGENT_CONFIG}
EOF
ARM="${ARM:-null}"
SIDES="${SIDES:-train}"
WMA_MODEL="${AWM_WMA_MODEL:-claude-opus-5}"
AWM_ROOT=/home/ben/agent/awm-src

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

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

PRIOR_ARGS=()
MEMORY_ARGS=()
[ "${ARM}" != traj ] || PRIOR_ARGS=(--prior-runs /home/ben/prior_runs)
[ "${ARM}" != retrieval ] || MEMORY_ARGS=(--memory-root /home/ben/wm-memory --memory-readonly)

python3 -m awm.cli wm init \
    --arm "${ARM}" \
    "${PRIOR_ARGS[@]}" \
    "${MEMORY_ARGS[@]}" \
    --memory-sides "${SIDES}" \
    --wma-model "${WMA_MODEL}" \
    --base-model "${MODEL_TO_TRAIN:-google/gemma-3-4b-pt}"
awm_init_rc=$?

WMA_PID=""
if [ "${awm_init_rc}" -eq 0 ] && [ -d "${AWM_ROOT}/wma" ]; then
    cp -r "${AWM_ROOT}/wma" /home/ben/wma
    (
        cd /home/ben/wma || exit
        exec env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT AWM_SESSION_DIR=/home/ben/task \
            claude --print --verbose --model "${WMA_MODEL}" \
            --output-format stream-json \
            --allowedTools "Read,Grep,Glob,ListAgents,SendMessage,Bash(ls:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*),Bash(rg:*),Bash(find:*),Bash(cat:*),Bash(sleep:*),Bash(awm:*),Bash(python3 -m awm.cli:*),Bash(mkdir:*)" \
            --dangerously-skip-permissions \
            "You are the world-model agent for this session. Read CLAUDE.md and the consult skill, then serve scientist consults for the whole run. Begin by running: sleep 120." \
            > /home/ben/task/wm/wma-session.jsonl \
            2> /home/ben/task/wm/wma-session.err
    ) &
    WMA_PID=$!
else
    printf 'WMA setup did not start (awm init rc=%s)\n' "${awm_init_rc}" \
        > /home/ben/task/wm/wma-session.err
fi

# Give the peer time to register, but do not replace PTB's success semantics
# with a custom readiness or artifact validator.
sleep 20

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

if [ -n "${WMA_PID}" ]; then
    kill "${WMA_PID}" 2>/dev/null || true
    wait "${WMA_PID}" 2>/dev/null || true
fi

exit "${scientist_rc}"
