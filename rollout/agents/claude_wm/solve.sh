#!/bin/bash
# claude_wm — C2 (raw files + trajectory WMA) and C3 (card-retrieval WMA).
#
# Two Claude Code sessions share the PTB sandbox and communicate with
# ListAgents/SendMessage.  The scientist owns training, evaluation, the GPU,
# and every decision.  The fixed-model WMA serves only the `consult` verb and
# records its cards and consult ledger under /home/ben/task/wm/.
#
# AGENT_CONFIG = <scientist model>:<arm>:<scope>
#   claude-opus-4-6:traj:train            C2, 143 raw prior trajectories
#   claude-opus-4-6:retrieval:train,test  C3, cards from all 193 trajectories
#
# C2 must not receive /home/ben/wm-memory. C3 expects it as a read-only bind and
# must not receive /home/ben/prior_runs. Both fail if their declared input is
# absent or the other condition's input leaks in.
set -uo pipefail
: "${AWM_REPO_COMMIT:?ERROR: AWM_REPO_COMMIT must be an immutable 40-hex commit}"
: "${AWM_WMA_MODEL:?ERROR: AWM_WMA_MODEL must pin the peer WMA Vertex model}"
: "${AWM_STUDY_REPETITION:?ERROR: explicit study repetition was not forwarded}"
: "${AWM_STUDY_MODE:?ERROR: production/smoke study mode was not forwarded}"
: "${AWM_STUDY_NUM_HOURS:?ERROR: study duration was not forwarded}"
: "${AWM_PTB_COMMIT:?ERROR: exact PostTrainBench commit was not forwarded}"
: "${AWM_PTB_SURFACE_MANIFEST_SHA256:?ERROR: PTB study-surface attestation was not forwarded}"
: "${AWM_EXPECTED_SCIENTIST_MODEL_ID:?ERROR: exact reported scientist model ID was not forwarded}"
: "${AWM_CLAUDE_CLI_VERSION:?ERROR: exact Claude CLI npm version was not forwarded}"
: "${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT:?ERROR: exact Claude CLI --version output was not forwarded}"
: "${STUDY_PROMPT_SHA256:?ERROR: exact study prompt SHA-256 was not forwarded}"
: "${STUDY_PROMPT_BYTES:?ERROR: exact study prompt byte length was not forwarded}"
[[ "${STUDY_PROMPT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: invalid STUDY_PROMPT_SHA256" >&2
    exit 2
}
[[ "${STUDY_PROMPT_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: invalid STUDY_PROMPT_BYTES" >&2
    exit 2
}
readonly STUDY_PROMPT_SHA256 STUDY_PROMPT_BYTES
[[ "${AWM_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "ERROR: AWM_REPO_COMMIT must be a full 40-hex commit, got ${AWM_REPO_COMMIT}" >&2
    exit 2
}

echo "claude_wm starting: AGENT_CONFIG=${AGENT_CONFIG}"
IFS=: read -r MODEL ARM SIDES EXTRA <<< "${AGENT_CONFIG}"
ARM="${ARM:-null}"; SIDES="${SIDES:-train}"
[ -z "${EXTRA:-}" ] || { echo "ERROR: unexpected AGENT_CONFIG field" >&2; exit 2; }
echo "scientist=${MODEL} wma=${AWM_WMA_MODEL} arm=${ARM} scope=${SIDES}"

# Vertex cells must never inherit a direct Anthropic/subscription credential,
# including one injected through APPTAINERENV_/SINGULARITYENV_ by the host.
for secret_name in ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN; do
    if [[ -v "${secret_name}" ]]; then
        echo "ERROR: direct Claude credential is present in the Vertex-only sandbox: ${secret_name}" >&2
        exit 2
    fi
done

[ "${CLAUDE_CODE_USE_VERTEX:-}" = 1 ] || {
    echo "ERROR: CLAUDE_CODE_USE_VERTEX=1 was not forwarded" >&2
    exit 2
}
[ -n "${ANTHROPIC_VERTEX_PROJECT_ID:-}" ] || {
    echo "ERROR: ANTHROPIC_VERTEX_PROJECT_ID was not forwarded" >&2
    exit 2
}
[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || {
    echo "ERROR: persistent ADC files are forbidden in the scientist sandbox" >&2
    exit 2
}
curl -fsS --connect-timeout 3 -o /dev/null -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token || {
    echo "ERROR: Vertex needs an attached Google service account" >&2
    exit 2
}
case "${AWM_STUDY_CONDITION:-}" in
    c2)
        : "${AWM_PRIOR_CORPUS_MANIFEST_SHA256:?ERROR: expected raw corpus manifest SHA-256 was not forwarded}"
        [ -f /home/ben/prior_runs/INDEX.md ] && [ -s /home/ben/prior_runs/index.jsonl ] || {
            echo "ERROR: C2 requires the read-only /home/ben/prior_runs mount with a non-empty index" >&2
            exit 2
        }
        [ ! -e /home/ben/wm-memory ] || {
            echo "ERROR: C2 must not receive historical card memory" >&2
            exit 2
        }
        [ "${ARM}" = traj ] || { echo "ERROR: C2 requires arm=traj" >&2; exit 2; }
        ;;
    c3)
        : "${AWM_CARD_CORPUS_MANIFEST_SHA256:?ERROR: expected card corpus manifest SHA-256 was not forwarded}"
        [ ! -e /home/ben/prior_runs ] || {
            echo "ERROR: C3 must not receive the direct prior-runs mount" >&2
            exit 2
        }
        [ "${ARM}" = retrieval ] || { echo "ERROR: C3 requires arm=retrieval" >&2; exit 2; }
        [ -s /home/ben/wm-memory/structured/cards.jsonl ] || {
            echo "ERROR: C3 requires seeded card memory at /home/ben/wm-memory" >&2
            exit 2
        }
        ;;
    *)
        echo "ERROR: claude_wm requires AWM_STUDY_CONDITION=c2 or c3" >&2
        exit 2
        ;;
esac
[ "${SIDES}" = train ] || [ "${SIDES}" = train,test ] || {
    echo "ERROR: memory sides must be train or train,test" >&2
    exit 2
}
CORPUS_VALIDATOR=/home/ben/agent/validate_study_corpus.py
BASE_CACHE_VALIDATOR=/home/ben/agent/validate_base_model_cache.py
RUNTIME_ATTESTER=/home/ben/agent/attest_claude_runtime.py
WMA_VALIDATOR=/home/ben/agent/validate_wma_session.py
FINAL_MODEL_VALIDATOR=/home/ben/agent/validate_c1_final_model.py
STREAM_REDACTOR=/home/ben/agent/redact_claude_stream.py
RESULT_SANITIZER=/home/ben/agent/sanitize_result_tree.py
BASE_MODEL_REVISION=cc012e0a6d0787b4adcc0fa2c4da74402494554d
BASE_MODEL_CHECKPOINT=/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
[ -x "${CORPUS_VALIDATOR}" ] || { echo "ERROR: study corpus validator is missing" >&2; exit 2; }
[ -x "${BASE_CACHE_VALIDATOR}" ] || { echo "ERROR: base-model cache validator is missing" >&2; exit 2; }
[ -x "${RUNTIME_ATTESTER}" ] || { echo "ERROR: Claude runtime attester is missing" >&2; exit 2; }
[ -x "${WMA_VALIDATOR}" ] || { echo "ERROR: WMA session validator is missing" >&2; exit 2; }
[ -x "${FINAL_MODEL_VALIDATOR}" ] || { echo "ERROR: final-model validator is missing" >&2; exit 2; }
[ -x "${STREAM_REDACTOR}" ] || { echo "ERROR: Claude stream redactor is missing" >&2; exit 2; }
[ -x "${RESULT_SANITIZER}" ] || { echo "ERROR: result-tree sanitizer is missing" >&2; exit 2; }
[ -d "${BASE_MODEL_CHECKPOINT}" ] && [ ! -L "${BASE_MODEL_CHECKPOINT}" ] || {
    echo "ERROR: cannot resolve the read-only official base-model checkpoint" >&2
    exit 2
}
if [ "${AWM_STUDY_CONDITION}" = c2 ]; then
    python3 "${CORPUS_VALIDATOR}" raw /home/ben/prior_runs \
        --sides "${SIDES}" \
        --expected-manifest-sha256 "${AWM_PRIOR_CORPUS_MANIFEST_SHA256}" \
        --require-readonly --condition c2 --repetition "${AWM_STUDY_REPETITION}" \
        --study-mode "${AWM_STUDY_MODE}" --num-hours "${AWM_STUDY_NUM_HOURS}" \
        --ptb-commit "${AWM_PTB_COMMIT}" --harness-commit "${AWM_REPO_COMMIT}" \
        --ptb-surface-manifest-sha256 "${AWM_PTB_SURFACE_MANIFEST_SHA256}" \
        --record /home/ben/task/study-input.json || exit 2
else
    python3 "${CORPUS_VALIDATOR}" cards /home/ben/wm-memory \
        --sides "${SIDES}" \
        --expected-manifest-sha256 "${AWM_CARD_CORPUS_MANIFEST_SHA256}" \
        --require-readonly \
        --condition c3 --repetition "${AWM_STUDY_REPETITION}" \
        --study-mode "${AWM_STUDY_MODE}" --num-hours "${AWM_STUDY_NUM_HOURS}" \
        --ptb-commit "${AWM_PTB_COMMIT}" --harness-commit "${AWM_REPO_COMMIT}" \
        --ptb-surface-manifest-sha256 "${AWM_PTB_SURFACE_MANIFEST_SHA256}" \
        --record /home/ben/task/study-input.json || exit 2
fi
BASE_CACHE_ARGS=()
[ "${AWM_STUDY_MODE}" = smoke ] && BASE_CACHE_ARGS+=(--full-hash)
python3 "${BASE_CACHE_VALIDATOR}" "${HF_HOME}" "${BASE_CACHE_ARGS[@]}" \
    --record /home/ben/task/base-model-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2

# Both peer sessions use the same exactly pinned Claude CLI.
[ "${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}" = 1 ] || {
    echo "ERROR: POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1 is required for a reproducible cell" >&2
    exit 2
}
python3 "${RUNTIME_ATTESTER}" install-cli \
    --version-file /home/ben/cli_version.txt \
    --package-version "${AWM_CLAUDE_CLI_VERSION}" \
    --expected-version-output "${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT}" \
    --record /home/ben/task/claude-cli-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2

# --- the commit-pinned WMA toolbelt ---------------------------------------
# setup.sh packages only awm/, input/, and wma/ from the exact Git commit.
# No results tree or Git object database enters the sandbox.
AWM_SRC=/home/ben/agent/awm-src
[ -f "${AWM_SRC}/awm/cli.py" ] && [ -f "${AWM_SRC}/AWM_COMMIT" ] && \
    [ -f "${AWM_SRC}/wma/CLAUDE.md" ] || {
    echo "ERROR: missing packaged awm runtime; rerun rollout/setup.sh" >&2
    exit 1
}
AWM_SHA="$(tr -d '[:space:]' < "${AWM_SRC}/AWM_COMMIT")"
[ "${AWM_SHA}" = "${AWM_REPO_COMMIT,,}" ] || {
    echo "ERROR: packaged awm ${AWM_SHA}, expected ${AWM_REPO_COMMIT}" >&2
    exit 1
}
echo "awm commit ${AWM_SHA}"
export PYTHONPATH="${AWM_SRC}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /home/ben/.local/bin
printf '#!/bin/bash\nexec python3 -m awm.cli "$@"\n' > /home/ben/.local/bin/awm
chmod +x /home/ben/.local/bin/awm
export PATH="/home/ben/.local/bin:${PATH}"
python3 -c "import awm.wm.consult, awm.wm.intake, yaml; print('awm import ok')" || {
    echo "ERROR: awm consult toolbelt does not import" >&2
    exit 1
}

export AWM_SESSION_DIR=/home/ben/task
if [ "${AWM_STUDY_CONDITION}" = c2 ]; then
    EVIDENCE_ARGS=(--prior-runs /home/ben/prior_runs)
else
    EVIDENCE_ARGS=(--memory-root /home/ben/wm-memory)
fi
INIT_ARGS=(--arm "${ARM}" "${EVIDENCE_ARGS[@]}" --memory-sides "${SIDES}"
           --memory-readonly --split-side test --wma-model "${AWM_WMA_MODEL}"
           --base-model google/gemma-3-4b-pt)
awm wm --dir /home/ben/task init "${INIT_ARGS[@]}" || { echo "ERROR: awm wm init failed" >&2; exit 1; }
echo "${AWM_SHA}" > /home/ben/task/wm/awm_sha.txt
awm wm --dir /home/ben/task memory stats

if [ "${AWM_STUDY_CONDITION}" = "c2" ]; then
    echo "prior_runs: $(find /home/ben/prior_runs -maxdepth 2 -mindepth 2 -type d | wc -l) run dirs"
else
    echo "prior_runs: intentionally not mounted (C3)"
fi

# --- the world-model peer session ----------------------------------------
# The WMA gets its own cwd/instructions and a read-mostly tool allowlist.  Its
# only writable study root is task/wm/, where the consult command validates
# every response and citation before appending the ledger.
rm -rf /home/ben/wma
cp -r "${AWM_SRC}/wma" /home/ben/wma
WMA_STREAM=/home/ben/task/wm/wma-session.jsonl
WMA_STDERR=/home/ben/task/wm/wma-session.err
WMA_EXIT=/home/ben/task/wm/wma-exit-code.txt
WMA_CAPTURE_EXIT=/home/ben/task/wm/wma-capture-exit-code.txt
WMA_FIFO=/home/ben/task/wm/.wma-stream.fifo
export BASH_MAX_TIMEOUT_MS="36000000"
rm -f "${WMA_FIFO}"
mkfifo -m 0600 "${WMA_FIFO}" || { echo "ERROR: cannot create WMA stream pipe" >&2; exit 2; }
(
    cd /home/ben/wma || exit 2
    exec env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT AWM_SESSION_DIR=/home/ben/task \
        claude --print --verbose --model "${AWM_WMA_MODEL}" \
        --output-format stream-json \
        --allowedTools "Read,Grep,Glob,ListAgents,SendMessage,Bash(ls:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*),Bash(rg:*),Bash(find:*),Bash(cat:*),Bash(sleep:*),Bash(awm:*),Bash(python3 -m awm.cli:*),Bash(mkdir:*)" \
        --dangerously-skip-permissions \
        "You are the world-model agent for this session. Read CLAUDE.md and the consult skill. A research scientist session will message you; serve consults for the whole run under the standing order. Begin by running: sleep 120." \
        >"${WMA_FIFO}" 2>"${WMA_STDERR}"
) &
WMA_PID=$!
python3 "${STREAM_REDACTOR}" --capture "${WMA_STREAM}" \
    <"${WMA_FIFO}" >/dev/null &
WMA_CAPTURE_PID=$!

# Wait for the WMA's init event instead of assuming a fixed startup time.
wma_ready=0
for _ in $(seq 1 60); do
    kill -0 "${WMA_PID}" 2>/dev/null || break
    if [ -s "${WMA_STREAM}" ] && grep -q '"subtype":"init"\|"subtype": "init"' "${WMA_STREAM}"; then
        wma_ready=1
        break
    fi
    sleep 1
done
[ "${wma_ready}" = 1 ] || {
    echo "ERROR: WMA peer session did not become ready" >&2
    kill "${WMA_PID}" 2>/dev/null || true
    kill "${WMA_CAPTURE_PID}" 2>/dev/null || true
    rm -f "${WMA_FIFO}"
    exit 2
}
echo "wma peer ready: pid=${WMA_PID} sockets=$(find /tmp/cc-socks -maxdepth 1 -type s 2>/dev/null | wc -l)"

# --- the scientist ----------------------------------------------------------
export CLAUDE_CODE_EFFORT_LEVEL="high"

SCIENTIST_STREAM=/home/ben/task/scientist-stream.jsonl
STUDY_PROMPT_FILE=/home/ben/task/instruction.md
STUDY_PROMPT_CHECKSUM=/home/ben/task/instruction.sha256
verify_study_prompt() {
    [ -f "${STUDY_PROMPT_FILE}" ] && [ ! -L "${STUDY_PROMPT_FILE}" ] && \
        [ -s "${STUDY_PROMPT_FILE}" ] && [ -f "${STUDY_PROMPT_CHECKSUM}" ] && \
        [ ! -L "${STUDY_PROMPT_CHECKSUM}" ] || return 1
    local actual_sha actual_bytes checksum_line
    actual_sha="$(sha256sum "${STUDY_PROMPT_FILE}" | cut -d' ' -f1)" || return 1
    actual_bytes="$(wc -c < "${STUDY_PROMPT_FILE}")" || return 1
    checksum_line="$(cat "${STUDY_PROMPT_CHECKSUM}")" || return 1
    [ "${actual_sha}" = "${STUDY_PROMPT_SHA256}" ] && \
        [ "${actual_bytes}" = "${STUDY_PROMPT_BYTES}" ] && \
        [ "${checksum_line}" = "${STUDY_PROMPT_SHA256}  instruction.md" ]
}
verify_study_prompt || {
    echo "ERROR: study prompt checksum failed before Claude launch" >&2
    exit 2
}
cat "${STUDY_PROMPT_FILE}" | claude --print --verbose --model "$MODEL" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions | \
    python3 "${STREAM_REDACTOR}" --capture "${SCIENTIST_STREAM}"
pipeline_status=("${PIPESTATUS[@]}")
prompt_rc="${pipeline_status[0]}"
rc="${pipeline_status[1]}"
redactor_rc="${pipeline_status[2]}"
printf '%s\n' "${rc}" > /home/ben/task/claude-exit-code.txt
echo "claude exit ${rc} (prompt=${prompt_rc} redactor_capture=${redactor_rc})"

# --- wind down the peer and validate both sessions ------------------------
for _ in $(seq 1 10); do
    kill -0 "${WMA_PID}" 2>/dev/null || break
    sleep 1
done
if kill -0 "${WMA_PID}" 2>/dev/null; then
    kill -INT "${WMA_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "${WMA_PID}" 2>/dev/null || break
        sleep 1
    done
fi
if kill -0 "${WMA_PID}" 2>/dev/null; then
    kill -TERM "${WMA_PID}" 2>/dev/null || true
    sleep 2
fi
if kill -0 "${WMA_PID}" 2>/dev/null; then
    kill -KILL "${WMA_PID}" 2>/dev/null || true
fi
set +e
wait "${WMA_PID}"
wma_rc=$?
for _ in $(seq 1 10); do
    kill -0 "${WMA_CAPTURE_PID}" 2>/dev/null || break
    sleep 1
done
if kill -0 "${WMA_CAPTURE_PID}" 2>/dev/null; then
    kill -TERM "${WMA_CAPTURE_PID}" 2>/dev/null || true
fi
wait "${WMA_CAPTURE_PID}"
wma_capture_rc=$?
printf '%s\n' "${wma_rc}" > "${WMA_EXIT}"
printf '%s\n' "${wma_capture_rc}" > "${WMA_CAPTURE_EXIT}"
rm -f "${WMA_FIFO}"

awm wm --dir /home/ben/task status || true
[ "${rc}" -eq 0 ] || exit "${rc}"
[ "${prompt_rc}" -eq 0 ] && [ "${redactor_rc}" -eq 0 ] || {
    echo "ERROR: failed to preserve the complete Claude stream" >&2
    exit 1
}
verify_study_prompt || {
    echo "ERROR: study prompt changed during Claude execution" >&2
    exit 2
}
python3 "${RESULT_SANITIZER}" /home/ben/task
sanitizer_rc=$?
case "${sanitizer_rc}" in
    0) ;;
    3)
        echo "ERROR: credential material was removed from task artifacts; quarantining this cell" >&2
        exit 3
        ;;
    *) exit 2 ;;
esac
python3 "${RUNTIME_ATTESTER}" model "${SCIENTIST_STREAM}" \
    --requested-alias "${MODEL}" \
    --expected-model-id "${AWM_EXPECTED_SCIENTIST_MODEL_ID}" \
    --record /home/ben/task/scientist-model-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2
python3 "${RUNTIME_ATTESTER}" model "${WMA_STREAM}" \
    --requested-alias "${AWM_WMA_MODEL}" \
    --expected-model-id "${AWM_WMA_MODEL}" \
    --study-input-key wma_model \
    --record /home/ben/task/wma-model-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2
[ -d /home/ben/task/final_model ] && [ -n "$(find /home/ben/task/final_model -mindepth 1 -print -quit)" ] || {
    echo "ERROR: Claude exited successfully without a non-empty final_model/" >&2
    exit 1
}
python3 "${WMA_VALIDATOR}" /home/ben/task \
    --record /home/ben/task/wma-session-attestation.json \
    --study-input /home/ben/task/study-input.json \
    --expected-arm "${ARM}" \
    --expected-wma-model "${AWM_WMA_MODEL}" \
    --expected-memory-sides "${SIDES}" \
    --expected-base-model google/gemma-3-4b-pt \
    --expected-base-checkpoint "${BASE_MODEL_CHECKPOINT}" || exit 2
python3 "${FINAL_MODEL_VALIDATOR}" /home/ben/task/final_model \
    --expected-base-model google/gemma-3-4b-pt \
    --expected-base-revision "${BASE_MODEL_REVISION}" \
    --expected-base-checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --task-root /home/ben/task \
    --study-input /home/ben/task/study-input.json \
    --record /home/ben/task/wma-final-model-attestation.json || exit 2
ls -la /home/ben/task/final_model
echo "claude_wm done"
