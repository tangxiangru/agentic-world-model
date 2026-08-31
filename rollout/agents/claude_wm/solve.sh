#!/bin/bash
# claude_wm — study conditions C2 (raw files + WMA) and C3 (WMA with seeded memory).
#
# The same Claude Code invocation as claude_non_api, plus the world-model
# runtime beside it: this script loads an awm payload from an immutable commit, puts
# `awm` on PATH, initialises /home/ben/task/wm, installs the scientist's
# skill and Stop hook, and hands the agent our prompt (POST_TRAIN_BENCH_PROMPT
# = prompt_wm or prompt_wm_fulltraj, chosen by the pack script). Whether the
# prior runs are mounted at /home/ben/prior_runs is likewise the pack's call.
#
# AGENT_CONFIG = <claude model>:<arm>:<scope>:ro
#   claude-opus-4-6:llm:train:ro    C2, raw priors + autonomous WMA
#   claude-opus-4-6:llm:train:ro    C3, card corpus + autonomous WMA
#
# C2 must not receive /home/ben/wm-memory. C3 expects it as a read-only bind and
# must not receive /home/ben/prior_runs. Both fail if their declared input is
# absent or the other condition's input leaks in.
set -uo pipefail
: "${AWM_REPO_COMMIT:?ERROR: AWM_REPO_COMMIT must be an immutable 40-hex commit}"
: "${AWM_WMA_MODEL:?ERROR: AWM_WMA_MODEL must pin the autonomous WMA Vertex model}"
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
IFS=: read -r MODEL ARM SIDES RO <<< "${AGENT_CONFIG}"
ARM="${ARM:-null}"; SIDES="${SIDES:-train}"
echo "model=${MODEL} arm=${ARM} memory_sides=${SIDES} readonly=${RO:-no}"

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
        [ "${ARM}" = llm ] || { echo "ERROR: C2 requires arm=llm" >&2; exit 2; }
        ;;
    c3)
        : "${AWM_CARD_CORPUS_MANIFEST_SHA256:?ERROR: expected card corpus manifest SHA-256 was not forwarded}"
        [ ! -e /home/ben/prior_runs ] || {
            echo "ERROR: C3 must not receive the direct prior-runs mount" >&2
            exit 2
        }
        [ "${ARM}" = llm ] || { echo "ERROR: C3 requires arm=llm" >&2; exit 2; }
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
[ "${RO:-}" = "ro" ] || { echo "ERROR: held-out Gemma WMA cells must be read-only (:ro)" >&2; exit 2; }
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

# The autonomous WMA invokes the same installed Claude CLI before the scientist
# starts, so update/verify it before `awm wm init` validates the llm arm.
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

# --- the runtime -----------------------------------------------------------
# setup.sh packages only awm/, input/, and .claude/ from the exact Git commit
# into PTB's agent payload.  No results tree or Git object database enters the
# sandbox, so a C2 scientist cannot discover the historical cards indirectly.
AWM_SRC=/home/ben/agent/awm-src
[ -f "${AWM_SRC}/awm/cli.py" ] && [ -f "${AWM_SRC}/AWM_COMMIT" ] || {
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
python3 -c "import awm.wm.runtime, yaml; print('awm import ok')" || { echo "ERROR: awm does not import" >&2; exit 1; }

export AWM_SESSION_DIR=/home/ben/task
cp "${AWM_SRC}/input/exp-card.template.yaml" /home/ben/task/exp-card.template.yaml
rm -rf /home/ben/task/.claude && cp -r "${AWM_SRC}/.claude" /home/ben/task/.claude

if [ "${AWM_STUDY_CONDITION}" = c2 ]; then
    MEM=/home/ben/wm-empty-memory
    CORPUS_ARGS=(--wma-corpus-kind raw --wma-corpus-root /home/ben/prior_runs)
else
    MEM=/home/ben/wm-memory
    CORPUS_ARGS=(--wma-corpus-kind cards)
fi
INIT_ARGS=(--arm "${ARM}" --submission /home/ben/task/final_model --submission-mode copy
           --memory-root "${MEM}" --memory-sides "${SIDES}"
           --wma-model "${AWM_WMA_MODEL}" --wma-max-budget-usd 6.0
           --wma-validation-attempts 3
           "${CORPUS_ARGS[@]}")
INIT_ARGS+=(--memory-readonly --split-side test)
awm wm --dir /home/ben/task init "${INIT_ARGS[@]}" || { echo "ERROR: awm wm init failed" >&2; exit 1; }
echo "${AWM_SHA}" > /home/ben/task/wm/awm_sha.txt
awm wm --dir /home/ben/task memory stats

if [ "${AWM_STUDY_CONDITION}" = "c2" ]; then
    echo "prior_runs: $(find /home/ben/prior_runs -maxdepth 2 -mindepth 2 -type d | wc -l) run dirs"
else
    echo "prior_runs: intentionally not mounted (C3)"
fi

# --- the scientist ----------------------------------------------------------
export BASH_MAX_TIMEOUT_MS="36000000"
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
    --dangerously-skip-permissions | python3 "${STREAM_REDACTOR}" | tee "${SCIENTIST_STREAM}"
pipeline_status=("${PIPESTATUS[@]}")
prompt_rc="${pipeline_status[0]}"
rc="${pipeline_status[1]}"
redactor_rc="${pipeline_status[2]}"
tee_rc="${pipeline_status[3]}"
printf '%s\n' "${rc}" > /home/ben/task/claude-exit-code.txt
echo "claude exit ${rc} (prompt=${prompt_rc} redactor=${redactor_rc} tee=${tee_rc})"

# --- what the runtime knows at the end ------------------------------------
awm wm --dir /home/ben/task status || true
awm wm --dir /home/ben/task pending || true
[ "${rc}" -eq 0 ] || exit "${rc}"
[ "${prompt_rc}" -eq 0 ] && [ "${redactor_rc}" -eq 0 ] && [ "${tee_rc}" -eq 0 ] || {
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
if grep -Eq '"event"[[:space:]]*:[[:space:]]*"agent_(degraded|failed)"' \
    /home/ben/task/wm/events.jsonl 2>/dev/null; then
    echo "ERROR: autonomous WMA degraded or failed; censoring this labelled cell" >&2
    exit 3
fi
[ -d /home/ben/task/final_model ] && [ -n "$(find /home/ben/task/final_model -mindepth 1 -print -quit)" ] || {
    echo "ERROR: Claude exited successfully without a non-empty final_model/" >&2
    exit 1
}
python3 "${WMA_VALIDATOR}" /home/ben/task \
    --record /home/ben/task/wma-session-attestation.json \
    --study-input /home/ben/task/study-input.json \
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
