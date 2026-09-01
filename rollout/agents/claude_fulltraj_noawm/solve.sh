#!/bin/bash
# claude_fulltraj_noawm — study condition C1, the full-information baseline.
#
# Identical to PostTrainBench's Claude Vertex scaffold: same Claude Code
# invocation, ambient Vertex routing, and effort level. The two things that differ
# are outside this file: the prompt (POST_TRAIN_BENCH_PROMPT=prompt_fulltraj,
# which adds a "Prior runs" section) and the read-only bind of the prior runs at
# /home/ben/prior_runs (POST_TRAIN_BENCH_EXTRA_BINDS, see
# rollout/patches/apply_extra_binds.py). No world-model agent, no runtime.
#
# AGENT_CONFIG = <Claude model id>:<train|train,test>.  The explicit scope is
# recorded in PTB's result path and checked against the mounted raw-run index.
set -uo pipefail
: "${AWM_PRIOR_CORPUS_MANIFEST_SHA256:?ERROR: expected raw corpus manifest SHA-256 was not forwarded}"
: "${AWM_STUDY_REPETITION:?ERROR: explicit study repetition was not forwarded}"
: "${AWM_STUDY_MODE:?ERROR: production/smoke study mode was not forwarded}"
: "${AWM_STUDY_NUM_HOURS:?ERROR: study duration was not forwarded}"
: "${AWM_PTB_COMMIT:?ERROR: exact PostTrainBench commit was not forwarded}"
: "${AWM_REPO_COMMIT:?ERROR: exact harness commit was not forwarded}"
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
IFS=: read -r MODEL SIDES EXTRA <<< "${AGENT_CONFIG}"
{ [ "${SIDES}" = train ] || [ "${SIDES}" = train,test ]; } && [ -z "${EXTRA:-}" ] || {
    echo "ERROR: C1 AGENT_CONFIG must be <model>:<train|train,test>" >&2
    exit 2
}
echo "claude_fulltraj_noawm starting: model=${MODEL} prior_scope=${SIDES}"

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
[ "${AWM_STUDY_CONDITION:-}" = "c1" ] || {
    echo "ERROR: claude_fulltraj_noawm requires AWM_STUDY_CONDITION=c1" >&2
    exit 2
}

CORPUS_VALIDATOR=/home/ben/agent/validate_study_corpus.py
BASE_CACHE_VALIDATOR=/home/ben/agent/validate_base_model_cache.py
C1_MODEL_VALIDATOR=/home/ben/agent/validate_c1_final_model.py
RUNTIME_ATTESTER=/home/ben/agent/attest_claude_runtime.py
STREAM_REDACTOR=/home/ben/agent/redact_claude_stream.py
RESULT_SANITIZER=/home/ben/agent/sanitize_result_tree.py
[ -x "${CORPUS_VALIDATOR}" ] || { echo "ERROR: study corpus validator is missing" >&2; exit 2; }
[ -x "${BASE_CACHE_VALIDATOR}" ] || { echo "ERROR: base-model cache validator is missing" >&2; exit 2; }
[ -x "${C1_MODEL_VALIDATOR}" ] || { echo "ERROR: C1 final-model validator is missing" >&2; exit 2; }
[ -x "${RUNTIME_ATTESTER}" ] || { echo "ERROR: Claude runtime attester is missing" >&2; exit 2; }
[ -x "${STREAM_REDACTOR}" ] || { echo "ERROR: Claude stream redactor is missing" >&2; exit 2; }
[ -x "${RESULT_SANITIZER}" ] || { echo "ERROR: result-tree sanitizer is missing" >&2; exit 2; }
python3 "${CORPUS_VALIDATOR}" raw /home/ben/prior_runs \
    --sides "${SIDES}" \
    --expected-manifest-sha256 "${AWM_PRIOR_CORPUS_MANIFEST_SHA256}" \
    --require-readonly --condition c1 --repetition "${AWM_STUDY_REPETITION}" \
    --study-mode "${AWM_STUDY_MODE}" --num-hours "${AWM_STUDY_NUM_HOURS}" \
    --ptb-commit "${AWM_PTB_COMMIT}" --harness-commit "${AWM_REPO_COMMIT}" \
    --ptb-surface-manifest-sha256 "${AWM_PTB_SURFACE_MANIFEST_SHA256}" \
    --record /home/ben/task/study-input.json || exit 2
BASE_CACHE_ARGS=()
[ "${AWM_STUDY_MODE}" = smoke ] && BASE_CACHE_ARGS+=(--full-hash)
python3 "${BASE_CACHE_VALIDATOR}" "${HF_HOME}" "${BASE_CACHE_ARGS[@]}" \
    --record /home/ben/task/base-model-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

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
claude_rc="${pipeline_status[1]}"
redactor_rc="${pipeline_status[2]}"
printf '%s\n' "${claude_rc}" > /home/ben/task/claude-exit-code.txt
echo "claude exit ${claude_rc} (prompt=${prompt_rc} redactor_capture=${redactor_rc})"
[ "${claude_rc}" -eq 0 ] || exit "${claude_rc}"
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
[ -d /home/ben/task/final_model ] && [ -n "$(find /home/ben/task/final_model -mindepth 1 -print -quit)" ] || {
    echo "ERROR: Claude exited successfully without a non-empty final_model/" >&2
    exit 1
}
BASE_MODEL_REVISION=cc012e0a6d0787b4adcc0fa2c4da74402494554d
BASE_MODEL_CHECKPOINT=/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
[ -d "${BASE_MODEL_CHECKPOINT}" ] && [ ! -L "${BASE_MODEL_CHECKPOINT}" ] || {
    echo "ERROR: cannot resolve the read-only official base-model checkpoint" >&2
    exit 2
}
# This rejects accidental substitutions using the independently attested base
# and the candidate's local HF structure. It intentionally does not claim to
# prove the causal process that trained the candidate weights.
python3 "${C1_MODEL_VALIDATOR}" /home/ben/task/final_model \
    --expected-base-model google/gemma-3-4b-pt \
    --expected-base-revision "${BASE_MODEL_REVISION}" \
    --expected-base-checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --task-root /home/ben/task \
    --study-input /home/ben/task/study-input.json \
    --record /home/ben/task/c1-final-model-attestation.json || exit 2
ls -la /home/ben/task/final_model
echo "claude_fulltraj_noawm done"
