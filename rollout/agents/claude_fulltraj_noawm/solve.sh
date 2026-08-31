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
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    [ -r "${GOOGLE_APPLICATION_CREDENTIALS}" ] || {
        echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS is not readable in the sandbox" >&2
        exit 2
    }
else
    curl -fsS --connect-timeout 3 -o /dev/null -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token || {
        echo "ERROR: Vertex needs ADC or an attached Google service account" >&2
        exit 2
    }
fi
[ "${AWM_STUDY_CONDITION:-}" = "c1" ] || {
    echo "ERROR: claude_fulltraj_noawm requires AWM_STUDY_CONDITION=c1" >&2
    exit 2
}

CORPUS_VALIDATOR=/home/ben/agent/validate_study_corpus.py
RUNTIME_ATTESTER=/home/ben/agent/attest_claude_runtime.py
[ -x "${CORPUS_VALIDATOR}" ] || { echo "ERROR: study corpus validator is missing" >&2; exit 2; }
[ -x "${RUNTIME_ATTESTER}" ] || { echo "ERROR: Claude runtime attester is missing" >&2; exit 2; }
python3 "${CORPUS_VALIDATOR}" raw /home/ben/prior_runs \
    --sides "${SIDES}" \
    --expected-manifest-sha256 "${AWM_PRIOR_CORPUS_MANIFEST_SHA256}" \
    --require-readonly --condition c1 --repetition "${AWM_STUDY_REPETITION}" \
    --study-mode "${AWM_STUDY_MODE}" --num-hours "${AWM_STUDY_NUM_HOURS}" \
    --ptb-commit "${AWM_PTB_COMMIT}" --harness-commit "${AWM_REPO_COMMIT}" \
    --ptb-surface-manifest-sha256 "${AWM_PTB_SURFACE_MANIFEST_SHA256}" \
    --record /home/ben/task/study-input.json || exit 2

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
printf '%s' "$PROMPT" | claude --print --verbose --model "$MODEL" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions | tee "${SCIENTIST_STREAM}"
pipeline_status=("${PIPESTATUS[@]}")
prompt_rc="${pipeline_status[0]}"
claude_rc="${pipeline_status[1]}"
tee_rc="${pipeline_status[2]}"
printf '%s\n' "${claude_rc}" > /home/ben/task/claude-exit-code.txt
echo "claude exit ${claude_rc} (prompt=${prompt_rc} tee=${tee_rc})"
[ "${claude_rc}" -eq 0 ] || exit "${claude_rc}"
[ "${prompt_rc}" -eq 0 ] && [ "${tee_rc}" -eq 0 ] || {
    echo "ERROR: failed to preserve the complete Claude stream" >&2
    exit 1
}
python3 "${RUNTIME_ATTESTER}" model "${SCIENTIST_STREAM}" \
    --requested-alias "${MODEL}" \
    --expected-model-id "${AWM_EXPECTED_SCIENTIST_MODEL_ID}" \
    --record /home/ben/task/scientist-model-attestation.json \
    --study-input /home/ben/task/study-input.json || exit 2
[ -d /home/ben/task/final_model ] && [ -n "$(find /home/ben/task/final_model -mindepth 1 -print -quit)" ] || {
    echo "ERROR: Claude exited successfully without a non-empty final_model/" >&2
    exit 1
}
ls -la /home/ben/task/final_model
echo "claude_fulltraj_noawm done"
