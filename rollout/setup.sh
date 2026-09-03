#!/bin/bash
# Create a private PostTrainBench checkout for the recorder study. The checkout
# is upstream PTB plus the study agents, the prompt templates, two mechanical
# bind hooks, the per-checkpoint evaluation hook, one explicit environment
# allowlist, and one prompt-file bridge. No study-specific release or
# credential gate is installed.
set -euo pipefail

: "${PTB_SOURCE_DIR:?set PTB_SOURCE_DIR to the PostTrainBench checkout}"
: "${HV_PTB_DIR:?set HV_PTB_DIR to a new private checkout path}"
: "${PTB_RESULTS_DIR:?set PTB_RESULTS_DIR to the private results directory}"
: "${AWM_REPO_COMMIT:?set AWM_REPO_COMMIT to the committed WMA harness revision}"

SRC="$(realpath "${PTB_SOURCE_DIR}")"
DST="$(realpath -m "${HV_PTB_DIR}")"
RESULTS="$(realpath -m "${PTB_RESULTS_DIR}")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWM_SOURCE_DIR="${AWM_SOURCE_DIR:-$(cd "${HERE}/.." && pwd)}"

[ ! -e "${DST}" ] || {
    echo "HV_PTB_DIR already exists; choose a new path so old runner patches cannot survive: ${DST}" >&2
    exit 2
}
[ -d "${SRC}/.git" ] || { echo "PTB_SOURCE_DIR is not a Git checkout: ${SRC}" >&2; exit 2; }
[ -f "${SRC}/.env" ] || { echo "PTB_SOURCE_DIR has no .env: ${SRC}" >&2; exit 2; }
git -C "${AWM_SOURCE_DIR}" cat-file -e "${AWM_REPO_COMMIT}^{commit}" 2>/dev/null || {
    echo "AWM_REPO_COMMIT is not a commit in ${AWM_SOURCE_DIR}" >&2
    exit 2
}

PTB_COMMIT="${HV_PTB_SHA:-$(git -C "${SRC}" rev-parse HEAD)}"
git clone --quiet --no-hardlinks "${SRC}" "${DST}"
git -C "${DST}" checkout --quiet --detach "${PTB_COMMIT}"

# This generated benchmark fixture is intentionally outside PTB Git, but the
# ordinary GSM8K runner needs it for its ordinary contamination-checking tool.
TEST_DATA_REL=src/eval/tasks/gsm8k/test_data.json
if [ -f "${SRC}/${TEST_DATA_REL}" ]; then
    install -D -m 0644 "${SRC}/${TEST_DATA_REL}" "${DST}/${TEST_DATA_REL}"
fi

for agent in claude_recorder claude_noprior_noawm claude_fulltraj_noawm claude_wm; do
    install -d "${DST}/agents/${agent}"
    install -m 0755 "${HERE}/agents/${agent}/solve.sh" "${DST}/agents/${agent}/solve.sh"
    install -m 0644 "${HERE}/agents/${agent}/api_keys.json" "${DST}/agents/${agent}/api_keys.json"
    install -m 0644 "${HERE}/agents/${agent}/env_passthrough.txt" "${DST}/agents/${agent}/env_passthrough.txt"
done

# The recorder (for `awm wm submit` and the card template) and the retired
# C2/C3 agent receive the awm code. The narrow payload hook copies this
# directory to /home/ben/agent inside those cells.
for payload_agent in claude_recorder claude_wm; do
    PAYLOAD="${DST}/agents/${payload_agent}/payload/awm-src"
    install -d "${PAYLOAD}"
    git -C "${AWM_SOURCE_DIR}" archive --format=tar "${AWM_REPO_COMMIT}" awm input wma \
        | tar -x -C "${PAYLOAD}" \
            --exclude=awm/credential_guard.py \
            --exclude=awm/wm/agents/llm.py \
            --exclude=awm/wm/scratch_server.py
    printf '%s\n' "${AWM_REPO_COMMIT}" > "${PAYLOAD}/AWM_COMMIT"
done

python3 "${HERE}/patches/apply_extra_binds.py" "${DST}/src/run_task.sh"
python3 "${HERE}/patches/apply_eval_results_bind.py" "${DST}/src/run_task.sh"
python3 "${HERE}/patches/apply_wm_checkpoint_eval.py" "${DST}/src/run_task.sh"
python3 "${HERE}/patches/apply_env_passthrough.py" "${DST}/src/run_task.sh"
python3 "${HERE}/patches/apply_agent_payload.py" "${DST}/src/run_task.sh"
python3 "${HERE}/patches/apply_prompt_file.py" "${DST}/src/run_task.sh"
python3 "${HERE}/build_prompts.py" --no-review "${DST}"

mkdir -p "${RESULTS}"
grep -q '^POST_TRAIN_BENCH_RESULTS_DIR=' "${SRC}/.env" || {
    echo "source PTB .env has no POST_TRAIN_BENCH_RESULTS_DIR" >&2
    exit 2
}
sed -E \
    -e "s#^POST_TRAIN_BENCH_RESULTS_DIR=.*#POST_TRAIN_BENCH_RESULTS_DIR=\"${RESULTS}\"#" \
    -e 's#^POST_TRAIN_BENCH_EXPERIMENT_NAME=.*##' \
    "${SRC}/.env" > "${DST}/.env"
chmod 0600 "${DST}/.env"

echo "PostTrainBench commit: $(git -C "${DST}" rev-parse HEAD)"
echo "WMA harness commit:   ${AWM_REPO_COMMIT}"
echo "private checkout:     ${DST}"
echo "results:              ${RESULTS}"
