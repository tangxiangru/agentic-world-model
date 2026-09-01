#!/bin/bash
# Launch exactly one study cell through PostTrainBench. Scheduler resources and
# site paths belong in the untracked submission wrapper.
set -euo pipefail

[ "$#" -eq 1 ] || {
    echo "usage: wm_pack.sbatch <one c1/c2/c3 study spec>" >&2
    exit 2
}

: "${HV_PTB_DIR:?set HV_PTB_DIR to the private checkout from rollout/setup.sh}"

SPEC="$1"
TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-google/gemma-3-4b-pt}"
NUM_HOURS="${PTB_NUM_HOURS:-10}"
RUN_ID="${PTB_RUN_ID:-${SLURM_JOB_ID:-local-$$}}"
SMOKE="${AWM_STUDY_SMOKE:-0}"

IFS=: read -r CONDITION SCIENTIST FIELD3 FIELD4 FIELD5 EXTRA <<EOF
${SPEC}
EOF

case "${SCIENTIST}" in
    claude-opus-4-8|claude-opus-5) ;;
    *) echo "unsupported scientist model in ${SPEC}" >&2; exit 2 ;;
esac

case "${CONDITION}" in
    c1)
        SCOPE="${FIELD3}"
        REPETITION="${FIELD4}"
        [ -z "${FIELD5}${EXTRA}" ] || { echo "invalid C1 spec: ${SPEC}" >&2; exit 2; }
        AGENT=claude_fulltraj_noawm
        AGENT_CONFIG="${SCIENTIST}"
        PROMPT_NAME=prompt_fulltraj
        : "${PRIOR_RUNS:?C1 requires PRIOR_RUNS}"
        [ -d "${PRIOR_RUNS}" ] || { echo "missing PRIOR_RUNS: ${PRIOR_RUNS}" >&2; exit 2; }
        BINDS=("${PRIOR_RUNS}:/home/ben/prior_runs:ro")
        ;;
    c2|c3)
        ARM="${FIELD3}"
        SCOPE="${FIELD4}"
        REPETITION="${FIELD5}"
        [ -z "${EXTRA}" ] || { echo "invalid ${CONDITION^^} spec: ${SPEC}" >&2; exit 2; }
        AGENT=claude_wm
        AGENT_CONFIG="${SCIENTIST}:${ARM}:${SCOPE}"
        if [ "${CONDITION}" = c2 ]; then
            [ "${ARM}" = traj ] || { echo "C2 requires the traj arm" >&2; exit 2; }
            : "${PRIOR_RUNS:?C2 requires PRIOR_RUNS}"
            [ -d "${PRIOR_RUNS}" ] || { echo "missing PRIOR_RUNS: ${PRIOR_RUNS}" >&2; exit 2; }
            BINDS=("${PRIOR_RUNS}:/home/ben/prior_runs:ro")
            PROMPT_NAME=prompt_wm_fulltraj
            [ "${SMOKE}" != 1 ] || PROMPT_NAME=prompt_wm_fulltraj_smoke
        else
            [ "${ARM}" = retrieval ] || { echo "C3 requires the retrieval arm" >&2; exit 2; }
            : "${WM_MEMORY:?C3 requires WM_MEMORY}"
            [ -d "${WM_MEMORY}" ] || { echo "missing WM_MEMORY: ${WM_MEMORY}" >&2; exit 2; }
            BINDS=("${WM_MEMORY}:/home/ben/wm-memory:ro")
            PROMPT_NAME=prompt_wm
            [ "${SMOKE}" != 1 ] || PROMPT_NAME=prompt_wm_smoke
        fi
        ;;
    *) echo "condition must be c1, c2, or c3: ${SPEC}" >&2; exit 2 ;;
esac

case "${SCOPE}" in train|train,test) ;; *) echo "invalid scope: ${SCOPE}" >&2; exit 2;; esac
case "${REPETITION}" in 1|2) ;; *) echo "invalid repetition: ${REPETITION}" >&2; exit 2;; esac

SAFE_SPEC="${SPEC//\//_}"
SAFE_SPEC="${SAFE_SPEC//:/_}"
SAFE_SPEC="${SAFE_SPEC//,/_}"
CELL_ID="${RUN_ID}_${SAFE_SPEC}"

cd "${HV_PTB_DIR}"
source src/commit_utils/set_env_vars.sh

export NUM_GPUS=1
export POST_TRAIN_BENCH_PROMPT="${PROMPT_NAME}"
export POST_TRAIN_BENCH_EXTRA_BINDS="$(IFS=,; echo "${BINDS[*]}")"

# Preserve the allocation's GPU visibility. PTB still performs its ordinary
# CUDA check and all ordinary evaluation/result handling.
if [ -z "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ]; then
    export POST_TRAIN_BENCH_VISIBLE_GPUS="${PTB_GPU_SLOTS:-${CUDA_VISIBLE_DEVICES:-}}"
fi

echo "study cell=${SPEC} ptb_id=${CELL_ID} gpu=${POST_TRAIN_BENCH_VISIBLE_GPUS:-scheduler-default}"
exec bash src/run_task.sh \
    "${TASK}" "${AGENT}" "${MODEL_TO_TRAIN}" "${CELL_ID}" \
    "${NUM_HOURS}" "${AGENT_CONFIG}" 1
