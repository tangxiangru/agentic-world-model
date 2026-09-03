#!/bin/bash
# Launch exactly one study cell through PostTrainBench. Scheduler resources and
# site paths belong in the untracked submission wrapper.
set -euo pipefail

[ "$#" -eq 1 ] || {
    echo "usage: wm_pack.sbatch <one study spec>" >&2
    echo "  r:<scientist>:<task>:<base>:<rep>   recorder cell: no prior information, experiments" >&2
    echo "                                      registered by command (task and base are part of the spec)" >&2
    echo "  c0:<scientist>:<rep>                 no-prior baseline without registration" >&2
    echo "                                      (task/base from PTB_TASK / PTB_MODEL)" >&2
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
    c0)
        # No prior information at all: c0:<scientist>:<repetition>. Nothing is
        # mounted and there is no scope factor.
        REPETITION="${FIELD3}"
        [ -z "${FIELD4}${FIELD5}${EXTRA}" ] || { echo "invalid C0 spec (c0:<model>:<rep>): ${SPEC}" >&2; exit 2; }
        AGENT=claude_noprior_noawm
        AGENT_CONFIG="${SCIENTIST}"
        PROMPT_NAME=prompt_noprior
        BINDS=()
        ;;
    r)
        # Recorder: r:<scientist>:<task>:<base>:<repetition>. The scientist gets
        # no prior information of any kind and registers each experiment itself
        # with `awm wm submit`; nothing historical is mounted and there is no
        # peer session. The task and base model are part of the spec so that one
        # string names the cell across the whole matrix.
        TASK="${FIELD3}"
        BASE_ALIAS="${FIELD4}"
        REPETITION="${FIELD5}"
        [ -z "${EXTRA}" ] || { echo "invalid recorder spec (r:<model>:<task>:<base>:<rep>): ${SPEC}" >&2; exit 2; }
        case "${TASK}" in
            gsm8k|gpqamain|healthbench) ;;
            *) echo "unsupported task in ${SPEC} (gsm8k, gpqamain, healthbench)" >&2; exit 2 ;;
        esac
        case "${BASE_ALIAS}" in
            gemma3-4b) MODEL_TO_TRAIN=google/gemma-3-4b-pt ;;
            qwen3-4b)  MODEL_TO_TRAIN=Qwen/Qwen3-4B-Base ;;
            *) echo "unsupported base model in ${SPEC} (gemma3-4b, qwen3-4b)" >&2; exit 2 ;;
        esac
        AGENT=claude_recorder
        AGENT_CONFIG="${SCIENTIST}"
        PROMPT_NAME=prompt_record
        [ "${SMOKE}" != 1 ] || PROMPT_NAME=prompt_record_smoke
        BINDS=()
        ;;
    *) echo "condition must be r or c0: ${SPEC}" >&2; exit 2 ;;
esac

case "${CONDITION}:${REPETITION}" in
    c0:[1-4]) ;;
    r:[1-8]) ;;
    *) echo "invalid repetition: ${REPETITION}" >&2; exit 2 ;;
esac

SAFE_SPEC="${SPEC//\//_}"
SAFE_SPEC="${SAFE_SPEC//:/_}"
SAFE_SPEC="${SAFE_SPEC//,/_}"
CELL_ID="${RUN_ID}_${SAFE_SPEC}"

cd "${HV_PTB_DIR}"
source src/commit_utils/set_env_vars.sh

export NUM_GPUS=1
export POST_TRAIN_BENCH_PROMPT="${PROMPT_NAME}"
export POST_TRAIN_BENCH_EXTRA_BINDS="$(IFS=,; echo "${BINDS[*]-}")"

# Preserve the allocation's GPU visibility. PTB still performs its ordinary
# CUDA check and all ordinary evaluation/result handling.
if [ -z "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ]; then
    export POST_TRAIN_BENCH_VISIBLE_GPUS="${PTB_GPU_SLOTS:-${CUDA_VISIBLE_DEVICES:-}}"
fi

echo "study cell=${SPEC} ptb_id=${CELL_ID} task=${TASK} base=${MODEL_TO_TRAIN} gpu=${POST_TRAIN_BENCH_VISIBLE_GPUS:-scheduler-default}"
exec bash src/run_task.sh \
    "${TASK}" "${AGENT}" "${MODEL_TO_TRAIN}" "${CELL_ID}" \
    "${NUM_HOURS}" "${AGENT_CONFIG}" 1
