#!/bin/bash
#SBATCH --job-name=wm-study
#SBATCH --partition=a3
#SBATCH --reservation=robtang-a3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=11:30:00
#SBATCH --output=/rmeng_data/robtang/ptb-hvrecipe/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/ptb-hvrecipe/slurm_logs/%x-%j.err
set -uo pipefail

# The scientist x information study: up to 8 one-GPU cells on one exclusive node.
# Unlike hv_pack.sbatch each ARM names both the agent and its config:
#
#   <agent>:<AGENT_CONFIG>
#
#   claude_fulltraj_noawm:claude-opus-4-8              C1  raw files, no WMA
#   claude_wm:claude-opus-4-8:traj                     C2  raw files + autonomous WMA over them (PRIOR_RUNS set)
#   claude_wm:claude-opus-4-8:retrieval:train,test     C3  WMA + seeded memory, no prior runs for this agent
#
# Environment (host side, read here and by run_task.sh):
#   PTB_TASK, PTB_MODEL, PTB_NUM_HOURS        task, base model, hours     (gsm8k, google/gemma-3-4b-pt, 10)
#   PRIOR_RUNS   host dir from tools/build_prior_runs.py -> bound read-only at /home/ben/prior_runs
#   WM_MEMORY    host dir for the WMA memory            -> bound at /home/ben/wm-memory
#   PRIOR_RUNS_FOR   comma list of agents that get the prior-runs bind (default: both claude_* agents)
#
# The prompt is chosen per cell from the agent and whether prior runs are mounted:
#   claude_fulltraj_noawm -> prompt_fulltraj
#   claude_wm             -> prompt_wm_fulltraj if PRIOR_RUNS, else prompt_wm
#
# Interleave conditions and models across GPU slots when you submit — "which
# condition" must not be the same variable as "which card":
#
#   # C1 + C2 (prior runs mounted for both agents; the WMA in C2 reads them itself)
#   sbatch rollout/wm_pack.sbatch \
#     claude_fulltraj_noawm:claude-opus-4-6 claude_wm:claude-opus-4-8:traj claude_wm:claude-opus-5:traj \
#     claude_fulltraj_noawm:claude-opus-4-8 claude_wm:claude-opus-5:traj   claude_wm:claude-opus-4-6:traj \
#     claude_fulltraj_noawm:claude-opus-5   claude_wm:claude-opus-4-6:traj
#   # C3 (no prior runs; memory seeded)
#   PRIOR_RUNS= sbatch rollout/wm_pack.sbatch \
#     claude_wm:claude-opus-4-6:retrieval claude_wm:claude-opus-4-8:retrieval claude_wm:claude-opus-5:retrieval
ARMS=("$@")
N="${#ARMS[@]}"
{ [ "$N" -ge 1 ] && [ "$N" -le 8 ]; } || { echo "FATAL: need 1..8 arms, got ${N}" >&2; exit 1; }

REPO="${HV_PTB_DIR:-/rmeng_data/robtang/ptb-hvrecipe}"
TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-google/gemma-3-4b-pt}"
NUM_HOURS="${PTB_NUM_HOURS:-10}"
STAGGER="${PTB_STAGGER:-60}"
PRIOR_RUNS="${PRIOR_RUNS:-}"
WM_MEMORY="${WM_MEMORY:-}"
PRIOR_RUNS_FOR="${PRIOR_RUNS_FOR:-claude_fulltraj_noawm,claude_wm}"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "cells=${N} arms=${ARMS[*]} task=${TASK} model=${MODEL_TO_TRAIN} hours=${NUM_HOURS}"
echo "prior_runs=${PRIOR_RUNS:-<none>} wm_memory=${WM_MEMORY:-<none>} prior_runs_for=${PRIOR_RUNS_FOR}"
[ -z "$PRIOR_RUNS" ] || [ -f "$PRIOR_RUNS/INDEX.md" ] || { echo "FATAL: ${PRIOR_RUNS}/INDEX.md missing — run tools/build_prior_runs.py" >&2; exit 1; }

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPO" || { echo "FATAL: no checkout at $REPO -- run rollout/setup.sh" >&2; exit 1; }
export REPO_ROOT="$(pwd)"
source src/commit_utils/set_env_vars.sh
unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE
grep -q "POST_TRAIN_BENCH_EXTRA_BINDS" src/run_task.sh || { echo "FATAL: run_task.sh lacks the extra-binds patch -- run rollout/setup.sh" >&2; exit 1; }

echo "repo_sha=$(git rev-parse HEAD)"
for a in claude_fulltraj_noawm claude_wm; do
    echo "solve_sh_sha256[$a]=$(sha256sum agents/$a/solve.sh | cut -d' ' -f1)"
done

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
RUN_TASK="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "${SCRATCH}/wm-src-${SLURM_JOB_ID}")"
echo "run_task=${RUN_TASK}"

LOGDIR="${REPO}/slurm_logs"; mkdir -p "$LOGDIR"
declare -a PIDS=()
for (( gpu=0; gpu<N; gpu++ )); do
    arm="${ARMS[$gpu]}"
    agent="${arm%%:*}"
    cfg="${arm#*:}"
    [ "$agent" != "$arm" ] || { echo "FATAL: arm ${arm} must be <agent>:<config>" >&2; exit 1; }
    binds=()
    wants_prior=0
    case ",${PRIOR_RUNS_FOR}," in *",${agent},"*) wants_prior=1;; esac
    if [ -n "$PRIOR_RUNS" ] && [ "$wants_prior" = 1 ]; then binds+=("${PRIOR_RUNS}:/home/ben/prior_runs:ro"); fi
    if [ "$agent" = "claude_wm" ] && [ -n "$WM_MEMORY" ]; then
        mkdir -p "$WM_MEMORY"
        case "$cfg" in *:ro) binds+=("${WM_MEMORY}:/home/ben/wm-memory:ro");; *) binds+=("${WM_MEMORY}:/home/ben/wm-memory");; esac
    fi
    case "$agent" in
        claude_fulltraj_noawm) prompt="prompt_fulltraj";;
        claude_wm) if [ -n "$PRIOR_RUNS" ] && [ "$wants_prior" = 1 ]; then prompt="prompt_wm_fulltraj"; else prompt="prompt_wm"; fi;;
        *) prompt="prompt";;
    esac
    cell="${SLURM_JOB_ID}_g${gpu}"
    safe="$(echo "$arm" | tr '/:,' '___')"
    log="${LOGDIR}/wm-${SLURM_JOB_ID}-g${gpu}-${safe}.log"
    (
        sleep $(( gpu * STAGGER ))
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${gpu}"
        export POST_TRAIN_BENCH_PROMPT="${prompt}"
        export POST_TRAIN_BENCH_EXTRA_BINDS="$(IFS=,; echo "${binds[*]:-}")"
        echo "cell start gpu=${gpu} agent=${agent} cfg=${cfg} prompt=${prompt} binds=${POST_TRAIN_BENCH_EXTRA_BINDS:-<none>} at $(date --iso-8601=seconds)"
        bash "${RUN_TASK}" "${TASK}" "${agent}" "${MODEL_TO_TRAIN}" "${cell}" "${NUM_HOURS}" "${cfg}" 1
    ) > "${log}" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${gpu} arm=${arm} prompt=${prompt} cell=${cell} pid=${PIDS[$gpu]} log=${log}"
done

rc_all=0
for (( gpu=0; gpu<N; gpu++ )); do
    wait "${PIDS[$gpu]}"; rc=$?
    echo "cell_exit gpu=${gpu} arm=${ARMS[$gpu]} cell=${SLURM_JOB_ID}_g${gpu} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done
echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
