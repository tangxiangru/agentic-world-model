#!/bin/bash
#SBATCH --job-name=hv-recipe
#SBATCH --partition=a3
# a3's 14 nodes are all spoken for: 11 by reservation robtang-a3, the other 3 by
# another user's UNLIMITED jobs. A pack submitted without the reservation waits
# on three nodes that never free and squeue reports (Resources) forever.
#SBATCH --reservation=robtang-a3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
# 5h was too tight to be safe. run_task.sh caps the agent at NUM_HOURS*60+5 = 185
# min, the scoring ladder after it has been measured at up to 59 min, and the last
# cell does not start until gpu*STAGGER = 5 min in -- 4h10 of committed work
# against a 5h wall, with model download and container setup unaccounted. A pack
# that dies at the wall loses every cell AFTER the training that earned it, which
# is the most expensive way to fail. 6h30 leaves the ladder 3h25.
#SBATCH --time=06:30:00
#SBATCH --output=/rmeng_data/robtang/ptb-hvrecipe/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/ptb-hvrecipe/slurm_logs/%x-%j.err
set -uo pipefail

# ---- hard fence: never occupy froilanchoi's three a3 nodes ----------------------------
# a3 is 23 nodes; 20 sit in reservations robtang-a3 and robtang-a3-ondem, and the
# complement is exactly slurm2-a3nodeset0-[0-2], which are froilanchoi's. A job reaches
# them only when the submit-side guard was bypassed: `--reservation=` with an EMPTY value
# is a PIN onto those three, not a relaxation, and so is any --exclude naming all twenty
# of mine. airs-ops/healthy_nodes.sh drops them from every exclude list it builds, but
# that binds only the callers who ask it. On 2026-09-04 jobs 91755 and 91723 each ran two
# hours on 0-0 and 0-1 before anyone noticed, and froilanchoi refilled all three within
# twenty minutes of being given them back -- so this is contention, not spare capacity.
# Refuse here instead. It forfeits nothing: the node was never ours to run on. Kept ahead
# of every other statement so nothing is half-created on the way out.
_fence_nodes="${SLURM_JOB_NODELIST:-}"
if [ -n "$_fence_nodes" ] && command -v scontrol >/dev/null 2>&1; then
    _fence_nodes="$(scontrol show hostnames "$_fence_nodes" 2>/dev/null | tr '\n' ' ' || true)"
fi
[ -n "${_fence_nodes// /}" ] || _fence_nodes="${SLURMD_NODENAME:-$(hostname -s)}"
for _fence_n in $_fence_nodes; do
    case "$_fence_n" in
        slurm2-a3nodeset0-0|slurm2-a3nodeset0-1|slurm2-a3nodeset0-2)
            echo "FATAL: allocated ${_fence_n}, which belongs to froilanchoi -- one of the three" >&2
            echo "       a3 nodes outside reservations robtang-a3 / robtang-a3-ondem. Resubmit" >&2
            echo "       WITH a reservation; never pass an empty --reservation= on partition a3." >&2
            exit 1 ;;
    esac
done
unset _fence_nodes _fence_n

# The crossed rollout, executor half: one deterministic trainer, N recipes, one
# GPU each on one exclusive node. Unlike ptb_pack.sbatch the ARM is the
# AGENT_CONFIG, not the agent -- the agent is always hv_recipe, because holding
# the executor fixed is the entire experiment.
#
# Interleave recipes across nodes and GPU indices. "Which recipe" must not be the
# same question as "which node" or "which card": these are 8 cards sharing one
# host's PCIe, memory bandwidth and page cache, and a recipe pinned to a node
# would carry that node's luck.
#
# The two packs actually submitted. Seeds alternate within a node and swap between
# them, and node B's order is rotated so no recipe lands on the same card index
# twice -- otherwise "recipe" and "GPU slot" are the same variable:
#
#   sbatch rollout/hv_pack.sbatch r733.s0 r699.s1 r600.s0 r544.s1 r401.s0 r042.s1 base
#   sbatch rollout/hv_pack.sbatch r042.s0 r401.s1 r544.s0 r600.s1 r699.s0 r733.s1 base
#
# Usage:  sbatch rollout/hv_pack.sbatch r733.s0 r699.s1 ... [<= 8]
ARMS=("$@")
N="${#ARMS[@]}"
{ [ "$N" -ge 1 ] && [ "$N" -le 8 ]; } || { echo "FATAL: need 1..8 arms, got ${N}" >&2; exit 1; }

REPO="${HV_PTB_DIR:-/rmeng_data/robtang/ptb-hvrecipe}"
TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-Qwen/Qwen3-1.7B-Base}"
NUM_HOURS="${PTB_NUM_HOURS:-3}"
STAGGER="${PTB_STAGGER:-45}"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "cells=${N} arms=${ARMS[*]} task=${TASK} model=${MODEL_TO_TRAIN} hours=${NUM_HOURS}"

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }
echo "apptainer=$(apptainer --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPO" || { echo "FATAL: no checkout at $REPO -- run rollout/setup.sh" >&2; exit 1; }
export REPO_ROOT="$(pwd)"
source src/commit_utils/set_env_vars.sh

# huggingface_hub ranks HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > "$HF_HOME/hub",
# and this cluster exports the middle one globally, so an --env HF_HOME loses and
# the download lands on the 64 MiB container root with an ENOSPC that names
# neither a cache nor a filesystem.
unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE
echo "hf_home=${HF_HOME:-<unset>} hub_cache_overrides=cleared"
echo "isolate_gpus=${POST_TRAIN_BENCH_ISOLATE_GPUS:-<unset>} eval_gpu_reap=${POST_TRAIN_BENCH_EVAL_GPU_REAP:-<unset>}"
echo "results_dir=${POST_TRAIN_BENCH_RESULTS_DIR}"

# Echo the launch SHA and the agents' own hashes with the run. A recipe table is
# the payload of this experiment; a delta quoted against an unrecorded solve.sh
# is not a delta. repo_sha covers neither agent -- both live untracked in this
# checkout, installed by rollout/setup.sh from the agentic-world-model repo -- so
# the hashes are the only record of what actually ran.
#
# hv_noop used to be omitted, and it is the file that pins the CONTROL's decode.
# An unrecorded control is not a floor: every recipe delta is quoted against it.
echo "repo_sha=$(git rev-parse HEAD)"
for a in hv_recipe hv_noop; do
    echo "solve_sh_sha256[$a]=$(sha256sum "agents/$a/solve.sh" | cut -d' ' -f1)"
done

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH on $(hostname)" >&2; exit 1; }
[ -w "$SCRATCH" ] || { echo "FATAL: $SCRATCH not writable on $(hostname)" >&2; exit 1; }
echo "scratch=$SCRATCH free=$(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"

# bash reads a script incrementally, so a commit in this checkout replaces the
# inode under a running job and the ESTALE lands hours later. Pin to node-local
# disk once per job; every cell reads it, none writes it.
RUN_TASK="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "${SCRATCH}/hv-src-${SLURM_JOB_ID}")"
echo "run_task=${RUN_TASK}"

LOGDIR="${REPO}/slurm_logs"
mkdir -p "$LOGDIR"
declare -a PIDS=()
for (( gpu=0; gpu<N; gpu++ )); do
    arm="${ARMS[$gpu]}"
    agent="hv_recipe"
    cfg="${arm}"
    # `base` is the untrained control: hv_noop ships the base model unmodified,
    # so it scores the floor every recipe has to clear. Without it a collapsed
    # spread cannot be told apart from a trainer that trains nothing.
    if [ "$arm" = "base" ]; then agent="hv_noop"; cfg="none"; fi
    cell="${SLURM_JOB_ID}_g${gpu}"
    log="${LOGDIR}/hv-${SLURM_JOB_ID}-g${gpu}-${arm}.log"
    (
        sleep $(( gpu * STAGGER ))
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${gpu}"
        echo "cell start gpu=${gpu} arm=${arm} agent=${agent} cell=${cell} at $(date --iso-8601=seconds)"
        bash "${RUN_TASK}" "${TASK}" "${agent}" "${MODEL_TO_TRAIN}" "${cell}" "${NUM_HOURS}" "${cfg}" 1
    ) > "${log}" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${gpu} arm=${arm} cell=${cell} pid=${PIDS[$gpu]} log=${log}"
done

rc_all=0
for (( gpu=0; gpu<N; gpu++ )); do
    wait "${PIDS[$gpu]}"
    rc=$?
    echo "cell_exit gpu=${gpu} arm=${ARMS[$gpu]} cell=${SLURM_JOB_ID}_g${gpu} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
