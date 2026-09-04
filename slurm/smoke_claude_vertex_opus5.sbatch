#!/usr/bin/env bash
#SBATCH --job-name=vertex-opus5-probe
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset1-1,slurm2-a3nodeset-0,slurm2-a3nodeset-1,slurm2-a3nodeset1-0,slurm2-a3nodeset1-2,slurm2-a3nodeset-2,slurm2-a3nodeset0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/rmeng_data/robtang/agentic-world-model/slurm/logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/agentic-world-model/slurm/logs/%x-%j.err

set -euo pipefail

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
umask 077

repo_root="/rmeng_data/robtang/agentic-world-model"
claude_bin="/home/robtang_google_com/.local/bin/claude"

# On this cluster the outer sbatch shell can inherit every host GPU. Slurm's
# task cgroup is applied correctly to srun job steps, so never run the agent in
# the outer shell.
if [ "${AWM_INSIDE_GPU_STEP:-0}" != "1" ]; then
    exec srun --exact --kill-on-bad-exit=1 \
        --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
        env AWM_INSIDE_GPU_STEP=1 "$0"
fi

probe_home="${repo_root}/.cache/claude-vertex-probe/${SLURM_JOB_ID}"

mkdir -p "$probe_home" "${probe_home}/.config" "${probe_home}/.cache" "${probe_home}/tmp"

export HOME="$probe_home"
export XDG_CONFIG_HOME="${probe_home}/.config"
export XDG_CACHE_HOME="${probe_home}/.cache"
export TMPDIR="${probe_home}/tmp"

export CLAUDE_CODE_USE_VERTEX="1"
export ANTHROPIC_VERTEX_PROJECT_ID="sercan-v1"
export GOOGLE_CLOUD_PROJECT="sercan-v1"
export CLOUD_ML_REGION="global"
export ANTHROPIC_MODEL="claude-opus-5"
export ANTHROPIC_DEFAULT_MODEL="claude-opus-5"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-5"
export CLAUDE_CODE_NO_MODEL_FALLBACK="1"
export VERTEX_REGION_CLAUDE_5_OPUS="global"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

visible_gpu_count="$(nvidia-smi -L | sed '/^[[:space:]]*$/d' | wc -l)"
if [ "$visible_gpu_count" -ne 1 ]; then
    echo "ERROR: expected one visible GPU, found ${visible_gpu_count}." >&2
    exit 20
fi
nvidia-smi -L

if ! curl -fsS --max-time 5 -H 'Metadata-Flavor: Google' -o /dev/null \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'; then
    echo "ERROR: Vertex metadata credentials are unavailable." >&2
    exit 21
fi

result_json="$($claude_bin --bare --print \
    --model claude-opus-5 \
    --effort low \
    --tools "" \
    --max-turns 1 \
    --no-session-persistence \
    --output-format json \
    'Reply with exactly VERTEX_OPUS5_OK and nothing else.')"

printf '%s\n' "$result_json" | jq -e '
    .is_error == false and
    .result == "VERTEX_OPUS5_OK" and
    .modelUsage["claude-opus-5"].provider == "vertex"
' >/dev/null

printf '%s\n' "$result_json" | jq '{
    result,
    total_cost_usd,
    model: .modelUsage["claude-opus-5"].canonicalModel,
    provider: .modelUsage["claude-opus-5"].provider,
    input_tokens: .modelUsage["claude-opus-5"].inputTokens,
    output_tokens: .modelUsage["claude-opus-5"].outputTokens
}'

echo "VERTEX_OPUS5_PROBE_PASSED"
