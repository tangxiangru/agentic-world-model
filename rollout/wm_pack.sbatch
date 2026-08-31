#!/bin/bash
set -euo pipefail

# The scientist x information study: exactly one cell in one OS-isolated GPU
# allocation per invocation.  A site-specific untracked scheduler wrapper
# should submit the 36 matrix entries as 36 independent one-GPU jobs.
# Scheduler resources, partition/reservation, logs, module setup, and all host
# paths belong in the site's untracked submit wrapper or local environment.
# This file can be submitted as a batch script or run directly inside an
# existing allocation.
#
# Each argument explicitly names the information condition. The harness derives
# the agent, prompt, mounts, and held-out/read-only WMA config from it:
#
#   c1:<model>:<prior-scope>:<rep>          raw files, no WMA
#   c2:<model>:llm:<prior-scope>:<rep>      raw files + autonomous WMA
#   c3:<model>:llm:<memory-sides>:<rep>     cards through autonomous WMA only
#
# ``rep`` is 1 or 2 and is part of the PTB result identifier.  This makes the
# full 3 x 3 x 2 x 2 design explicit and prevents an accidental overwrite or
# relabelled duplicate from masquerading as a second repetition.
#
# Environment (host side, read here and by run_task.sh):
#   PTB_TASK, PTB_MODEL, PTB_NUM_HOURS        task, base model, hours     (gsm8k, google/gemma-3-4b-pt, 10)
#   HV_PTB_DIR   private checkout prepared by rollout/setup.sh (required)
#   PRIOR_RUNS   host dir from tools/build_prior_runs.py -> bound read-only at /home/ben/prior_runs
#   WM_MEMORY    C3 seeded card memory                  -> bound read-only at /home/ben/wm-memory
#   AWM_REPO_COMMIT  exact 40-hex harness/runtime commit for every cell (required)
#   AWM_WMA_MODEL    one fixed, explicit Vertex Claude model for every WMA cell
#   AWM_SCIENTIST_MODEL_ID_4_6, AWM_SCIENTIST_MODEL_ID_4_8,
#   AWM_SCIENTIST_MODEL_ID_5_0 exact stream-reported provider IDs for aliases
#   AWM_CLAUDE_CLI_VERSION exact npm package version installed in every cell
#   AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT exact expected `claude --version`
#   AWM_PRIOR_CORPUS_MANIFEST_SHA256 expected corpus-manifest.json digest for C1/C2
#   AWM_CARD_CORPUS_MANIFEST_SHA256 combined selected-side manifest digest for C3
#   PTB_GPU_SLOTS comma-separated GPU identifiers, unless inherited through CUDA_VISIBLE_DEVICES
#   PTB_RUN_ID    filesystem-safe run identifier (defaults to Slurm job id or local-<pid>)
#   PTB_LOG_DIR   per-cell log directory (defaults inside the private checkout)
#   AWM_STUDY_SMOKE=1 with explicit PTB_NUM_HOURS=1 runs a labelled smoke;
#                  production rejects every duration other than 10 hours
#   VERTEX_ADC_FILE optional host ADC JSON; otherwise use the node service account
#   Vertex routing variables listed in agents/claude_*/env_passthrough.txt
#
# Example after a site wrapper has provided a one-GPU cgroup/allocation:
#
#   PTB_GPU_SLOTS=0 bash rollout/wm_pack.sbatch c1:claude-opus-4-6:train:1
SPECS=("$@")
N="${#SPECS[@]}"
[ "$N" -eq 1 ] || {
    echo "FATAL: GPU safety requires exactly one cell per one-GPU invocation" >&2
    exit 2
}

: "${HV_PTB_DIR:?set HV_PTB_DIR to the private checkout prepared by rollout/setup.sh}"
REPO="${HV_PTB_DIR}"
TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-google/gemma-3-4b-pt}"
NUM_HOURS="${PTB_NUM_HOURS:-10}"
STUDY_SMOKE="${AWM_STUDY_SMOKE:-0}"
STAGGER="${PTB_STAGGER:-60}"
PRIOR_RUNS="${PRIOR_RUNS:-}"
WM_MEMORY="${WM_MEMORY:-}"
VERTEX_ADC_FILE="${VERTEX_ADC_FILE:-}"
AWM_REPO_COMMIT="${AWM_REPO_COMMIT:-}"
AWM_WMA_MODEL="${AWM_WMA_MODEL:-}"
AWM_SCIENTIST_MODEL_ID_4_6="${AWM_SCIENTIST_MODEL_ID_4_6:-}"
AWM_SCIENTIST_MODEL_ID_4_8="${AWM_SCIENTIST_MODEL_ID_4_8:-}"
AWM_SCIENTIST_MODEL_ID_5_0="${AWM_SCIENTIST_MODEL_ID_5_0:-}"
AWM_CLAUDE_CLI_VERSION="${AWM_CLAUDE_CLI_VERSION:-}"
AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT="${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT:-}"
AWM_PRIOR_CORPUS_MANIFEST_SHA256="${AWM_PRIOR_CORPUS_MANIFEST_SHA256:-}"
AWM_CARD_CORPUS_MANIFEST_SHA256="${AWM_CARD_CORPUS_MANIFEST_SHA256:-}"
RUN_ID="${PTB_RUN_ID:-${SLURM_JOB_ID:-local-$$}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "FATAL: PTB_RUN_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
}
[ "${TASK}" = gsm8k ] || { echo "FATAL: this 36-cell study requires PTB_TASK=gsm8k" >&2; exit 2; }
[ "${MODEL_TO_TRAIN}" = google/gemma-3-4b-pt ] || {
    echo "FATAL: this 36-cell study requires PTB_MODEL=google/gemma-3-4b-pt" >&2
    exit 2
}
case "${STUDY_SMOKE}" in
    0)
        STUDY_MODE=production
        [ "${NUM_HOURS}" = 10 ] || {
            echo "FATAL: production study cells require PTB_NUM_HOURS=10" >&2
            exit 2
        }
        ;;
    1)
        STUDY_MODE=smoke
        [ "${PTB_NUM_HOURS+x}" = x ] && [ "${NUM_HOURS}" = 1 ] || {
            echo "FATAL: AWM_STUDY_SMOKE=1 requires explicit PTB_NUM_HOURS=1" >&2
            exit 2
        }
        ;;
    *) echo "FATAL: AWM_STUDY_SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
[[ "${AWM_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "FATAL: every cell requires AWM_REPO_COMMIT as a full 40-hex harness commit" >&2
    exit 2
}
PACK_AWM_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel 2>/dev/null)" || {
    echo "FATAL: wm_pack.sbatch must run from its auditable AWM Git checkout" >&2
    exit 2
}
git -C "${PACK_AWM_ROOT}" cat-file -e "${AWM_REPO_COMMIT}^{commit}" 2>/dev/null && \
    git -C "${PACK_AWM_ROOT}" show "${AWM_REPO_COMMIT}:rollout/wm_pack.sbatch" | \
    cmp -s - "${BASH_SOURCE[0]}" || {
    echo "FATAL: live wm_pack.sbatch does not byte-match AWM_REPO_COMMIT" >&2
    exit 2
}

declare -a CONDITIONS=() AGENTS=() CONFIGS=() PROMPTS=() REPETITIONS=()
declare -a EXPECTED_SCIENTIST_MODELS=() CELL_IDS=()
declare -A SEEN_CELL_KEYS=()
need_prior=0
need_wma=0
need_memory=0
need_test_side=0
need_test_prior=0
study_sides=""
for spec in "${SPECS[@]}"; do
    IFS=: read -r condition scientist field3 field4 field5 extra <<< "${spec}"
    case "${scientist}" in
        claude-opus-4-6) expected_scientist_model="${AWM_SCIENTIST_MODEL_ID_4_6}" ;;
        claude-opus-4-8) expected_scientist_model="${AWM_SCIENTIST_MODEL_ID_4_8}" ;;
        claude-opus-5) expected_scientist_model="${AWM_SCIENTIST_MODEL_ID_5_0}" ;;
        *) echo "FATAL: unsupported scientist model in ${spec}" >&2; exit 2;;
    esac
    [ -n "${expected_scientist_model}" ] && \
        [[ "${expected_scientist_model}" != *[[:space:]]* ]] || {
        echo "FATAL: ${scientist} needs its exact stream-reported provider ID in the untracked scientist-model mapping" >&2
        exit 2
    }
    case "${condition}" in
        c1)
            declared_sides="${field3}"
            repetition="${field4}"
            { [ "${declared_sides}" = train ] || [ "${declared_sides}" = train,test ]; } && \
                { [ "${repetition}" = 1 ] || [ "${repetition}" = 2 ]; } && \
                [ -z "${field5}${extra}" ] || {
                echo "FATAL: C1 syntax is c1:<model>:<train|train,test>:<1|2>, got ${spec}" >&2
                exit 2
            }
            agent=claude_fulltraj_noawm
            config="${scientist}:${declared_sides}"
            prompt=prompt_fulltraj
            need_prior=1
            [ "${declared_sides}" = train,test ] && need_test_prior=1
            ;;
        c2|c3)
            expected_arm=llm
            arm="${field3}"
            declared_sides="${field4}"
            repetition="${field5}"
            [ "${arm}" = "${expected_arm}" ] && \
                { [ "${declared_sides}" = train ] || [ "${declared_sides}" = train,test ]; } && \
                { [ "${repetition}" = 1 ] || [ "${repetition}" = 2 ]; } && \
                [ -z "${extra}" ] || {
                echo "FATAL: ${condition^^} syntax is ${condition}:<model>:${expected_arm}:<train|train,test>:<1|2>, got ${spec}" >&2
                exit 2
            }
            agent=claude_wm
            config="${scientist}:${arm}:${declared_sides}:ro"
            if [ "${condition}" = c2 ]; then
                prompt=prompt_wm_fulltraj
                need_prior=1
                [ "${declared_sides}" = train,test ] && need_test_prior=1
            else
                prompt=prompt_wm
                need_memory=1
                [ "${declared_sides}" = train,test ] && need_test_side=1
            fi
            need_wma=1
            ;;
        *) echo "FATAL: condition must be c1, c2, or c3 in ${spec}" >&2; exit 2;;
    esac
    cell_key="${condition}|${scientist}|${declared_sides}|${repetition}"
    [ -z "${SEEN_CELL_KEYS[$cell_key]+x}" ] || {
        echo "FATAL: duplicate study cell in one pack: ${cell_key}" >&2
        exit 2
    }
    SEEN_CELL_KEYS[$cell_key]=1
    if [ -z "${study_sides}" ]; then
        study_sides="${declared_sides}"
    elif [ "${study_sides}" != "${declared_sides}" ]; then
        echo "FATAL: one pack cannot mix ${study_sides} and ${declared_sides} prior scopes" >&2
        exit 2
    fi
    CONDITIONS+=("${condition}")
    AGENTS+=("${agent}")
    CONFIGS+=("${config}")
    PROMPTS+=("${prompt}")
    REPETITIONS+=("${repetition}")
    EXPECTED_SCIENTIST_MODELS+=("${expected_scientist_model}")
    safe_spec="$(printf '%s' "${spec}" | tr '/:,' '___')"
    CELL_IDS+=("${RUN_ID}_${STUDY_MODE}_${safe_spec}")
done

[ "${CLAUDE_CODE_USE_VERTEX:-}" = 1 ] || { echo "FATAL: export CLAUDE_CODE_USE_VERTEX=1" >&2; exit 2; }
[ -n "${ANTHROPIC_VERTEX_PROJECT_ID:-}" ] || { echo "FATAL: export ANTHROPIC_VERTEX_PROJECT_ID" >&2; exit 2; }
[[ "${AWM_CLAUDE_CLI_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]] || {
    echo "FATAL: AWM_CLAUDE_CLI_VERSION must be one exact npm semantic version" >&2
    exit 2
}
[ -n "${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT}" ] && \
    [[ "${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT}" != *$'\n'* ]] && \
    [[ "${AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT}" != *$'\r'* ]] || {
    echo "FATAL: AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT must be the exact one-line output" >&2
    exit 2
}
if [ -n "${VERTEX_ADC_FILE}" ]; then
    [ -f "${VERTEX_ADC_FILE}" ] || { echo "FATAL: VERTEX_ADC_FILE is not a file" >&2; exit 2; }
fi
if [ "${need_prior}" = 1 ]; then
    [ -n "${PRIOR_RUNS}" ] || { echo "FATAL: C1/C2 requires PRIOR_RUNS" >&2; exit 2; }
    [ -f "${PRIOR_RUNS}/INDEX.md" ] && [ -s "${PRIOR_RUNS}/index.jsonl" ] || {
        echo "FATAL: ${PRIOR_RUNS} lacks INDEX.md or a non-empty index.jsonl" >&2
        exit 2
    }
    [[ "${AWM_PRIOR_CORPUS_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "FATAL: C1/C2 requires AWM_PRIOR_CORPUS_MANIFEST_SHA256 as 64 lowercase hex" >&2
        exit 2
    }
    grep -q '"side": "train"' "${PRIOR_RUNS}/index.jsonl" || {
        echo "FATAL: PRIOR_RUNS has no train-side runs" >&2
        exit 2
    }
    if [ "${need_test_prior}" = 1 ]; then
        grep -q '"side": "test"' "${PRIOR_RUNS}/index.jsonl" || {
            echo "FATAL: a C2 train,test cell requires test-side runs in PRIOR_RUNS" >&2
            exit 2
        }
    fi
    if [ -n "${study_sides}" ]; then
        prior_has_test=0
        grep -q '"side": "test"' "${PRIOR_RUNS}/index.jsonl" && prior_has_test=1
        if { [ "${study_sides}" = train,test ] && [ "${prior_has_test}" != 1 ]; } ||
           { [ "${study_sides}" = train ] && [ "${prior_has_test}" = 1 ]; }; then
            echo "FATAL: PRIOR_RUNS scope does not match declared study sides ${study_sides}" >&2
            exit 2
        fi
    fi
fi
if [ "${need_memory}" = 1 ]; then
    [ -s "${WM_MEMORY}/structured/cards.jsonl" ] || {
        echo "FATAL: C3 requires seeded WM_MEMORY/structured/cards.jsonl" >&2
        exit 2
    }
    grep -q '"split_side": "train"' "${WM_MEMORY}/structured/cards.jsonl" || {
        echo "FATAL: WM_MEMORY has no train-side cards" >&2
        exit 2
    }
    [[ "${AWM_CARD_CORPUS_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "FATAL: C3 requires AWM_CARD_CORPUS_MANIFEST_SHA256 as 64 lowercase hex" >&2
        exit 2
    }
    if [ "${need_test_side}" = 1 ]; then
        grep -q '"split_side": "test"' "${WM_MEMORY}/structured/cards.jsonl" || {
            echo "FATAL: a train,test cell requires test-side cards in WM_MEMORY" >&2
            exit 2
        }
    fi
fi
if [ "${need_wma}" = 1 ]; then
    [ -n "${AWM_WMA_MODEL}" ] && [[ "${AWM_WMA_MODEL}" != *[[:space:]]* ]] || {
        echo "FATAL: C2/C3 requires one explicit AWM_WMA_MODEL (no whitespace)" >&2
        exit 2
    }
    case "${AWM_WMA_MODEL,,}" in
        default|opus|sonnet|haiku)
            echo "FATAL: AWM_WMA_MODEL must be a pinned version, not ${AWM_WMA_MODEL}" >&2
            exit 2
            ;;
    esac
fi
# Respect a scheduler-provided visibility list, or an explicit list from an
# untracked local wrapper. Never guess host GPU indices: doing so can escape the
# intended allocation on sites that do not enforce GPU cgroups.
GPU_SLOTS_RAW="${PTB_GPU_SLOTS:-${CUDA_VISIBLE_DEVICES:-}}"
[ -n "${GPU_SLOTS_RAW}" ] || {
    echo "FATAL: set PTB_GPU_SLOTS or launch with CUDA_VISIBLE_DEVICES set" >&2
    exit 2
}
IFS=, read -r -a GPU_SLOTS <<< "${GPU_SLOTS_RAW}"
[ "${#GPU_SLOTS[@]}" -eq 1 ] || {
    echo "FATAL: GPU safety requires exactly one declared GPU slot; use one OS/cgroup-isolated allocation per cell" >&2
    exit 2
}
for (( gpu=0; gpu<N; gpu++ )); do
    [ -n "${GPU_SLOTS[$gpu]}" ] && [[ "${GPU_SLOTS[$gpu]}" != *[[:space:]]* ]] || {
        echo "FATAL: invalid empty/whitespace GPU slot at position ${gpu}" >&2
        exit 2
    }
done
declare -A SEEN_GPU_SLOTS=()
for (( gpu=0; gpu<N; gpu++ )); do
    slot="${GPU_SLOTS[$gpu]}"
    [ -z "${SEEN_GPU_SLOTS[$slot]+x}" ] || {
        echo "FATAL: duplicate GPU identifier assigned to multiple cells: ${slot}" >&2
        exit 2
    }
    SEEN_GPU_SLOTS[$slot]=1
done

echo "node=$(hostname) run=${RUN_ID} start=$(date --iso-8601=seconds)"
echo "cells=${N} mode=${STUDY_MODE} specs=${SPECS[*]} task=${TASK} model=${MODEL_TO_TRAIN} hours=${NUM_HOURS}"
echo "gpu_slots=${GPU_SLOTS[*]}"
echo "prior_runs=${PRIOR_RUNS:-<none>} wm_memory=${WM_MEMORY:-<none>} awm_commit=${AWM_REPO_COMMIT:-<none>} wma_model=${AWM_WMA_MODEL:-<none>}"
echo "vertex_auth=$([ -n "${VERTEX_ADC_FILE}" ] && echo bound-adc || echo attached-service-account)"

cd "$REPO" || { echo "FATAL: no checkout at $REPO -- run rollout/setup.sh" >&2; exit 1; }
export REPO_ROOT="$(pwd)"
AWM_PTB_COMMIT="$(git rev-parse HEAD)"
[[ "${AWM_PTB_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "FATAL: private PTB checkout does not resolve to a full commit" >&2
    exit 2
}
PTB_GIT_DIR="$(git rev-parse --absolute-git-dir)"
[ -f "${PTB_GIT_DIR}/awm-study-harness-commit" ] && \
    [ "$(tr -d '[:space:]' < "${PTB_GIT_DIR}/awm-study-harness-commit")" = "${AWM_REPO_COMMIT,,}" ] || {
    echo "FATAL: private PTB checkout was not prepared from this exact harness commit" >&2
    exit 2
}
[ -f "${PTB_GIT_DIR}/awm-study-ptb-commit" ] && \
    [ "$(tr -d '[:space:]' < "${PTB_GIT_DIR}/awm-study-ptb-commit")" = "${AWM_PTB_COMMIT,,}" ] || {
    echo "FATAL: private PTB checkout HEAD does not match its setup attestation" >&2
    exit 2
}
SURFACE_ATTESTER=src/commit_utils/attest_study_surface.py
[ -x "${SURFACE_ATTESTER}" ] && \
    git -C "${PACK_AWM_ROOT}" show "${AWM_REPO_COMMIT}:rollout/attest_ptb_surface.py" | \
    cmp -s - "${SURFACE_ATTESTER}" || {
    echo "FATAL: private PTB surface attester does not match the harness commit" >&2
    exit 2
}
AWM_PTB_SURFACE_MANIFEST_SHA256="$(python3 "${SURFACE_ATTESTER}" verify \
    --root "${REPO_ROOT}" --manifest "${PTB_GIT_DIR}/awm-study-surface.json" \
    --awm-commit "${AWM_REPO_COMMIT}" --ptb-commit "${AWM_PTB_COMMIT}")" || exit 2
[[ "${AWM_PTB_SURFACE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "FATAL: invalid PTB study-surface manifest attestation" >&2
    exit 2
}
echo "ptb_surface_manifest_sha256=${AWM_PTB_SURFACE_MANIFEST_SHA256}"
source src/commit_utils/set_env_vars.sh
unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE
grep -q "POST_TRAIN_BENCH_EXTRA_BINDS" src/run_task.sh || { echo "FATAL: run_task.sh lacks the extra-binds patch -- run rollout/setup.sh" >&2; exit 1; }
[ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-}" = 1 ] || {
    echo "FATAL: private PTB .env must set POST_TRAIN_BENCH_ISOLATE_GPUS=1" >&2
    exit 2
}
case "${POST_TRAIN_BENCH_EVAL_GPU_REAP:-}" in
    own|none) ;;
    *) echo "FATAL: set POST_TRAIN_BENCH_EVAL_GPU_REAP=own or none" >&2; exit 2;;
esac
# The study installs an exact npm version itself. Keep the copied upstream
# updater inert and disable subscription-only reward judges for this
# Vertex-only, objectively-scored GSM8K study.
export POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip

command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "FATAL: no nvidia-smi on $(hostname)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CORPUS_VALIDATOR="agents/claude_fulltraj_noawm/payload/validate_study_corpus.py"
[ -x "${CORPUS_VALIDATOR}" ] || {
    echo "FATAL: private PTB checkout lacks the study corpus validator -- run rollout/setup.sh" >&2
    exit 2
}
if [ "${need_prior}" = 1 ]; then
    python3 "${CORPUS_VALIDATOR}" raw "${PRIOR_RUNS}" \
        --sides "${study_sides}" \
        --expected-manifest-sha256 "${AWM_PRIOR_CORPUS_MANIFEST_SHA256}"
fi
if [ "${need_memory}" = 1 ]; then
    python3 "${CORPUS_VALIDATOR}" cards "${WM_MEMORY}" --sides "${study_sides}" \
        --expected-manifest-sha256 "${AWM_CARD_CORPUS_MANIFEST_SHA256}"
fi

echo "repo_sha=$(git rev-parse HEAD)"
for a in claude_fulltraj_noawm claude_wm; do
    echo "solve_sh_sha256[$a]=$(sha256sum agents/$a/solve.sh | cut -d' ' -f1)"
done

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
PINNED_REPO="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "${SCRATCH}/wm-src-${CELL_IDS[0]}")"
[ -d "${PINNED_REPO}" ] && [ ! -L "${PINNED_REPO}" ] && \
    [ -x "${PINNED_REPO}/src/run_task.sh" ] || {
    echo "FATAL: source pin helper did not return a valid immutable snapshot root" >&2
    exit 2
}
echo "pinned_repo=${PINNED_REPO}"

# Reserve the exact upstream result directories atomically. The identifier
# encodes condition, scientist, scope, and repetition, so C2/C3 and separate
# packs cannot overwrite one another when they share PTB_RUN_ID.
[ -n "${POST_TRAIN_BENCH_RESULTS_DIR:-}" ] && \
    [ -d "${POST_TRAIN_BENCH_RESULTS_DIR}" ] && \
    [ ! -L "${POST_TRAIN_BENCH_RESULTS_DIR}" ] || {
    echo "FATAL: private PTB results root is missing, linked, or invalid" >&2
    exit 2
}
RESULT_PREFIX_SAFE="$(printf '%s' "${MODEL_TO_TRAIN}" | tr '/:[]' '____')"
declare -a RESULT_DIRS=()
for (( gpu=0; gpu<N; gpu++ )); do
    config_safe="$(printf '%s' "${CONFIGS[$gpu]}" | tr '/:[]' '____')"
    result_dir="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENTS[$gpu]}_${config_safe}_${NUM_HOURS}h${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}/${TASK}_${RESULT_PREFIX_SAFE}_${CELL_IDS[$gpu]}"
    [ ! -e "${result_dir}" ] && [ ! -L "${result_dir}" ] || {
        echo "FATAL: refusing to overwrite existing study result: ${result_dir}" >&2
        exit 2
    }
    RESULT_DIRS+=("${result_dir}")
done
for result_dir in "${RESULT_DIRS[@]}"; do
    mkdir -p "$(dirname "${result_dir}")"
    mkdir "${result_dir}" || {
        echo "FATAL: could not reserve unique study result: ${result_dir}" >&2
        exit 2
    }
done

LOGDIR="${PTB_LOG_DIR:-${REPO}/run_logs}"; mkdir -p "$LOGDIR"
declare -a PIDS=()
for (( gpu=0; gpu<N; gpu++ )); do
    spec="${SPECS[$gpu]}"
    condition="${CONDITIONS[$gpu]}"
    agent="${AGENTS[$gpu]}"
    cfg="${CONFIGS[$gpu]}"
    prompt="${PROMPTS[$gpu]}"
    repetition="${REPETITIONS[$gpu]}"
    binds=()
    if [ -n "${VERTEX_ADC_FILE}" ]; then
        binds+=("${VERTEX_ADC_FILE}:/home/ben/.config/gcloud/application_default_credentials.json:ro")
    fi
    case "${condition}" in
        c1) binds+=("${PRIOR_RUNS}:/home/ben/prior_runs:ro");;
        c2)
            binds+=("${PRIOR_RUNS}:/home/ben/prior_runs:ro")
            ;;
        c3) binds+=("${WM_MEMORY}:/home/ben/wm-memory:ro");;
    esac
    cell="${CELL_IDS[$gpu]}"
    safe="$(echo "$spec" | tr '/:,' '___')"
    log="${LOGDIR}/wm-${RUN_ID}-g${gpu}-${safe}.log"
    (
        sleep $(( gpu * STAGGER ))
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${GPU_SLOTS[$gpu]}"
        # PTB has used both names across the pinned runner revisions this
        # harness supports.  Forward both; each runner reads only its own.
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${GPU_SLOTS[$gpu]}"
        export POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES="${GPU_SLOTS[$gpu]}"
        export POST_TRAIN_BENCH_PROMPT="${prompt}"
        export POST_TRAIN_BENCH_EXTRA_BINDS="$(IFS=,; echo "${binds[*]:-}")"
        export AWM_STUDY_CONDITION="${condition}"
        export AWM_STUDY_REPETITION="${repetition}"
        export AWM_STUDY_MODE="${STUDY_MODE}"
        export AWM_STUDY_NUM_HOURS="${NUM_HOURS}"
        export AWM_REPO_COMMIT
        export AWM_PTB_COMMIT
        export AWM_PTB_SURFACE_MANIFEST_SHA256
        export AWM_WMA_MODEL
        export AWM_PRIOR_CORPUS_MANIFEST_SHA256
        export AWM_CARD_CORPUS_MANIFEST_SHA256
        export AWM_EXPECTED_SCIENTIST_MODEL_ID="${EXPECTED_SCIENTIST_MODELS[$gpu]}"
        export AWM_CLAUDE_CLI_VERSION
        export AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT
        if [ -n "${VERTEX_ADC_FILE}" ]; then
            export GOOGLE_APPLICATION_CREDENTIALS=/home/ben/.config/gcloud/application_default_credentials.json
        else
            unset GOOGLE_APPLICATION_CREDENTIALS
        fi
        echo "cell start gpu=${gpu} repetition=${repetition} condition=${condition} agent=${agent} cfg=${cfg} prompt=${prompt} binds=${POST_TRAIN_BENCH_EXTRA_BINDS:-<none>} at $(date --iso-8601=seconds)"
        cd "${PINNED_REPO}" || exit 2
        bash src/run_task.sh "${TASK}" "${agent}" "${MODEL_TO_TRAIN}" "${cell}" "${NUM_HOURS}" "${cfg}" 1
    ) > "${log}" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${gpu} repetition=${repetition} spec=${spec} prompt=${prompt} cell=${cell} pid=${PIDS[$gpu]} log=${log}"
done

rc_all=0
for (( gpu=0; gpu<N; gpu++ )); do
    if wait "${PIDS[$gpu]}"; then rc=0; else rc=$?; fi
    echo "cell_exit gpu=${gpu} repetition=${REPETITIONS[$gpu]} spec=${SPECS[$gpu]} cell=${CELL_IDS[$gpu]} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done
echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
