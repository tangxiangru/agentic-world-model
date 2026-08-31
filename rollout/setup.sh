#!/bin/bash
# Build a PRIVATE PostTrainBench checkout for the rollout studies.
#
# Do not modify a checkout that another session may be running from: bash reads
# scripts incrementally, so changing one can break a long-running job hours
# later. This makes an isolated clone pinned to an explicit SHA, gives it its
# own results directory, and installs the study agents into it.
#
# Containers and the HF cache are shared read-only -- they are large, immutable,
# and nothing here writes to them.
#
# Local paths are deliberately required inputs rather than tracked defaults:
#
#   PTB_SOURCE_DIR=/path/to/PostTrainBench \
#   HV_PTB_DIR=/path/to/private-checkout \
#   PTB_RESULTS_DIR=/path/to/private-results \
#   AWM_REPO_COMMIT=<exact-40-hex-local-commit> \
#     bash rollout/setup.sh
set -euo pipefail

: "${PTB_SOURCE_DIR:?set PTB_SOURCE_DIR to the source PostTrainBench checkout}"
: "${HV_PTB_DIR:?set HV_PTB_DIR to the private checkout to create/update}"
: "${PTB_RESULTS_DIR:?set PTB_RESULTS_DIR to the private results directory}"
: "${AWM_REPO_COMMIT:?set AWM_REPO_COMMIT to the exact runtime commit to package}"
SRC="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PTB_SOURCE_DIR}")"
DST="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${HV_PTB_DIR}")"
RESULTS="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PTB_RESULTS_DIR}")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWM_SOURCE_DIR="${AWM_SOURCE_DIR:-$(cd "${HERE}/.." && pwd)}"

[ "${SRC}" != "${DST}" ] || {
    echo "FATAL: HV_PTB_DIR must be a private checkout, not PTB_SOURCE_DIR" >&2
    exit 2
}
case "${DST}/" in
    "${SRC}/"*) echo "FATAL: HV_PTB_DIR must not be nested inside PTB_SOURCE_DIR" >&2; exit 2 ;;
esac
case "${SRC}/" in
    "${DST}/"*) echo "FATAL: HV_PTB_DIR must not contain PTB_SOURCE_DIR" >&2; exit 2 ;;
esac

[[ "${AWM_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "FATAL: AWM_REPO_COMMIT must be a full 40-hex commit" >&2
    exit 2
}
git -C "${AWM_SOURCE_DIR}" cat-file -e "${AWM_REPO_COMMIT}^{commit}" 2>/dev/null || {
    echo "FATAL: AWM_REPO_COMMIT is not present in AWM_SOURCE_DIR: ${AWM_SOURCE_DIR}" >&2
    exit 2
}

# Fail if any live file which this setup/launcher consumes differs from the
# exact harness commit recorded in every cell. Runtime code itself is extracted
# by `git archive` below; these files are the bootstrap/setup surface which is
# necessarily read from the invoking worktree.
AWM_LIVE_ROOT="$(cd "${HERE}/.." && pwd)"
STUDY_BOOTSTRAP_FILES=(
    input/instruction.md
    rollout/setup.sh
    rollout/wm_pack.sbatch
    rollout/study_matrix.py
    rollout/build_prompts.py
    rollout/pin_ptb_source.sh
    rollout/attest_ptb_surface.py
    rollout/patches/apply_study_runner.py
    rollout/patches/apply_extra_binds.py
)
for a in hv_recipe hv_noop claude_fulltraj_noawm claude_wm; do
    STUDY_BOOTSTRAP_FILES+=(
        "rollout/agents/${a}/solve.sh"
        "rollout/agents/${a}/api_keys.json"
    )
    [ ! -e "${AWM_LIVE_ROOT}/rollout/agents/${a}/env_passthrough.txt" ] || \
        STUDY_BOOTSTRAP_FILES+=("rollout/agents/${a}/env_passthrough.txt")
done
for rel in "${STUDY_BOOTSTRAP_FILES[@]}"; do
    live="${AWM_LIVE_ROOT}/${rel}"
    [ -f "${live}" ] && [ ! -L "${live}" ] && \
        git -C "${AWM_SOURCE_DIR}" cat-file -e "${AWM_REPO_COMMIT}:${rel}" 2>/dev/null && \
        git -C "${AWM_SOURCE_DIR}" show "${AWM_REPO_COMMIT}:${rel}" | cmp -s - "${live}" || {
        echo "FATAL: live study bootstrap file does not byte-match AWM_REPO_COMMIT: ${rel}" >&2
        exit 2
    }
done

git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "FATAL: PTB_SOURCE_DIR is not a Git checkout: $SRC" >&2
    exit 2
}
[ -f "$SRC/.env" ] || { echo "FATAL: PTB_SOURCE_DIR has no .env: $SRC" >&2; exit 2; }
grep -q '^POST_TRAIN_BENCH_RESULTS_DIR=' "$SRC/.env" || {
    echo "FATAL: source .env has no POST_TRAIN_BENCH_RESULTS_DIR setting" >&2
    exit 2
}
if [ -e "$DST" ]; then
    git -C "$DST" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        echo "FATAL: HV_PTB_DIR exists but is not a Git checkout: $DST" >&2
        exit 2
    }
    SENTINEL="$DST/.git/awm-study-source"
    if [ -f "$SENTINEL" ]; then
        [ "$(cat "$SENTINEL")" = "$SRC" ] || {
            echo "FATAL: existing HV_PTB_DIR belongs to a different source checkout" >&2
            exit 2
        }
    else
        ORIGIN="$(git -C "$DST" remote get-url origin 2>/dev/null || true)"
        ORIGIN_RESOLVED="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$ORIGIN")"
        [ -n "$ORIGIN" ] && [ "$ORIGIN_RESOLVED" = "$SRC" ] || {
            echo "FATAL: existing HV_PTB_DIR has no verifiable private-clone origin" >&2
            exit 2
        }
        printf '%s\n' "$SRC" > "$SENTINEL"
    fi
else
    echo "cloning $SRC -> $DST"
    git clone --quiet "$SRC" "$DST"
    printf '%s\n' "$SRC" > "$DST/.git/awm-study-source"
fi
PIN="${HV_PTB_SHA:-$(git -C "$SRC" rev-parse HEAD)}"
[[ "$PIN" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "FATAL: HV_PTB_SHA must be a full 40-hex commit, not a branch/tag/abbreviation" >&2
    exit 2
}
PIN_COMMIT="$(git -C "$SRC" rev-parse --verify "${PIN}^{commit}" 2>/dev/null)" || {
    echo "FATAL: HV_PTB_SHA is not a commit in PTB_SOURCE_DIR: ${PIN}" >&2
    exit 2
}
git -C "$DST" fetch --quiet origin
git -C "$DST" checkout --quiet --detach "$PIN_COMMIT"
rm -f "$DST/.git/awm-study-harness-commit"
printf '%s\n' "${PIN_COMMIT,,}" > "$DST/.git/awm-study-ptb-commit"
echo "pinned to ${PIN_COMMIT,,}"

# PTB generates benchmark test copies locally and intentionally ignores them in
# Git.  The GSM8K runner requires this file for the read-only contamination
# checker exposed to the scientist, so carry the site's generated copy into the
# private checkout explicitly.  It becomes part of the attested study surface
# below; neither the bytes nor their site path enter this repository.
STUDY_TEST_DATA_REL="src/eval/tasks/gsm8k/test_data.json"
STUDY_TEST_DATA_SRC="$SRC/$STUDY_TEST_DATA_REL"
[ -f "$STUDY_TEST_DATA_SRC" ] && [ ! -L "$STUDY_TEST_DATA_SRC" ] || {
    echo "FATAL: PTB_SOURCE_DIR lacks regular $STUDY_TEST_DATA_REL; run src/judges/test_data_download/download_test_data.py --tasks gsm8k there first" >&2
    exit 2
}
install -D -m 0600 "$STUDY_TEST_DATA_SRC" "$DST/$STUDY_TEST_DATA_REL"

# Add the study contract to the private pinned checkout before probing it.  The
# patchers are idempotent and fail if the upstream runner no longer has the
# exact expected shape; neither one touches PTB_SOURCE_DIR.
python3 "$HERE/patches/apply_study_runner.py" "$DST/src/run_task.sh"
python3 "$HERE/patches/apply_extra_binds.py" "$DST/src/run_task.sh"
install -m 0755 "$HERE/pin_ptb_source.sh" "$DST/src/commit_utils/pin_src_locally.sh"
install -m 0755 "$HERE/attest_ptb_surface.py" "$DST/src/commit_utils/attest_study_surface.py"

# Vertex routing is intentionally passed by variable name through PTB's clean
# Apptainer environment.  Refuse an older runner that would silently discard
# those variables and fail only after a GPU allocation starts.
grep -q 'env_passthrough.txt' "$DST/src/run_task.sh" || {
    echo "FATAL: pinned PTB runner lacks agents/<agent>/env_passthrough.txt support" >&2
    exit 2
}
grep -q 'POST_TRAIN_BENCH_PROMPT' "$DST/src/eval/general/get_prompt.py" || {
    echo "FATAL: pinned PTB prompt loader cannot select the study prompts" >&2
    exit 2
}
grep -q 'POST_TRAIN_BENCH_JUDGE_AUTH_MODE' "$DST/src/run_task.sh" || {
    echo "FATAL: pinned PTB runner cannot disable subscription-only judges" >&2
    exit 2
}
[ -x "$DST/src/commit_utils/pin_src_locally.sh" ] || {
    echo "FATAL: private PTB checkout lacks the portable source pin helper" >&2
    exit 2
}
for capability in 'agents/${AGENT}/payload' instruction.sha256 PROMPT_ENV_ARGS PROMPT_BIND_ARGS 'prompt generation failed' SOLVE_RC solve_exit_code.txt POST_TRAIN_BENCH_VISIBLE_GPUS POST_TRAIN_BENCH_ISOLATE_GPUS POST_TRAIN_BENCH_EVAL_GPU_REAP 'OS-visible GPU isolation probe' POST_TRAIN_BENCH_EXTRA_BINDS; do
    grep -Fq "$capability" "$DST/src/run_task.sh" || {
        echo "FATAL: pinned PTB runner lacks required study capability: $capability" >&2
        exit 2
    }
done

for a in hv_recipe hv_noop claude_fulltraj_noawm claude_wm; do
    AGENT_DST="$DST/agents/$a"
    case "$AGENT_DST" in "$DST/agents/"*) ;; *) echo "FATAL: invalid study agent path" >&2; exit 2;; esac
    if [ -e "$AGENT_DST" ] || [ -L "$AGENT_DST" ]; then
        rm -rf "$AGENT_DST"
    fi
    install -d "$AGENT_DST"
    install -m 0755 "$HERE/agents/$a/solve.sh"     "$DST/agents/$a/solve.sh"
    install -m 0644 "$HERE/agents/$a/api_keys.json" "$DST/agents/$a/api_keys.json"
    if [ -f "$HERE/agents/$a/env_passthrough.txt" ]; then
        install -m 0644 "$HERE/agents/$a/env_passthrough.txt" "$DST/agents/$a/env_passthrough.txt"
    fi
done

# Package only the runtime surface into the WMA agent payload.  In particular,
# do not copy results/ or the historical-card corpus into C1/C2: the exact
# commit remains auditable while the scientist cannot browse unrelated repo
# content.  PTB copies payload/ to /home/ben/agent inside the sandbox.
PAYLOAD_ROOT="$DST/agents/claude_wm/payload"
PAYLOAD_STAGE="$(mktemp -d "$DST/agents/claude_wm/.payload-stage.XXXXXX")"
mkdir -p "$PAYLOAD_STAGE/awm-src"
git -C "$AWM_SOURCE_DIR" archive --format=tar "$AWM_REPO_COMMIT" \
    awm input .claude rollout/validate_study_corpus.py rollout/attest_claude_runtime.py rollout/validate_wma_session.py \
    | tar -x -C "$PAYLOAD_STAGE/awm-src"
printf '%s\n' "${AWM_REPO_COMMIT,,}" > "$PAYLOAD_STAGE/awm-src/AWM_COMMIT"
mv "$PAYLOAD_STAGE/awm-src/rollout/validate_study_corpus.py" \
    "$PAYLOAD_STAGE/validate_study_corpus.py"
mv "$PAYLOAD_STAGE/awm-src/rollout/attest_claude_runtime.py" \
    "$PAYLOAD_STAGE/attest_claude_runtime.py"
mv "$PAYLOAD_STAGE/awm-src/rollout/validate_wma_session.py" \
    "$PAYLOAD_STAGE/validate_wma_session.py"
rmdir "$PAYLOAD_STAGE/awm-src/rollout"
chmod 0755 "$PAYLOAD_STAGE/validate_study_corpus.py"
chmod 0755 "$PAYLOAD_STAGE/attest_claude_runtime.py"
chmod 0755 "$PAYLOAD_STAGE/validate_wma_session.py"

# C1 has no AWM runtime but receives the same commit-pinned standalone
# validator at /home/ben/agent/validate_study_corpus.py.
C1_PAYLOAD="$DST/agents/claude_fulltraj_noawm/payload"
C1_PAYLOAD_STAGE="$(mktemp -d "$DST/agents/claude_fulltraj_noawm/.payload-stage.XXXXXX")"
install -m 0755 "$PAYLOAD_STAGE/validate_study_corpus.py" \
    "$C1_PAYLOAD_STAGE/validate_study_corpus.py"
install -m 0755 "$PAYLOAD_STAGE/attest_claude_runtime.py" \
    "$C1_PAYLOAD_STAGE/attest_claude_runtime.py"
rm -rf "$C1_PAYLOAD"
mv "$C1_PAYLOAD_STAGE" "$C1_PAYLOAD"

rm -rf "$PAYLOAD_ROOT"
mv "$PAYLOAD_STAGE" "$PAYLOAD_ROOT"
echo "awm payload : ${AWM_REPO_COMMIT,,} (runtime + corpus/model/CLI attesters)"

# Authentication and the immutable awm commit are forwarded by variable name at
# launch time.  The values remain in the submitting environment, outside Git.
for a in claude_fulltraj_noawm claude_wm; do
    rm -f "$DST/agents/$a/oauth_token"
    echo "Vertex environment allowlist installed for $a"
done

# The study's prompt files.
python3 "$HERE/build_prompts.py" --no-review "$DST"

# Own results dir; everything else copied from the shared checkout's .env so the
# container name and caches match what the corpus runs used.
mkdir -p "$RESULTS"
sed -e "s#^POST_TRAIN_BENCH_RESULTS_DIR=.*#POST_TRAIN_BENCH_RESULTS_DIR=\"$RESULTS\"#" \
    -e 's#^POST_TRAIN_BENCH_EXPERIMENT_NAME=.*##' \
    "$SRC/.env" > "$DST/.env"
chmod 600 "$DST/.env"

# These are site/runtime choices and intentionally remain in the untracked PTB
# .env. Validate the private copy now rather than consuming a GPU only to have
# wm_pack reject it. Use the same shell assignment semantics PTB itself uses,
# without printing any other values from the credential-bearing file.
(
    unset POST_TRAIN_BENCH_ISOLATE_GPUS POST_TRAIN_BENCH_EVAL_GPU_REAP
    # shellcheck disable=SC1090  # the path is the private checkout chosen above
    source "$DST/.env"
    [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-}" = 1 ] || {
        echo "FATAL: private PTB .env must set POST_TRAIN_BENCH_ISOLATE_GPUS=1" >&2
        exit 2
    }
    case "${POST_TRAIN_BENCH_EVAL_GPU_REAP:-}" in
        own|none) ;;
        *) echo "FATAL: private PTB .env must set POST_TRAIN_BENCH_EVAL_GPU_REAP=own or none" >&2; exit 2 ;;
    esac
)

SURFACE_MANIFEST="$DST/.git/awm-study-surface.json"
python3 "$DST/src/commit_utils/attest_study_surface.py" write \
    --root "$DST" --manifest "$SURFACE_MANIFEST" \
    --awm-commit "${AWM_REPO_COMMIT,,}" --ptb-commit "${PIN_COMMIT,,}" >/dev/null

echo "results dir : $RESULTS"
echo "checkout    : $DST"
grep -c . "$DST/.env" >/dev/null
echo "agents installed:"
ls "$DST/agents" | grep -E "hv_|claude_(fulltraj|wm)"
printf '%s\n' "${AWM_REPO_COMMIT,,}" > "$DST/.git/awm-study-harness-commit"
