#!/bin/bash
# Build a PRIVATE PostTrainBench checkout for the crossed rollout.
#
# Not the shared one. /rmeng_data/robtang/PostTrainBench is another session's
# live working directory and a commit there kills every job already running out
# of it (bash reads a script incrementally, so the ESTALE lands hours later when
# the long command returns). This makes an isolated clone pinned to an explicit
# SHA, with its own results dir, and installs the hv_recipe agent into it.
#
# Containers and the HF cache are shared read-only -- they are large, immutable,
# and nothing here writes to them.
#
#   bash rollout/setup.sh
set -euo pipefail

SRC=/rmeng_data/robtang/PostTrainBench
DST=${HV_PTB_DIR:-/rmeng_data/robtang/ptb-hvrecipe}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PIN used to default to the SOURCE checkout's current HEAD, which quietly re-pins
# this one on every re-run -- and setup.sh is the natural way to reinstall an agent,
# so the reinstall moved the apparatus. SRC is another session's live working repo
# (three commits on 2026-08-31 alone; on 2026-08-30 three separate commits each
# rewrote src/run_task.sh), so "the SHA I audited" and "the SHA I ran" would drift
# apart with nothing in the log saying so.
#
# Now: a fresh clone takes SRC's HEAD, an existing checkout keeps the SHA it is on,
# and moving an existing checkout takes an explicit HV_PTB_SHA. Reinstalling an
# agent no longer touches the pin.
#
# Never run this while cells are live either way: the checkout below rewrites
# tracked-file inodes, and bash reads a running script incrementally.
if [ ! -d "$DST/.git" ]; then
    echo "cloning $SRC -> $DST"
    git clone --quiet "$SRC" "$DST"
    PIN="${HV_PTB_SHA:-$(git -C "$SRC" rev-parse HEAD)}"
else
    PIN="${HV_PTB_SHA:-$(git -C "$DST" rev-parse HEAD)}"
fi
CURRENT="$(git -C "$DST" rev-parse HEAD)"
if [ "$PIN" != "$CURRENT" ]; then
    echo "re-pinning $CURRENT -> $PIN"
    git -C "$DST" fetch --quiet origin
    git -C "$DST" checkout --quiet --detach "$PIN"
fi
echo "pinned to $(git -C "$DST" rev-parse HEAD)"

for a in hv_recipe hv_noop; do
    install -d "$DST/agents/$a"
    install -m 0755 "$HERE/agents/$a/solve.sh"     "$DST/agents/$a/solve.sh"
    install -m 0644 "$HERE/agents/$a/api_keys.json" "$DST/agents/$a/api_keys.json"
done
mkdir -p "$DST/slurm_logs"

# `git clone` carries tracked files only, and the held-out test sets are ignored
# by design (.gitignore: `**/test_data.json`) so an agent cannot reach them
# through the repo. run_task.sh line 195 nevertheless hard-requires the one for
# the task being run -- it hands the agent the same n-gram checker and test-set
# copy the contamination judge gets. A clone therefore produces a checkout that
# looks complete and dies four minutes into every cell:
#
#   ERROR: src/eval/tasks/gsm8k/test_data.json not found -- required for the
#          agent's decontamination tooling
#
# That is how jobs 84279 and 84280 lost all 14 cells on 2026-08-31. Copy them
# from the source checkout rather than re-downloading, so this checkout scores
# against byte-identical test data to every other run in the corpus -- a
# re-download that differed by one item would make the numbers incomparable
# without saying so.
copied=0
for td in "$SRC"/src/eval/tasks/*/test_data.json; do
    [ -e "$td" ] || continue
    task="$(basename "$(dirname "$td")")"
    install -m 0600 "$td" "$DST/src/eval/tasks/$task/test_data.json"
    copied=$((copied + 1))
done
[ "$copied" -gt 0 ] || { echo "FATAL: no test_data.json under $SRC/src/eval/tasks -- run src/judges/test_data_download/download_test_data.py there first" >&2; exit 1; }
echo "test data   : $copied task(s) copied from $SRC"

# Fail here rather than four minutes into a GPU cell. The gate below is the same
# condition run_task.sh checks, evaluated for the task this rollout actually runs.
TASK_CHECK="${PTB_TASK:-gsm8k}"
[ -f "$DST/src/eval/tasks/$TASK_CHECK/test_data.json" ] \
    || { echo "FATAL: $DST/src/eval/tasks/$TASK_CHECK/test_data.json missing after copy" >&2; exit 1; }

# Own results dir; everything else copied from the shared checkout's .env so the
# container name and caches match what the corpus runs used.
RESULTS=/rmeng_data/robtang/ptb-hvrecipe-results
mkdir -p "$RESULTS"
sed -e "s#^POST_TRAIN_BENCH_RESULTS_DIR=.*#POST_TRAIN_BENCH_RESULTS_DIR=\"$RESULTS\"#" \
    -e 's#^POST_TRAIN_BENCH_EXPERIMENT_NAME=.*##' \
    "$SRC/.env" > "$DST/.env"
chmod 600 "$DST/.env"

echo "results dir : $RESULTS"
echo "checkout    : $DST"
grep -c . "$DST/.env" >/dev/null
echo "agent installed:"; ls -la "$DST/agents/hv_recipe"
