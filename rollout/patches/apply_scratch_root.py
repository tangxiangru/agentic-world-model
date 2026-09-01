#!/usr/bin/env python3
"""Put each PostTrainBench cell under an explicit, capacity-checked scratch root.

Upstream creates the complete writable home under ``/tmp`` and later tries to
remove a different, non-unique path.  A Gemma checkpoint can therefore fill a
compute node's root filesystem and leave the unique directory behind.  The
study launcher already provides ``POST_TRAIN_BENCH_TMP_ROOT``; this patch makes
the pinned private runner use it, fail closed on ownership/mode/headroom, and
remove only the exact directory returned by ``mktemp -d``.

The patch is site-neutral.  The actual scratch path remains an untracked local
launcher setting.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: explicit owned scratch root (rollout/patches/apply_scratch_root.py) ---"
DEFAULT_MIN_FREE_BYTES = 96 * 1024 * 1024 * 1024
DEFAULT_MIN_FREE_INODES = 100_000

OLD_SETUP = (
    'export TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"\n'
    "\n"
    'JOB_DIR="${TMP_SUBDIR}/job_dir"\n'
)

SETUP_BLOCK = f'''{MARK}
: "${{POST_TRAIN_BENCH_TMP_ROOT:?set POST_TRAIN_BENCH_TMP_ROOT to a private scratch directory}}"
case "${{POST_TRAIN_BENCH_TMP_ROOT}}" in
    /*) ;;
    *) echo "ERROR: POST_TRAIN_BENCH_TMP_ROOT must be absolute" >&2; exit 2 ;;
esac
[ -d "${{POST_TRAIN_BENCH_TMP_ROOT}}" ] && [ ! -L "${{POST_TRAIN_BENCH_TMP_ROOT}}" ] || {{
    echo "ERROR: POST_TRAIN_BENCH_TMP_ROOT must be a real directory" >&2
    exit 2
}}
AWM_SCRATCH_REAL="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${{POST_TRAIN_BENCH_TMP_ROOT}}")" || exit 2
[ "${{AWM_SCRATCH_REAL}}" = "${{POST_TRAIN_BENCH_TMP_ROOT}}" ] || {{
    echo "ERROR: POST_TRAIN_BENCH_TMP_ROOT must not traverse symlinks or aliases" >&2
    exit 2
}}
readonly AWM_SCRATCH_ROOT="${{POST_TRAIN_BENCH_TMP_ROOT}}"
[ "$(stat -c '%u' "${{POST_TRAIN_BENCH_TMP_ROOT}}")" = "$(id -u)" ] && \
    [ "$(stat -c '%a' "${{POST_TRAIN_BENCH_TMP_ROOT}}")" = 700 ] || {{
    echo "ERROR: POST_TRAIN_BENCH_TMP_ROOT must be owned by this uid and mode 0700" >&2
    exit 2
}}
AWM_MIN_SCRATCH_FREE_BYTES="${{POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES:-{DEFAULT_MIN_FREE_BYTES}}}"
AWM_MIN_SCRATCH_FREE_INODES="${{POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES:-{DEFAULT_MIN_FREE_INODES}}}"
[[ "${{AWM_MIN_SCRATCH_FREE_BYTES}}" =~ ^[1-9][0-9]*$ ]] && \
    [[ "${{AWM_MIN_SCRATCH_FREE_INODES}}" =~ ^[1-9][0-9]*$ ]] || {{
    echo "ERROR: scratch headroom thresholds must be positive decimal integers" >&2
    exit 2
}}
AWM_SCRATCH_FREE_KIB="$(df -Pk "${{POST_TRAIN_BENCH_TMP_ROOT}}" | awk 'NR == 2 {{print $4}}')"
AWM_SCRATCH_FREE_INODES="$(df -Pi "${{POST_TRAIN_BENCH_TMP_ROOT}}" | awk 'NR == 2 {{print $4}}')"
[[ "${{AWM_SCRATCH_FREE_KIB}}" =~ ^[0-9]+$ ]] && \
    [[ "${{AWM_SCRATCH_FREE_INODES}}" =~ ^[0-9]+$ ]] || {{
    echo "ERROR: could not measure scratch free blocks/inodes" >&2
    exit 2
}}
(( AWM_SCRATCH_FREE_KIB * 1024 >= AWM_MIN_SCRATCH_FREE_BYTES )) && \
    (( AWM_SCRATCH_FREE_INODES >= AWM_MIN_SCRATCH_FREE_INODES )) || {{
    echo "ERROR: scratch root lacks required free blocks/inodes" >&2
    exit 2
}}
AWM_OWNED_SCRATCH=""
awm_cleanup_owned_scratch() {{
    local scratch="${{AWM_OWNED_SCRATCH:-}}"
    [ -n "${{scratch}}" ] || return 0
    [ "${{scratch%/*}}" = "${{AWM_SCRATCH_ROOT}}" ] || {{
        echo "ERROR: refusing non-child scratch cleanup target" >&2
        return 2
    }}
    case "${{scratch##*/}}" in
        posttrain_container_*) ;;
        *) echo "ERROR: refusing ambiguous scratch cleanup target" >&2; return 2 ;;
    esac
    [ -d "${{scratch}}" ] && [ ! -L "${{scratch}}" ] && \
        [ "$(stat -c '%u' "${{scratch}}")" = "$(id -u)" ] || {{
        echo "ERROR: refusing changed scratch cleanup target" >&2
        return 2
    }}
    rm -rf -- "${{scratch}}" || {{
        echo "ERROR: could not remove owned cell scratch directory" >&2
        return 2
    }}
    [ ! -e "${{scratch}}" ] && [ ! -L "${{scratch}}" ] || {{
        echo "ERROR: owned cell scratch directory remains after cleanup" >&2
        return 2
    }}
    AWM_OWNED_SCRATCH=""
}}
awm_exit_with_scratch_cleanup() {{
    local original_status=$?
    local cleanup_status=0
    trap - EXIT
    if awm_cleanup_owned_scratch; then
        cleanup_status=0
    else
        cleanup_status=$?
    fi
    if [ "${{original_status}}" -ne 0 ]; then
        exit "${{original_status}}"
    fi
    exit "${{cleanup_status}}"
}}
trap awm_exit_with_scratch_cleanup EXIT

# Keep every path component launcher-owned. Task/model strings are metadata,
# not path material; mktemp's suffix alone supplies collision resistance.
TMP_SUBDIR="$(mktemp -d "${{AWM_SCRATCH_ROOT}}/posttrain_container_XXXXXXXX")" || exit 2
AWM_OWNED_SCRATCH="${{TMP_SUBDIR}}"
export TMP_SUBDIR
[ -d "${{TMP_SUBDIR}}" ] && [ ! -L "${{TMP_SUBDIR}}" ] && \
    [ "${{TMP_SUBDIR%/*}}" = "${{AWM_SCRATCH_ROOT}}" ] && \
    [ "$(stat -c '%u' "${{TMP_SUBDIR}}")" = "$(id -u)" ] && \
    [ "$(stat -c '%a' "${{TMP_SUBDIR}}")" = 700 ] || {{
    echo "ERROR: mktemp did not create the expected private cell scratch directory" >&2
    exit 2
}}

JOB_DIR="${{TMP_SUBDIR}}/job_dir"
'''

OLD_BROAD_CLEANUP = "rm -rf /tmp/posttrain_container\n"
NEW_BROAD_CLEANUP = (
    "# The exact mktemp-created cell directory is removed by the EXIT trap.\n"
)


def apply(text: str) -> str:
    if MARK in text:
        required = (
            'POST_TRAIN_BENCH_TMP_ROOT:?set POST_TRAIN_BENCH_TMP_ROOT',
            "AWM_MIN_SCRATCH_FREE_BYTES",
            "mktemp -d",
            "trap awm_exit_with_scratch_cleanup EXIT",
            "could not remove owned cell scratch directory",
        )
        if any(fragment not in text for fragment in required) or OLD_BROAD_CLEANUP in text:
            raise SystemExit("run_task.sh: explicit scratch patch is incomplete")
        return text
    if text.count(OLD_SETUP) != 1:
        raise SystemExit(
            "run_task.sh: expected exactly one upstream /tmp scratch setup; "
            "the runner changed shape — update apply_scratch_root.py"
        )
    if text.count(OLD_BROAD_CLEANUP) != 1:
        raise SystemExit(
            "run_task.sh: expected exactly one unsafe broad scratch cleanup; "
            "the runner changed shape — update apply_scratch_root.py"
        )
    text = text.replace(OLD_SETUP, SETUP_BLOCK, 1)
    return text.replace(OLD_BROAD_CLEANUP, NEW_BROAD_CLEANUP, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    original = path.read_text()
    patched = apply(original)
    if patched == original:
        print(f"{path}: already patched")
        return 0
    path.write_text(patched)
    print(f"{path}: patched (explicit owned scratch root)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
