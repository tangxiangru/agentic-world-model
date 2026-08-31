#!/usr/bin/env python3
"""Add the portable study-runner contract to PostTrainBench.

The study intentionally keeps its PostTrainBench checkout private and pinned.
The currently pinned upstream runner (``11c41c9``) predates several facilities
needed by the rollout harness, so ``rollout/setup.sh`` applies this source
transformation to that private checkout.  It never edits the shared/source
checkout or the tracked submodule.

The transformation is deliberately narrow and fail-closed: every upstream
anchor must occur exactly once, a partially applied patch is rejected, and a
second complete application is a no-op.  It adds:

* an identifier-only ``agents/<agent>/env_passthrough.txt`` allowlist whose
  values are forwarded as individual argv entries through ``--cleanenv``;
* ``agents/<agent>/payload/`` copied to ``/home/ben/agent``;
* explicit, validated per-cell GPU visibility/isolation;
* preservation and result-side recording of the real agent ``SOLVE_RC``; and
* evaluation cleanup modes ``none`` and ``own``.  ``own`` can only signal a
  same-UID process on the declared device which inherited this cell's random
  token.  There is no cluster-wide ``nvidia-smi | kill`` path.

Usage::

    python rollout/patches/apply_study_runner.py <ptb>/src/run_task.sh
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: portable study runner (rollout/patches/apply_study_runner.py) ---"

COMPLETE_NEEDLES = (
    "env_passthrough.txt",
    'agents/${AGENT}/payload',
    "SOLVE_RC",
    "solve_exit_code.txt",
    "POST_TRAIN_BENCH_VISIBLE_GPUS",
    "POST_TRAIN_BENCH_ISOLATE_GPUS",
    "POST_TRAIN_BENCH_EVAL_GPU_REAP",
    "POST_TRAIN_BENCH_CELL_TOKEN",
    "OS-visible GPU isolation probe",
    "skipping optional judges and evaluating the preserved final_model",
    "deterministic evaluation produced no valid metrics.json",
    "math.isfinite(float(accuracy))",
)


ENV_ANCHOR = (
    'echo "API keys provisioned for agent=${AGENT} task=${EVALUATION_TASK}: '
    '${ALLOWED_API_KEYS[*]:-<none>}"\n'
)

ENV_BLOCK = f'''\

{MARK}
# An agent may request ordinary (non-secret or ambient-auth) variables by name.
# Parse names without eval and pass values as single argv entries so whitespace,
# quotes, glob characters, and leading dashes remain data rather than shell code.
AGENT_ENV_ARGS=()
AGENT_ENV_NAMES=()
declare -A AGENT_ENV_SEEN=()
AGENT_ENV_FILE="agents/${{AGENT}}/env_passthrough.txt"
if [ -L "$AGENT_ENV_FILE" ] || {{ [ -e "$AGENT_ENV_FILE" ] && [ ! -f "$AGENT_ENV_FILE" ]; }}; then
    echo "ERROR: $AGENT_ENV_FILE exists but is linked or not a regular file" >&2
    exit 2
fi
if [ -f "$AGENT_ENV_FILE" ]; then
    while IFS= read -r _env_name || [ -n "$_env_name" ]; do
        _env_name="${{_env_name%$'\\r'}}"
        [ -z "$_env_name" ] && continue
        [[ "$_env_name" =~ ^[[:space:]]*# ]] && continue
        if [[ ! "$_env_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            echo "ERROR: invalid variable name in $AGENT_ENV_FILE: $_env_name" >&2
            exit 2
        fi
        case "$_env_name" in
            CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|POST_TRAIN_BENCH_VISIBLE_GPUS|POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES|POST_TRAIN_BENCH_ISOLATE_GPUS|POST_TRAIN_BENCH_CELL_TOKEN|PATH|HOME|HF_HOME|NUM_GPUS|PROMPT|AGENT_CONFIG)
                echo "ERROR: reserved runner variable in $AGENT_ENV_FILE: $_env_name" >&2
                exit 2
                ;;
        esac
        [ -z "${{AGENT_ENV_SEEN[$_env_name]+x}}" ] || {{
            echo "ERROR: duplicate variable name in $AGENT_ENV_FILE: $_env_name" >&2
            exit 2
        }}
        AGENT_ENV_SEEN[$_env_name]=1
        if [[ -v "$_env_name" ]]; then
            AGENT_ENV_ARGS+=(--env "${{_env_name}}=${{!_env_name}}")
            AGENT_ENV_NAMES+=("$_env_name")
        fi
    done < "$AGENT_ENV_FILE"
fi
echo "Environment variables provisioned for agent=${{AGENT}}: ${{AGENT_ENV_NAMES[*]:-<none>}}"
'''


PAYLOAD_ANCHOR = 'cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"\n'

PAYLOAD_BLOCK = '''\
if [ -e "agents/${AGENT}/payload" ]; then
    if [ ! -d "agents/${AGENT}/payload" ] || [ -L "agents/${AGENT}/payload" ]; then
        echo "ERROR: agents/${AGENT}/payload must be a real directory" >&2
        exit 2
    fi
    mkdir -p "${JOB_DIR}/agent"
    cp -a "agents/${AGENT}/payload/." "${JOB_DIR}/agent/"
fi
'''


UUID_ANCHOR = "RANDOM_UUID=$(uuidgen)\n"
UUID_BLOCK = '''\
# A random per-cell marker is inherited by agent/evaluator children.  The safe
# evaluation reaper requires this exact marker before it can signal a process.
export POST_TRAIN_BENCH_CELL_TOKEN="awm-${RANDOM_UUID}"
'''


OLD_GPU_BLOCK = '''\
    GPU_PIN_ENV=()
    [ -n "${POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES:-}" ] && GPU_PIN_ENV+=(--env "CUDA_VISIBLE_DEVICES=${POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES}")
    [ -n "${POST_TRAIN_BENCH_GPU_NAME_MATCH+x}" ] && GPU_PIN_ENV+=(--env "POST_TRAIN_BENCH_GPU_NAME_MATCH=${POST_TRAIN_BENCH_GPU_NAME_MATCH}")
    [ -n "${POST_TRAIN_BENCH_ALLOW_BUSY_GPU:-}" ] && GPU_PIN_ENV+=(--env "POST_TRAIN_BENCH_ALLOW_BUSY_GPU=${POST_TRAIN_BENCH_ALLOW_BUSY_GPU}")
'''

NEW_GPU_BLOCK = '''\
    GPU_PIN_ENV=()
    case "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" in
        0|1) ;;
        *) echo "ERROR: POST_TRAIN_BENCH_ISOLATE_GPUS must be 0 or 1" >&2; return 2 ;;
    esac
    if [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ] && \
       [ -n "${POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES:-}" ] && \
       [ "${POST_TRAIN_BENCH_VISIBLE_GPUS}" != "${POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES}" ]; then
        echo "ERROR: POST_TRAIN_BENCH_VISIBLE_GPUS and POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES disagree" >&2
        return 2
    fi
    EFFECTIVE_VISIBLE_GPUS="${POST_TRAIN_BENCH_VISIBLE_GPUS:-${POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES:-}}"
    if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" = 1 ] && [ -z "$EFFECTIVE_VISIBLE_GPUS" ]; then
        echo "ERROR: GPU isolation requires POST_TRAIN_BENCH_VISIBLE_GPUS" >&2
        return 2
    fi
    if [ -n "$EFFECTIVE_VISIBLE_GPUS" ]; then
        IFS=',' read -r -a _visible_gpu_array <<< "$EFFECTIVE_VISIBLE_GPUS"
        for _visible_gpu in "${_visible_gpu_array[@]}"; do
            if [[ ! "$_visible_gpu" =~ ^[A-Za-z0-9_.:/-]+$ ]]; then
                echo "ERROR: invalid GPU identifier in POST_TRAIN_BENCH_VISIBLE_GPUS: $_visible_gpu" >&2
                return 2
            fi
        done
        if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" = 1 ] && \
           [ "${#_visible_gpu_array[@]}" -ne "$NUM_GPUS" ]; then
            echo "ERROR: isolated GPU list has ${#_visible_gpu_array[@]} entries, expected NUM_GPUS=${NUM_GPUS}" >&2
            return 2
        fi
        export POST_TRAIN_BENCH_VISIBLE_GPUS="$EFFECTIVE_VISIBLE_GPUS"
        export POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES="$EFFECTIVE_VISIBLE_GPUS"
        GPU_PIN_ENV+=(
            --env "CUDA_VISIBLE_DEVICES=${EFFECTIVE_VISIBLE_GPUS}"
            --env "NVIDIA_VISIBLE_DEVICES=${EFFECTIVE_VISIBLE_GPUS}"
            --env "POST_TRAIN_BENCH_VISIBLE_GPUS=${EFFECTIVE_VISIBLE_GPUS}"
            --env "POST_TRAIN_BENCH_ISOLATE_GPUS=${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}"
            --env "POST_TRAIN_BENCH_CELL_TOKEN=${POST_TRAIN_BENCH_CELL_TOKEN}"
        )
    fi
    [ -n "${POST_TRAIN_BENCH_GPU_NAME_MATCH+x}" ] && GPU_PIN_ENV+=(--env "POST_TRAIN_BENCH_GPU_NAME_MATCH=${POST_TRAIN_BENCH_GPU_NAME_MATCH}")
    [ -n "${POST_TRAIN_BENCH_ALLOW_BUSY_GPU:-}" ] && GPU_PIN_ENV+=(--env "POST_TRAIN_BENCH_ALLOW_BUSY_GPU=${POST_TRAIN_BENCH_ALLOW_BUSY_GPU}")
'''


API_ARG_ANCHOR = '        "${API_KEY_ENV_ARGS[@]}" \\\n'
AGENT_ARG_LINE = '        "${AGENT_ENV_ARGS[@]}" \\\n'

OLD_SOLVE_LINE = (
    '        bash -c "{ python /home/ben/check_cuda.py && python '
    '/home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & '
    'MONITOR_PID=\\$!; bash /home/ben/agent_solve.sh; kill \\$MONITOR_PID 2>/dev/null; } '
    '2>&1 | python /home/ben/timestamp_lines.py" > "${SOLVE_OUT}" 2>&1\n'
)

NEW_SOLVE_LINE_V2 = (
    '        bash -o pipefail -c "{ python /home/ben/check_cuda.py && python '
    '/home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & '
    'MONITOR_PID=\\$!; bash /home/ben/agent_solve.sh; SOLVE_RC=\\$?; '
    'kill \\$MONITOR_PID 2>/dev/null || true; wait \\$MONITOR_PID 2>/dev/null || true; '
    'exit \\$SOLVE_RC; } 2>&1 | python /home/ben/timestamp_lines.py" '
    '> "${SOLVE_OUT}" 2>&1\n'
)

NEW_SOLVE_LINE = (
    '        bash -o pipefail -c "{ echo OS-visible GPU isolation probe >&2; '
    'env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES '
    "python -c 'import sys,torch; count=torch.cuda.device_count(); "
    "print(f\"gpu_device_count={count}\"); "
    "sys.exit(0 if count == 1 else 86)' || exit \\$?; "
    'python /home/ben/check_cuda.py && python '
    '/home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & '
    'MONITOR_PID=\\$!; bash /home/ben/agent_solve.sh; SOLVE_RC=\\$?; '
    'kill \\$MONITOR_PID 2>/dev/null || true; wait \\$MONITOR_PID 2>/dev/null || true; '
    'exit \\$SOLVE_RC; } 2>&1 | python /home/ben/timestamp_lines.py" '
    '> "${SOLVE_OUT}" 2>&1\n'
)


SOLVE_EXIT_ANCHOR = "SOLVE_EXIT=$?\n"
SOLVE_EXIT_RECORD = '''\
printf '%s\\n' "$SOLVE_EXIT" > "${EVAL_DIR}/solve_exit_code.txt"
'''


OLD_REAP_BLOCK = '''\
    # hv-patches (SAFETY): upstream assumes an exclusively-allocated HTCondor
    # node and SIGKILLs *every* CUDA process on the box before each eval
    # attempt. On a shared workstation that murders other users' jobs. Opt out
    # with POST_TRAIN_BENCH_KILL_GPU_PROCS=0; default stays upstream behaviour.
    if [ "${POST_TRAIN_BENCH_KILL_GPU_PROCS:-1}" = "1" ]; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    else
        echo "POST_TRAIN_BENCH_KILL_GPU_PROCS=0: not killing other GPU processes" >&2
    fi
    sleep 5
'''


REAP_FUNCTION = '''\
# Reap only a process which is on this cell's declared device, belongs to this
# Unix uid, and inherited this cell's unguessable token.  Revalidate all three
# properties immediately before TERM and KILL to reduce PID-reuse races.
reap_cell_gpu_processes() {
    local _mode="${POST_TRAIN_BENCH_EVAL_GPU_REAP:-none}"
    case "$_mode" in
        none)
            echo "POST_TRAIN_BENCH_EVAL_GPU_REAP=none: no GPU processes signalled" >&2
            return 0
            ;;
        own) ;;
        *)
            echo "ERROR: POST_TRAIN_BENCH_EVAL_GPU_REAP must be own or none" >&2
            return 2
            ;;
    esac
    if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" != 1 ] || \
       [ -z "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ] || \
       [ -z "${POST_TRAIN_BENCH_CELL_TOKEN:-}" ]; then
        echo "ERROR: own GPU reaping requires isolation, a device list, and a cell token" >&2
        return 2
    fi

    local _uid _gpu _query _pid _round _any_alive _attribution_rc
    local -a _owned_pids=()
    local -A _seen_pids=()
    _uid="$(id -u)"
    IFS=',' read -r -a _reap_gpu_array <<< "$POST_TRAIN_BENCH_VISIBLE_GPUS"

    # Return 0 only while PID is still a same-uid, token-bearing compute
    # process on at least one declared device. Return 2 when device enumeration
    # itself fails so callers abort cleanup rather than making an unsafe guess.
    _cell_pid_is_attributable() {
        local _candidate="$1" _candidate_gpu _candidate_query
        [ -r "/proc/${_candidate}/environ" ] || return 1
        [ "$(stat -c %u "/proc/${_candidate}" 2>/dev/null)" = "$_uid" ] || return 1
        tr '\\0' '\\n' < "/proc/${_candidate}/environ" 2>/dev/null | \
            grep -Fxq "POST_TRAIN_BENCH_CELL_TOKEN=${POST_TRAIN_BENCH_CELL_TOKEN}" || return 1
        for _candidate_gpu in "${_reap_gpu_array[@]}"; do
            if ! _candidate_query="$(nvidia-smi --id="${_candidate_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
                return 2
            fi
            grep -Fxq "${_candidate}" <<< "$_candidate_query" && return 0
        done
        return 1
    }

    for _gpu in "${_reap_gpu_array[@]}"; do
        if ! _query="$(nvidia-smi --id="${_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
            echo "ERROR: cannot enumerate processes for declared GPU ${_gpu}" >&2
            return 2
        fi
        while IFS= read -r _pid; do
            [[ "$_pid" =~ ^[0-9]+$ ]] && [ "$_pid" -gt 1 ] || continue
            [ -z "${_seen_pids[$_pid]+x}" ] || continue
            _seen_pids[$_pid]=1
            if _cell_pid_is_attributable "$_pid"; then
                _owned_pids+=("$_pid")
            else
                _attribution_rc=$?
                [ "$_attribution_rc" -ne 2 ] || {
                    echo "ERROR: cannot revalidate declared GPU membership" >&2
                    return 2
                }
            fi
        done <<< "$_query"
    done

    for _pid in "${_owned_pids[@]}"; do
        if _cell_pid_is_attributable "$_pid"; then
            echo "reaping cell-owned GPU process pid=${_pid}" >&2
            kill -TERM "$_pid" 2>/dev/null || true
        else
            _attribution_rc=$?
            [ "$_attribution_rc" -ne 2 ] || return 2
        fi
    done
    for _round in 1 2 3 4 5; do
        _any_alive=0
        for _pid in "${_owned_pids[@]}"; do
            if kill -0 "$_pid" 2>/dev/null; then
                _any_alive=1
                break
            fi
        done
        [ "$_any_alive" = 1 ] || break
        sleep 1
    done
    for _pid in "${_owned_pids[@]}"; do
        if _cell_pid_is_attributable "$_pid"; then
            kill -KILL "$_pid" 2>/dev/null || true
        else
            _attribution_rc=$?
            [ "$_attribution_rc" -ne 2 ] || return 2
        fi
    done
}

'''

NEW_REAP_BLOCK = '''\
    reap_cell_gpu_processes || return $?
'''


DECLARE_ANCHOR = (
    'bash -c "$(declare -f run_evaluation with_huggingface_overlay); '
    'run_evaluation \\"$max_tokens_arg\\" \\"$EVAL_COUNTER\\""\n'
)
DECLARE_REPLACEMENT = (
    'bash -c "$(declare -f run_evaluation with_huggingface_overlay '
    'reap_cell_gpu_processes); run_evaluation \\"$max_tokens_arg\\" '
    '\\"$EVAL_COUNTER\\""\n'
)


COPY_TASK_ANCHOR = 'cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"\n'
FAILED_NO_MODEL_BLOCK = '''\
if [ "$SOLVE_EXIT" -ne 0 ] && { [ ! -d "${EVAL_DIR}/final_model" ] || \
   [ -z "$(find "${EVAL_DIR}/final_model" -mindepth 1 -print -quit 2>/dev/null)" ]; }; then
    echo "ERROR: solve failed with ${SOLVE_EXIT} and produced no final_model; skipping judges/evaluation" >&2
    exit "$SOLVE_EXIT"
fi
if [ "$SOLVE_EXIT" -ne 0 ]; then
    # A partial but valid model is still useful diagnostic evidence. Optional
    # judges must not abort before deterministic evaluation or replace the
    # scientist's recorded exit code.
    echo "WARNING: solve failed with ${SOLVE_EXIT}; skipping optional judges and evaluating the preserved final_model" >&2
    JUDGE_AUTH_MODE=skip
fi
'''

FAILED_NO_MODEL_BLOCK_V1 = '''\
if [ "$SOLVE_EXIT" -ne 0 ] && { [ ! -d "${EVAL_DIR}/final_model" ] || \
   [ -z "$(find "${EVAL_DIR}/final_model" -mindepth 1 -print -quit 2>/dev/null)" ]; }; then
    echo "ERROR: solve failed with ${SOLVE_EXIT} and produced no final_model; skipping judges/evaluation" >&2
    exit "$SOLVE_EXIT"
fi
'''


EOF_ANCHOR = '''\
echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"
'''
EOF_REPLACEMENT = EOF_ANCHOR + '''\

# The benchmark artifacts remain available even after a failed solve, but the
# cell itself must never be reported successful when its scientist failed or
# deterministic GSM8K evaluation failed to produce a finite numeric accuracy.
EVALUATION_EXIT=0
if [ ! -s "${EVAL_DIR}/metrics.json" ] || ! python3 -c \
   'import json,math,sys; value=json.load(open(sys.argv[1])); accuracy=value.get("accuracy") if isinstance(value,dict) else None; assert not isinstance(accuracy,bool) and isinstance(accuracy,(int,float)) and math.isfinite(float(accuracy))' \
   "${EVAL_DIR}/metrics.json"; then
    echo "ERROR: deterministic evaluation produced no valid metrics.json" >&2
    EVALUATION_EXIT=1
fi
if [ "$SOLVE_EXIT" -ne 0 ]; then
    echo "ERROR: returning recorded scientist exit code ${SOLVE_EXIT}" >&2
    exit "$SOLVE_EXIT"
fi
exit "$EVALUATION_EXIT"
'''

EOF_REPLACEMENT_V2 = EOF_ANCHOR + '''\

# The benchmark artifacts remain available even after a failed solve, but the
# cell itself must never be reported successful when its scientist failed or
# deterministic evaluation failed to produce a valid JSON object.
EVALUATION_EXIT=0
if [ ! -s "${EVAL_DIR}/metrics.json" ] || ! python3 -c \
   'import json,sys; value=json.load(open(sys.argv[1])); assert isinstance(value, dict)' \
   "${EVAL_DIR}/metrics.json"; then
    echo "ERROR: deterministic evaluation produced no valid metrics.json" >&2
    EVALUATION_EXIT=1
fi
if [ "$SOLVE_EXIT" -ne 0 ]; then
    echo "ERROR: returning recorded scientist exit code ${SOLVE_EXIT}" >&2
    exit "$SOLVE_EXIT"
fi
exit "$EVALUATION_EXIT"
'''

EOF_REPLACEMENT_V1 = EOF_ANCHOR + '''\

# The benchmark artifacts remain available even after a failed solve, but the
# cell itself must never be reported successful when its scientist failed.
if [ "$SOLVE_EXIT" -ne 0 ]; then
    echo "ERROR: returning recorded scientist exit code ${SOLVE_EXIT}" >&2
fi
exit "$SOLVE_EXIT"
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"run_task.sh: expected exactly one {label} anchor, found {count}; "
            "the runner changed shape -- update apply_study_runner.py"
        )
    return text.replace(old, new, 1)


def apply(text: str) -> str:
    if MARK in text:
        # A marker on its own (or an otherwise damaged earlier patch) is not an
        # upgrade candidate.  Check every stable capability before looking for
        # the two revisions which this patcher knows how to upgrade exactly.
        upgrade_needles = {
            "OS-visible GPU isolation probe",
            "skipping optional judges and evaluating the preserved final_model",
            "deterministic evaluation produced no valid metrics.json",
            "math.isfinite(float(accuracy))",
        }
        stable_missing = [
            needle
            for needle in COMPLETE_NEEDLES
            if needle not in upgrade_needles and needle not in text
        ]
        if stable_missing:
            raise SystemExit(
                "run_task.sh: portable study patch is incomplete; missing "
                + ", ".join(stable_missing)
            )
        # Upgrade earlier complete revisions of this local patch in place. The
        # private checkout is intentionally reusable across setup invocations.
        if "skipping optional judges and evaluating the preserved final_model" not in text:
            text = _replace_once(
                text,
                FAILED_NO_MODEL_BLOCK_V1,
                FAILED_NO_MODEL_BLOCK,
                "v1 failed-solve diagnostic",
            )
        if "deterministic evaluation produced no valid metrics.json" not in text:
            text = _replace_once(
                text,
                EOF_REPLACEMENT_V1,
                EOF_REPLACEMENT,
                "v1 evaluation footer",
            )
        elif "math.isfinite(float(accuracy))" not in text:
            text = _replace_once(
                text,
                EOF_REPLACEMENT_V2,
                EOF_REPLACEMENT,
                "v2 evaluation metric footer",
            )
        if "OS-visible GPU isolation probe" not in text:
            text = _replace_once(
                text,
                NEW_SOLVE_LINE_V2,
                NEW_SOLVE_LINE,
                "v2 OS-visible GPU isolation",
            )
        missing = [needle for needle in COMPLETE_NEEDLES if needle not in text]
        if missing:
            raise SystemExit(
                "run_task.sh: portable study patch is incomplete; missing "
                + ", ".join(missing)
            )
        return text

    text = _replace_once(text, UUID_ANCHOR, UUID_ANCHOR + UUID_BLOCK, "UUID")
    text = _replace_once(text, ENV_ANCHOR, ENV_ANCHOR + ENV_BLOCK, "API allowlist")
    text = _replace_once(text, PAYLOAD_ANCHOR, PAYLOAD_ANCHOR + PAYLOAD_BLOCK, "agent solve")
    text = _replace_once(text, OLD_GPU_BLOCK, NEW_GPU_BLOCK, "GPU pin")
    text = _replace_once(text, API_ARG_ANCHOR, API_ARG_ANCHOR + AGENT_ARG_LINE, "API argv")
    text = _replace_once(text, OLD_SOLVE_LINE, NEW_SOLVE_LINE, "agent command")
    text = _replace_once(
        text,
        SOLVE_EXIT_ANCHOR,
        SOLVE_EXIT_ANCHOR + SOLVE_EXIT_RECORD,
        "solve exit",
    )
    text = _replace_once(text, "run_evaluation() {\n", REAP_FUNCTION + "run_evaluation() {\n", "evaluation function")
    text = _replace_once(text, OLD_REAP_BLOCK, NEW_REAP_BLOCK, "unsafe GPU cleanup")
    text = _replace_once(text, DECLARE_ANCHOR, DECLARE_REPLACEMENT, "evaluation subprocess")
    text = _replace_once(
        text,
        COPY_TASK_ANCHOR,
        COPY_TASK_ANCHOR + FAILED_NO_MODEL_BLOCK,
        "task artifact copy",
    )
    text = _replace_once(text, EOF_ANCHOR, EOF_REPLACEMENT, "evaluation footer")
    return text


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    text = path.read_text()
    new = apply(text)
    if new == text:
        print(f"{path}: already patched")
        return 0
    path.write_text(new)
    print(f"{path}: patched (portable study runner)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
