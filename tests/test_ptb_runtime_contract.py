"""Static regression tests for PostTrainBench runtime sandbox boundaries.

These tests intentionally require neither Slurm nor a GPU. They protect values
that must cross the final-evaluation ``bash -c`` boundary and host paths that
must be visible inside its contained Apptainer sandbox.
"""

import re
from pathlib import Path

RUN_TASK = (
    Path(__file__).resolve().parents[1] / "third_party" / "PostTrainBench" / "src" / "run_task.sh"
)
SINGLE_TASK = RUN_TASK.parent / "commit_utils" / "slurm" / "single_task.sbatch"
RECOVERY_EVAL = (
    RUN_TASK.parents[1] / "dev_utils" / "test_evaluation" / "run_only_evaluation.sh"
)
RECOVERY_JUDGES = RUN_TASK.parent / "judges" / "run_judges.sh"
JUDGE_LIB = RUN_TASK.parent / "judges" / "judge_lib.sh"


def shell_function(script: str, name: str, next_name: str) -> str:
    """Return one top-level shell function using the following function marker."""
    start = script.index(f"{name}() {{")
    end = script.index(f"{next_name}() {{", start)
    return script[start:end]


def test_job_tmp_is_exported_for_final_eval_retry_subshell():
    script = RUN_TASK.read_text(encoding="utf-8")
    run_evaluation = shell_function(script, "run_evaluation", "run_evaluation_with_retry")

    assert re.search(r'^export JOB_TMP="\$\{TMP_SUBDIR\}/tmp"$', script, re.MULTILINE)
    assert 'bash -c "$(declare -f run_evaluation' in script
    assert '--bind "${JOB_TMP}:/tmp"' in run_evaluation


def test_result_directory_is_bound_into_final_eval_container():
    script = RUN_TASK.read_text(encoding="utf-8")
    run_evaluation = shell_function(script, "run_evaluation", "run_evaluation_with_retry")

    assert '--bind "${EVAL_DIR}:${EVAL_DIR}"' in run_evaluation
    assert '--model-path "$EVAL_DIR/final_model"' in run_evaluation
    assert '--json-output-file "${EVAL_DIR}/metrics.json"' in run_evaluation


def test_gres_container_uses_logical_cuda_selector_and_records_physical_ids():
    script = SINGLE_TASK.read_text(encoding="utf-8")

    assert 'POST_TRAIN_BENCH_ALLOCATED_GPUS="${SLURM_JOB_GPUS:-}"' in script
    assert 'POST_TRAIN_BENCH_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES}"' in script
    assert 'POST_TRAIN_BENCH_VISIBLE_GPUS="${SLURM_JOB_GPUS' not in script


def test_root_owned_allocation_drops_privileges_before_task_setup():
    script = SINGLE_TASK.read_text(encoding="utf-8")
    privilege_drop = script.index(
        'exec /usr/sbin/runuser --user "$RUN_AS_USER" -- /bin/bash "$ENTRYPOINT"'
    )
    argument_check = script.index('if [ "$#" -ne 7 ]')

    assert 'if [ "$(id -u)" = "0" ]; then' in script
    assert privilege_drop < argument_check


def test_final_eval_caches_live_on_job_local_storage():
    script = RUN_TASK.read_text(encoding="utf-8")
    run_evaluation = shell_function(script, "run_evaluation", "run_evaluation_with_retry")

    assert 'local eval_home="${JOB_TMP}/eval-home"' in run_evaluation
    assert '--home "${eval_home}:${HOME}"' in run_evaluation
    assert '--env "VLLM_CACHE_ROOT=${HOME}/.cache/vllm"' in run_evaluation
    assert '--env "TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor"' in run_evaluation
    assert '--env "TRITON_CACHE_DIR=${HOME}/.cache/triton"' in run_evaluation


def test_recovery_eval_is_allocation_scoped_and_uses_unique_scratch():
    script = RECOVERY_EVAL.read_text(encoding="utf-8")

    assert 'RANDOM_UUID="${SLURM_JOB_ID:-$(uuidgen)}"' in script
    assert 'RECOVERY_SCRATCH_ROOT="${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"' in script
    assert '--home "${eval_home}:${HOME}"' in script
    assert '--bind "${JOB_TMP}:/tmp"' in script
    assert '--env "TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor"' in script
    assert '--env "TRITON_CACHE_DIR=${HOME}/.cache/triton"' in script
    assert 'if [ ! -s "${EVAL_DIR}/metrics.json" ]; then' in script
    assert "--query-compute-apps=pid" not in script
    assert "xargs -r kill" not in script


def test_recovery_judges_serialize_official_auth_and_load_site_apptainer():
    script = RECOVERY_JUDGES.read_text(encoding="utf-8")
    judge_lib = JUDGE_LIB.read_text(encoding="utf-8")

    assert 'export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):${PATH}"' in script
    assert 'exec 9>"$JUDGE_LOCK_FILE"' in script
    assert "flock -x 9" in script
    assert "flock -u 9" in script
    assert 'JUDGE_SCRATCH_ROOT="${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"' in script
    assert 'mktemp -d -p "$JUDGE_SCRATCH_ROOT" judge-recovery.XXXXXX' in script
    assert 'python3 "$JUDGES_DIR/get_judge_prompt.py"' in judge_lib
    assert 'python3 "$JUDGES_REPO_ROOT/src/trace_parsing/parse_trace.py"' in judge_lib
