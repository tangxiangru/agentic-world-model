"""Static regression tests for PostTrainBench runtime sandbox boundaries.

These tests intentionally require neither Slurm nor a GPU. They protect values
that must cross the final-evaluation ``bash -c`` boundary and host paths that
must be visible inside its contained Apptainer sandbox.
"""

import re
from pathlib import Path


RUN_TASK = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "PostTrainBench"
    / "src"
    / "run_task.sh"
)


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
