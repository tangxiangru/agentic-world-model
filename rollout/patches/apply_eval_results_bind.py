#!/usr/bin/env python3
"""Expose PTB's result directory to its evaluation container.

PostTrainBench passes ``$EVAL_DIR/final_model`` to the evaluator as an absolute
host path.  Relocatable unprivileged Apptainer does not automatically mount the
sibling results tree, so the path is otherwise absent inside the container.
Bind the directory at the identical path; this changes no PTB evaluation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: evaluation result bind (rollout/patches/apply_eval_results_bind.py) ---"
EXEC_ANCHOR = "    with_huggingface_overlay apptainer exec \\\n"
REPO_BIND = '        --bind "${REPO_ROOT}:${REPO_ROOT}" \\\n'
EVAL_BIND = '        --bind "${EVAL_DIR}:${EVAL_DIR}" \\\n'


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(EXEC_ANCHOR) != 1:
        raise SystemExit(
            "run_task.sh: expected exactly one evaluation Apptainer exec"
        )
    if text.count(REPO_BIND) != 1:
        raise SystemExit(
            "run_task.sh: expected exactly one evaluation repository bind"
        )
    text = text.replace(EXEC_ANCHOR, f"    {MARK}\n{EXEC_ANCHOR}", 1)
    return text.replace(REPO_BIND, REPO_BIND + EVAL_BIND, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    text = path.read_text()
    new = apply(text)
    if new == text:
        print(f"{path}: already patched")
        return 0
    path.write_text(new)
    print(f"{path}: patched (evaluation result bind)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
