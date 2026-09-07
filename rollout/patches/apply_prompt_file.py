#!/usr/bin/env python3
"""Give only the study agents PTB's generated prompt as a task file.

Apptainer's multiline ``--env PROMPT=...`` transport can truncate the long
study prompts. The task directory is already part of PTB's normal sandbox
home, so copying the generated prompt there needs no additional bind or
runtime validation. Other PTB agents keep their original behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: prompt file for study agents ---"
ANCHOR = 'echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"\n'
BLOCK = f'''{MARK}
case "$AGENT" in
    claude_noprior_noawm|claude_fulltraj_noawm|claude_wm|claude_recorder|opencode_recorder)
        cp "${{EVAL_DIR}}/prompt.txt" "${{JOB_DIR}}/task/instruction.md"
        ;;
esac
'''


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected exactly one prompt.txt write")
    return text.replace(ANCHOR, ANCHOR + BLOCK, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    old = path.read_text()
    new = apply(old)
    if new != old:
        path.write_text(new)
        print(f"{path}: patched (study prompt file)")
    else:
        print(f"{path}: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
