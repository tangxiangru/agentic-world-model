#!/usr/bin/env python3
"""Copy a study agent's packaged files into its PTB sandbox home."""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: packaged study-agent payload ---"
ANCHOR = 'cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"\n'
BLOCK = f'''{MARK}
if [ -d "agents/${{AGENT}}/payload" ]; then
    mkdir -p "${{JOB_DIR}}/agent"
    cp -a "agents/${{AGENT}}/payload/." "${{JOB_DIR}}/agent/"
fi
'''


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected one agent solve-script copy")
    return text.replace(ANCHOR, ANCHOR + BLOCK, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    old = path.read_text()
    new = apply(old)
    if new != old:
        path.write_text(new)
        print(f"{path}: patched (study-agent payload)")
    else:
        print(f"{path}: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
