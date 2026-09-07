#!/usr/bin/env python3
"""Stop passing the study prompt through Apptainer's ``--env PROMPT``.

Apptainer serialises ``--env`` values into a generated ``/.inject-apptainer-env.sh``
that the container sources. A long multiline markdown prompt does not survive
that round trip: the generated script fails to parse (``'>' must be followed by
a word``) and **every** variable in it is lost, not just PROMPT — including the
Vertex routing names, ``AGENT_CONFIG`` and ``MODEL_TO_TRAIN``. The agent then
starts with no model, no base model and no credentials.

``apply_prompt_file.py`` already gives the study agents the prompt as
``task/instruction.md``, and their ``solve.sh`` prefers that file, so the
environment copy is redundant for them. Blank it for those agents and leave
every other PostTrainBench agent untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: keep the long study prompt out of the container environment ---"
STUDY_AGENTS = "claude_noprior_noawm|claude_fulltraj_noawm|claude_wm|claude_recorder|opencode_recorder"
ANCHOR = '    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \\\n'
OLD_ENV = '        --env PROMPT="${PROMPT}" \\\n'
NEW_ENV = '        --env PROMPT="${AGENT_PROMPT_ENV}" \\\n'
BLOCK = f'''{MARK}
AGENT_PROMPT_ENV="${{PROMPT}}"
case "$AGENT" in
    {STUDY_AGENTS})
        # These agents read task/instruction.md; a 7 KB --env value corrupts the
        # whole injected environment.
        AGENT_PROMPT_ENV=""
        ;;
esac
'''


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(OLD_ENV) != 1:
        raise SystemExit("run_task.sh: expected exactly one --env PROMPT for the agent")
    if text.count(ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected exactly one agent timeout wrapper")
    text = text.replace(OLD_ENV, NEW_ENV, 1)
    return text.replace(ANCHOR, BLOCK + ANCHOR, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    old = path.read_text()
    new = apply(old)
    if new != old:
        path.write_text(new)
        print(f"{path}: patched (study prompt kept out of the container environment)")
    else:
        print(f"{path}: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
