#!/usr/bin/env python3
"""Claude Code Stop hook: block the end of a turn once if a locked card has no conclusion.

Standard library only. The session dir is AWM_SESSION_DIR, else the hook's cwd.
Optional; not installed by default. Does nothing for Codex.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def open_locked_cards(session: Path) -> list[str]:
    cards = session / "memory" / "cards"
    if not cards.is_dir():
        return []
    out = []
    for lock in sorted(cards.glob("exp-*.lock.json")):
        card = cards / (lock.name.replace(".lock.json", ".yaml"))
        if not card.is_file():
            continue
        text = card.read_text()
        # A closed card has a conclusion with a decision; a flat scan is enough.
        if "\nconclusion:" in text and "decision:" in text.split("\nconclusion:", 1)[1]:
            continue
        out.append(card.stem)
    return out


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return 0
    session = Path(os.environ.get("AWM_SESSION_DIR", hook_input.get("cwd", ".")))
    open_cards = open_locked_cards(session)
    if open_cards and not hook_input.get("stop_hook_active", False):
        print(json.dumps({
            "decision": "block",
            "reason": ("Locked cards without a conclusion: " + ", ".join(open_cards) +
                       ". If the run finished, fill sections 5-6 and run "
                       "`awm exp_protocol close --dir <dir> <card>`; if it is still running, say so."),
        }))
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
