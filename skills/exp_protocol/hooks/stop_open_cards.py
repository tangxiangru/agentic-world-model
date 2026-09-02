#!/usr/bin/env python3
"""Claude Code Stop hook: block the end of a turn once if a locked card has no conclusion.

Standard library only. The session dir is AWM_SESSION_DIR, else the hook's cwd.
Optional; not installed by default. Does nothing for Codex.

"Closed" means the card has a top-level ``conclusion:`` section whose own
``decision:`` key carries a real value — not null, not empty, not a template
placeholder such as ``adopt | reject | iterate | abandon_line``. The scan is
indentation-aware so a ``decision:`` inside prose or a nested mapping does
not count. A card that cannot be read at all is treated as open.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"[|<>]")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def card_is_closed(text: str) -> bool:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() != "conclusion:":
            continue
        for nxt in lines[i + 1:]:
            if not nxt.strip() or nxt.lstrip().startswith("#"):
                continue
            if _indent(nxt) == 0:
                return False  # the section ended without a decision
            m = re.match(r"^(\s+)decision:\s*(.*?)\s*(#.*)?$", nxt)
            if m and _indent(nxt) == _indent_of_first_child(lines, i):
                value = m.group(2).strip().strip("'\"")
                return bool(value) and value not in ("null", "~") and not PLACEHOLDER.search(value)
        return False
    return False


def _indent_of_first_child(lines: list[str], section_index: int) -> int:
    for nxt in lines[section_index + 1:]:
        if nxt.strip() and not nxt.lstrip().startswith("#"):
            return _indent(nxt)
    return -1


def open_locked_cards(session: Path) -> list[str]:
    cards = session / "memory" / "cards"
    if not cards.is_dir():
        return []
    out = []
    for lock in sorted(cards.glob("exp-*.lock.json")):
        card = cards / (lock.name.replace(".lock.json", ".yaml"))
        if not card.is_file():
            continue
        try:
            text = card.read_text()
        except OSError:
            out.append(card.stem)
            continue
        if not card_is_closed(text):
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
