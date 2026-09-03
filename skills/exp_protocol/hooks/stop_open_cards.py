#!/usr/bin/env python3
"""Claude Code Stop hook: keep the session alive while a locked card has no conclusion.

In PostTrainBench the scientist runs as one `claude --print` turn. When the
turn ends the session ends, and Claude Code kills every background process it
started — a training run included. Round 00, cell p00r08 (job 90482): exp-02
locked, full SFT launched, and at step 207 of 4154 the scientist wrote "I'll
report back when the run finishes" and ended the turn; the run died with it and
9.4 of 10 hours went unused. So while a locked card has no conclusion, this hook
blocks the end of the turn, says why, and says how to wait. It blocks at most
MAX_BLOCKS times per session (a counter in memory/.stop_hook.json), so a
scientist that cannot close a card — the CLI broke, say — is not held forever;
a card can always be closed by hand by filling `result` and `conclusion`.

Standard library only. The session dir is AWM_SESSION_DIR, else the hook's cwd.
Installed by `awm sandbox setup --exp-protocol --stop-hook`. Does nothing for Codex.

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
MAX_BLOCKS = 12
COUNTER = Path("memory") / ".stop_hook.json"
REASON = (
    "Locked cards without a conclusion: {cards}. Ending this turn ENDS THE SESSION: "
    "there is no next turn, and every background process you started dies with it, "
    "a training run included. If a run is still going, wait for it in the foreground, on the "
    "process not the clock (`while kill -0 <pid> 2>/dev/null; do sleep 300; tail -n 1 <log>; done`, "
    "long Bash timeout; a tail that did not change across one wait means the run is dead), "
    "then evaluate, fill sections "
    "5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record "
    "it (result.execution: failed or killed) and close the card. If the CLI is broken, "
    "fill result and conclusion in the YAML by hand; that closes it too. "
    "(block {n} of {max})"
)


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


def _blocks_so_far(session: Path) -> int:
    try:
        return int(json.loads((session / COUNTER).read_text()).get("blocks", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def _record_block(session: Path, n: int) -> None:
    try:
        (session / COUNTER).parent.mkdir(parents=True, exist_ok=True)
        (session / COUNTER).write_text(json.dumps({"blocks": n}) + "\n")
    except OSError:
        pass  # the block still happens; only the bound is lost


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return 0
    session = Path(os.environ.get("AWM_SESSION_DIR", hook_input.get("cwd", ".")))
    open_cards = open_locked_cards(session)
    # `stop_hook_active` says a previous block already made Claude continue once; this
    # hook blocks again on purpose while a card is open, bounded by MAX_BLOCKS instead.
    blocks = _blocks_so_far(session)
    if open_cards and blocks < MAX_BLOCKS:
        _record_block(session, blocks + 1)
        print(json.dumps({
            "decision": "block",
            "reason": REASON.format(cards=", ".join(open_cards), n=blocks + 1, max=MAX_BLOCKS),
        }))
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
