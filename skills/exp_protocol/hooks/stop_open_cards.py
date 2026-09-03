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
an ordinary card can be closed by hand by filling `result` and `conclusion`.

Standard library only. The session dir is AWM_SESSION_DIR, else the hook's cwd.
Installed by `awm sandbox setup --exp-protocol --stop-hook`. Does nothing for Codex.

"Closed" means the card has a top-level ``conclusion:`` section whose own
``decision:`` key carries a real value — not null, not empty, not a template
placeholder such as ``adopt | reject | iterate | abandon_line``. The scan is
indentation-aware so a ``decision:`` inside prose or a nested mapping does
not count. A card that cannot be read at all is treated as open.
An opted-in deferred comparator additionally requires its valid close receipt.
"""

from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
from pathlib import Path
from functools import lru_cache

PLACEHOLDER = re.compile(r"[|<>]")
MAX_BLOCKS = 12
COUNTER = Path("memory") / ".stop_hook.json"
MANUAL_CLOSE = ("If the CLI is broken, fill result and conclusion in the YAML by hand; "
                "that closes it too. ")
REASON = (
    "Locked cards without a conclusion: {cards}. Ending this turn ENDS THE SESSION: "
    "there is no next turn, and every background process you started dies with it, "
    "a training run included. If a run is still going, wait for it in the foreground "
    "(`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do "
    "sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections "
    "5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record "
    "it (result.execution: failed or killed) and close the card. "
    + MANUAL_CLOSE + "(block {n} of {max})"
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


@lru_cache(maxsize=1)
def _comparator_helper():
    spec = importlib.util.spec_from_file_location(
        "_hook_comparator_receipt", Path(__file__).with_name("comparator_receipt.py")
    )
    if spec is None or spec.loader is None:
        raise OSError("comparator receipt helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lock_info(path: Path) -> dict:
    try:
        info = json.loads(path.read_text())
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


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
        unresolved = not card_is_closed(text)
        info = _lock_info(lock)
        if "deferred_comparator" in info:
            try:
                unresolved = unresolved or not _comparator_helper().completion_state(card, info)["valid"]
            except (OSError, ValueError, TypeError, ImportError, SyntaxError):
                unresolved = True
        if unresolved:
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
        reason = REASON.format(cards=", ".join(open_cards), n=blocks + 1, max=MAX_BLOCKS)
        if any("deferred_comparator" in _lock_info(session / "memory" / "cards" / f"{c}.lock.json")
               for c in open_cards):
            reason = reason.replace(MANUAL_CLOSE, "")
            reason = reason.replace("Locked cards without a conclusion:",
                                    "Locked cards awaiting conclusion or verified comparator closure:")
            reason += (" Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. "
                       "Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.")
        print(json.dumps({
            "decision": "block",
            "reason": reason,
        }))
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
