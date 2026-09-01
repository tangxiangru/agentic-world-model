"""Put back the commands the conversion overwrote.

codex restarts item numbering at ``item_1`` in every turn, so a reprompt run
holds several ``item_12`` events; the conversion keyed on that id and let the
later episode's arguments land on the earlier episode's event. The result is a
record that keeps one turn's index and timestamp beside another turn's command.

It looked unrepairable until an annotator noticed every event carries
``source_ref.line``, a pointer back into the raw ``solve_out.txt``. The true
command is right there, so the repair is a lookup rather than a reconversion.

Confined to what it can prove: a command is replaced only when the raw line
that event points at is a ``command_execution`` whose text differs. Everything
else is left exactly as it was.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from awm import paths

#: ``[2026-04-25T18:24:57Z] {"type":"item.started",…}``
_STAMP = re.compile(r"^\[[^\]]+\]\s*")


def raw_path(run_id: str, root: Path | None = None) -> Path | None:
    """Where this run's ``solve_out.txt`` lives, if it was kept."""
    if "__" not in run_id:
        return None
    family, tail = run_id.split("__", 1)
    base = root or paths.raw_dir("posttrainbench")
    candidate = Path(base) / family / tail / "solve_out.txt"
    return candidate if candidate.exists() else None


def commands_by_line(path: Path) -> dict[int, str]:
    """``line number -> the command that line started``, 1-based."""
    out: dict[int, str] = {}
    for number, text in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        try:
            payload = json.loads(_STAMP.sub("", text))
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        # Only the line that *started* a command. ``item.completed`` repeats
        # the command, and counting both doubled the denominator — every codex
        # run then read as having lost exactly half its commands.
        if payload.get("type") != "item.started":
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        command = item.get("command") or (
            action.get("command") if isinstance(action, dict) else None
        )
        if isinstance(command, str) and command:
            out[number] = command
    return out


def repair(
    run_id: str, events: list[dict[str, Any]], root: Path | None = None
) -> tuple[list[dict[str, Any]], int]:
    """The events with overwritten commands restored, and how many were.

    Returns the events unchanged when the raw file is gone — the caller gets a
    count of zero and can say so, rather than being handed a silent no-op.
    """
    path = raw_path(run_id, root)
    if path is None:
        return events, 0
    truth = commands_by_line(path)
    repaired = 0
    for event in events:
        if event.get("type") != "tool_use":
            continue
        line = (event.get("source_ref") or {}).get("line")
        actual = truth.get(line)
        if actual is None:
            continue
        args = event.get("args") or {}
        if args.get("command") and args["command"] != actual:
            event["args"] = {**args, "command": actual, "command_before_repair": args["command"]}
            repaired += 1
    return events, repaired


__all__ = ["commands_by_line", "raw_path", "repair"]
