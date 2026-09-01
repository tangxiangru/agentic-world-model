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


def lost(
    run_id: str, events: list[dict[str, Any]], root: Path | None = None
) -> list[dict[str, Any]]:
    """The command launches that never reached the event stream, as events.

    A set difference: every ``item.started`` line in the raw file, minus every
    line some event's ``source_ref`` already claims. An annotator pointed out
    that this is mechanical — locating them needs no judgement, only reading
    them does — and that repairing without it makes things worse. A displaced
    ``args.command`` is sometimes the stream's *only* record of a real
    evaluation; putting the true command back then deletes that evaluation from
    every table. Reinstating the lost launches is what makes the repair whole.

    The synthetic events carry the raw timestamp and an ``i`` interpolated after
    the nearest earlier claimed line, so they sort into place. They are marked
    ``origin: "reinstated"`` — they have no ``tool_result``, and any duration
    read off them is a launch time only.
    """
    path = raw_path(run_id, root)
    if path is None:
        return []
    truth = commands_by_line(path)
    stamps = _timestamps_by_line(path)
    claimed = {
        (e.get("source_ref") or {}).get("line") for e in events if e.get("source_ref")
    }
    anchor = {
        (e.get("source_ref") or {}).get("line"): (e.get("i"), e.get("turn"))
        for e in events if e.get("source_ref")
    }
    out: list[dict[str, Any]] = []
    for line, command in sorted(truth.items()):
        if line in claimed:
            continue
        earlier = [ln for ln in anchor if ln is not None and ln < line]
        base_i, turn = anchor[max(earlier)] if earlier else (0, None)
        out.append({
            "run_id": run_id,
            "i": (base_i or 0) + 0.5,
            "ts": stamps.get(line),
            "turn": turn,
            "type": "tool_use",
            "tool": "command_execution",
            "origin": "reinstated",
            "source_ref": {"file": path.name, "line": line},
            "args": {"command": command},
        })
    return out


def _timestamps_by_line(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for number, text in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        m = re.match(r"^\[([^\]]+)\]", text)
        if m:
            out[number] = m.group(1)
    return out


__all__ = ["commands_by_line", "lost", "raw_path", "repair"]
