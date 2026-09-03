"""Per-session numbers for comparing protocol variants.

A session is a scientist's task directory (``{dir}/memory/cards``). The
official score, if present, is ``metrics.json`` in that directory or its
parent — the shape PostTrainBench writes. Everything else comes from the
cards, their locks, and their preflight summaries.

Conclusion-based columns stay raw for cross-variant comparison. The deferred
columns report verified/failed/unverified receipt outcomes separately; a raw
``n_closed`` or ``adopted`` count is not a verified deferred comparison.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from . import comparator
from .lineage import cards_dir, load_cards
from .lock import read_lock
from .questions import REQUIRED
from .schema import get

COLUMNS = ("session", "accuracy", "hours_used", "n_cards", "n_unreadable", "n_closed", "n_locked",
           "n_locked_open", "n_relocked", "n_overrides", "preflight_fail", "pitfalls_hit",
           "pitfalls_cost_h", "adopted", "fields_filled")
DEFERRED_COLUMNS = ("n_deferred", "n_deferred_verified", "n_deferred_failed_closed", "n_deferred_unverified")
COLUMNS += DEFERRED_COLUMNS


def _label(session: Path) -> str:
    """PostTrainBench session dirs are all named ``task``; the cell is the parent directory."""
    return f"{session.parent.name}/{session.name}" if session.name == "task" else session.name


def _accuracy(session: Path) -> float | str:
    for candidate in (session / "metrics.json", session.parent / "metrics.json"):
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("accuracy"), (int, float)):
                return payload["accuracy"]
    return ""


def _hours_used(session: Path) -> float | str:
    """Wall-clock the scientist used, from PostTrainBench's ``time_taken.txt`` (``HH:MM:SS``).

    Round 00, cell p00r05, stopped at 6.3 of 10 h with "all work is complete"; a
    cell that ends early, by choice or by losing its session, is invisible to the
    card counts, so the hours go beside the score.
    """
    for candidate in (session / "time_taken.txt", session.parent / "time_taken.txt"):
        if candidate.is_file():
            try:
                parts = candidate.read_text().strip().split(":")
                h, m, sec = (int(float(x)) for x in parts[-3:])
            except (OSError, ValueError):
                continue
            return round(h + m / 60 + sec / 3600, 2)
    return ""


def _filled(card: dict[str, Any]) -> float:
    present = sum(1 for f in REQUIRED if get(card, f) not in (None, [], ""))
    return present / len(REQUIRED)


def collect(session_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for s in session_dirs:
        s = Path(s)
        cdir = cards_dir(s)
        problems: list[tuple[Path, str]] = []
        cards = load_cards(cdir, problems=problems) if cdir.is_dir() else {}
        n_closed = n_locked = n_locked_open = fails = hits = adopted = n_relocked = n_overrides = 0
        cost = 0.0
        deferred = dict.fromkeys(DEFERRED_COLUMNS, 0)
        filled: list[float] = []
        for card in cards.values():
            closed = bool(get(card, "conclusion.decision"))
            info = read_lock(Path(card["_path"]))
            state = comparator.completion_state(Path(card["_path"]), card, info)
            if state["required"]:
                deferred["n_deferred"] += 1
                if not state["valid"]:
                    deferred["n_deferred_unverified"] += 1
                elif state["outcome"] == "verified":
                    deferred["n_deferred_verified"] += 1
                else:
                    deferred["n_deferred_failed_closed"] += 1
            n_closed += closed
            n_locked += info is not None
            n_locked_open += (info is not None and not closed)
            if info:
                fails += int((info.get("preflight") or {}).get("fail", 0) or 0)
                n_relocked += bool(info.get("relocked_from"))
                n_overrides += len(info.get("overrides") or {})
                n_overrides += sum(len(h.get("overrides") or {}) for h in info.get("relocked_from") or [])
            for hit in get(card, "situation.pitfalls_hit") or []:
                if isinstance(hit, dict):
                    hits += 1
                    if isinstance(hit.get("cost_h"), (int, float)):
                        cost += float(hit["cost_h"])
            adopted += get(card, "conclusion.decision") == "adopt"
            filled.append(_filled(card))
        rows.append({
            "session": _label(s), "accuracy": _accuracy(s), "hours_used": _hours_used(s),
            "n_cards": len(cards),
            "n_unreadable": len(problems),
            "n_closed": n_closed, "n_locked": n_locked, "n_locked_open": n_locked_open,
            "n_relocked": n_relocked, "n_overrides": n_overrides,
            "preflight_fail": fails, "pitfalls_hit": hits, "pitfalls_cost_h": cost,
            "adopted": adopted,
            "fields_filled": round(sum(filled) / len(filled), 3) if filled else "",
            **deferred,
        })
    return rows


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS))
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in COLUMNS})
    return buf.getvalue()
