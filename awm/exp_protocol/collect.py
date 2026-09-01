"""Per-session numbers for comparing protocol variants.

A session is a scientist's task directory (``{dir}/memory/cards``). The
official score, if present, is ``metrics.json`` in that directory or its
parent — the shape PostTrainBench writes. Everything else comes from the
cards, their locks, and their preflight summaries.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .lineage import cards_dir, load_cards
from .lock import read_lock
from .questions import REQUIRED
from .schema import get

COLUMNS = ("session", "accuracy", "n_cards", "n_closed", "n_locked", "n_locked_open",
           "preflight_fail", "pitfalls_hit", "pitfalls_cost_h", "adopted", "fields_filled")


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


def _filled(card: dict[str, Any]) -> float:
    present = sum(1 for f in REQUIRED if get(card, f) not in (None, [], ""))
    return present / len(REQUIRED)


def collect(session_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for s in session_dirs:
        s = Path(s)
        cdir = cards_dir(s)
        cards = load_cards(cdir) if cdir.is_dir() else {}
        n_closed = n_locked = n_locked_open = fails = hits = adopted = 0
        cost = 0.0
        filled: list[float] = []
        for card in cards.values():
            closed = bool(get(card, "conclusion.decision"))
            info = read_lock(Path(card["_path"]))
            n_closed += closed
            n_locked += info is not None
            n_locked_open += (info is not None and not closed)
            if info:
                fails += int((info.get("preflight") or {}).get("fail", 0) or 0)
            for hit in get(card, "situation.pitfalls_hit") or []:
                if isinstance(hit, dict):
                    hits += 1
                    if isinstance(hit.get("cost_h"), (int, float)):
                        cost += float(hit["cost_h"])
            adopted += get(card, "conclusion.decision") == "adopt"
            filled.append(_filled(card))
        rows.append({
            "session": s.name, "accuracy": _accuracy(s), "n_cards": len(cards),
            "n_closed": n_closed, "n_locked": n_locked, "n_locked_open": n_locked_open,
            "preflight_fail": fails, "pitfalls_hit": hits, "pitfalls_cost_h": cost,
            "adopted": adopted,
            "fields_filled": round(sum(filled) / len(filled), 3) if filled else "",
        })
    return rows


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS))
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in COLUMNS})
    return buf.getvalue()
