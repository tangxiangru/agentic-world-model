"""The ledger: all verdict files under some directories, and what they say about each skill.

A verdict is written once and never changed. The outcome lives in the card
— exp_protocol's audit record — and scoring is a pure function of the two,
computed here at read time. The truth for a verdict is the card beside it;
in an offline replay, where the card in the session must stay open, it is
the truth card the replayer kept outside the session, found by layout:
``<out>/<run>/<card>/memory/cards/<card>.verdict.json`` →
``<out>/_truth/<run>/<card>.yaml``.

One row per verdict; one summary line per (skill, backend, mode). Rates
exclude unscorable levels. Coverage of the effect interval is reported
next to the mean interval width, so it cannot be bought by widening.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from awm.exp_protocol.schema import CardError, load_card, migrate_v1

from . import schema


def truth_for(verdict_path: Path) -> tuple[Path | None, dict[str, Any]]:
    """Where this verdict's outcome is, and what it says. (None, empty truth) when there is none yet."""
    verdict_path = Path(verdict_path)
    card = verdict_path.with_name(verdict_path.name.replace(".verdict.json", ".yaml"))
    candidates = [card]
    # replay layout: .../<out>/<run>/<card>/memory/cards/<card>.verdict.json
    try:
        session = verdict_path.parents[2]
        run_dir = session.parent
        candidates.append(run_dir.parent / "_truth" / run_dir.name / f"{session.name}.yaml")
    except IndexError:
        pass
    for c in candidates:
        if not c.is_file():
            continue
        try:
            t = schema.truth_from_card(migrate_v1(load_card(c)))
        except CardError:
            continue
        if t["execution"] is not None or t["decision"] is not None:
            return c, t
    return None, schema.truth_from_card({})

SUMMARY_COLUMNS = ("wma_skill", "backend", "mode", "n", "n_reconciled", "L0_hit", "L1_hit", "L2_coverage",
                   "L2_width_mean", "L3_hit", "gpu_h_saved", "gpu_h_wrongly_killed", "cost_cpu_min_mean",
                   "cost_wall_min_mean")


def rows(dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("exp-*.verdict.json")):
            try:
                v = schema.load_verdict(p)
            except ValueError:
                continue
            lv = v.get("levels") or {}
            iv = (lv.get("L2_effect") or {}).get("interval")
            truth_path, truth = truth_for(p)
            scored = schema.score(v, truth) if truth_path else {}
            out.append({
                "path": str(p), "card_id": v.get("card_id"), "wma_skill": v.get("wma_skill") or "",
                "backend": v.get("backend") or "", "mode": v.get("mode") or "",
                "L3_answer": (lv.get("L3_worth_now") or {}).get("answer"),
                "L2_width": (iv[1] - iv[0]) if isinstance(iv, list) and len(iv) == 2 else None,
                "reconciled": truth_path is not None, "truth_path": str(truth_path) if truth_path else "",
                "scored": scored,
                "actual": {k: truth.get(k) for k in ("execution", "decision", "wall_h", "delta")},
                "cost": v.get("cost") or {},
            })
    return out


def _rate(values: list[str], hit: tuple[str, ...]) -> float | str:
    scorable = [x for x in values if x != "unscorable"]
    if not scorable:
        return ""
    return round(sum(x in hit for x in scorable) / len(scorable), 3)


def _mean(values: list[float]) -> float | str:
    return round(sum(values) / len(values), 4) if values else ""


def summarize(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        groups[(r["wma_skill"], r["backend"], r["mode"])].append(r)
    out = []
    for (skill, backend, mode), rs in sorted(groups.items()):
        rec = [r for r in rs if r["reconciled"]]
        saved = sum(float(r["actual"].get("wall_h") or 0) for r in rec
                    if r["L3_answer"] in ("no", "defer") and r["actual"].get("decision") in ("reject", "abandon_line"))
        killed = sum(float(r["actual"].get("wall_h") or 0) for r in rec
                     if r["L3_answer"] in ("no", "defer") and r["actual"].get("decision") == "adopt")
        out.append({
            "wma_skill": skill, "backend": backend, "mode": mode, "n": len(rs), "n_reconciled": len(rec),
            "L0_hit": _rate([r["scored"].get("L0", "unscorable") for r in rec], ("hit",)),
            "L1_hit": _rate([r["scored"].get("L1", "unscorable") for r in rec], ("hit",)),
            "L2_coverage": _rate([r["scored"].get("L2", "unscorable") for r in rec], ("in_interval",)),
            "L2_width_mean": _mean([r["L2_width"] for r in rs if r["L2_width"] is not None]),
            "L3_hit": _rate([r["scored"].get("L3", "unscorable") for r in rec], ("hit",)),
            "gpu_h_saved": round(saved, 3), "gpu_h_wrongly_killed": round(killed, 3),
            "cost_cpu_min_mean": _mean([float(r["cost"].get("cpu_min") or 0) for r in rs]),
            "cost_wall_min_mean": _mean([float(r["cost"].get("wall_min") or 0) for r in rs]),
        })
    return out


def to_csv(summary: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(SUMMARY_COLUMNS))
    w.writeheader()
    for s in summary:
        w.writerow({k: s.get(k, "") for k in SUMMARY_COLUMNS})
    return buf.getvalue()


def render(summary: list[dict[str, Any]]) -> str:
    if not summary:
        return "(no verdicts)"
    head = "| " + " | ".join(SUMMARY_COLUMNS) + " |"
    sep = "|" + "---|" * len(SUMMARY_COLUMNS)
    body = ["| " + " | ".join(str(s.get(k, "")) for k in SUMMARY_COLUMNS) + " |" for s in summary]
    return "\n".join([head, sep, *body])
