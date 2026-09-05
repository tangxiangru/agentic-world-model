"""The ledger: all verdict files under some directories, and what they say about each skill.

A verdict is written once and never changed. The outcome lives in the card
— exp_protocol's audit record — and scoring is a pure function of the two,
computed here at read time. The truth for a verdict is the card beside it;
in an offline replay, where the card in the session must stay open, it is
the truth card the replayer kept outside the session, found by layout:
``<out>/<run>/<card>/memory/cards/<card>.verdict.json`` →
``<out>/_truth/<run>/<card>.yaml``.

One row per verdict; one summary line per (skill, backend, mode). Rates
exclude unscorable levels and verdicts flagged ``leak_suspected`` (the
backend saw the agent read outside its fence). Coverage of the effect
interval is reported next to the mean interval width, so it cannot be
bought by widening; cost is the backend's measured figure, not the
agent's own estimate.
"""

from __future__ import annotations

import csv
import json
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from awm.exp_protocol.schema import CardError, load_card, migrate_v1

from . import schema


def truth_for(verdict_path: Path) -> tuple[Path | None, dict[str, Any]]:
    """Where this verdict's outcome is, and what it says. (None, empty truth) when there is none yet."""
    verdict_path = Path(verdict_path)
    card = schema.card_path_for(verdict_path)
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

SUMMARY_COLUMNS = ("wma_skill", "backend", "model", "effort", "mode", "slice", "n", "n_scored", "n_leak_suspected",
                   "L0_hit", "L0_recall_failed", "L1_hit", "L1_recall_invalid",
                   "L2_coverage", "L2_width_mean", "L2_width_over_noise", "n_L2_scorable", "L3_hit",
                   "gpu_h_saved", "gpu_h_wrongly_killed", "cost_usd_sum", "cost_usd_mean", "cost_wall_min_mean")
#: ``summarize(rows, by=...)``: slice each group by the WMA's change types (a verdict naming two types counts
#: in both; none → "(untyped)") or by the card's method family.
SLICES = ("type", "family")


def rows(dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("exp-*.verdict*.json")):
            try:
                v = schema.load_verdict(p)
            except ValueError:
                continue
            lv = v.get("levels") or {}
            iv = (lv.get("L2_effect") or {}).get("interval")
            truth_path, truth = truth_for(p)
            scored = schema.score(v, truth) if truth_path else {}
            width = (iv[1] - iv[0]) if isinstance(iv, list) and len(iv) == 2 else None
            noise = schema.noise_floor(truth.get("n")) if truth_path else None
            out.append({
                "path": str(p), "card_id": v.get("card_id"), "wma_skill": v.get("wma_skill") or "",
                "backend": v.get("backend") or "", "model": v.get("model") or "", "effort": v.get("effort") or "",
                "mode": v.get("mode") or "",
                "change_types": [x for x in (v.get("change_types") or []) if isinstance(x, str)],
                "family": truth.get("family") or "" if truth_path else "",
                "L3_answer": (lv.get("L3_worth_now") or {}).get("answer"),
                "L2_width": width,
                "L2_noise": noise,
                "L2_width_over_noise": round(width / noise, 3) if width is not None and noise else None,
                "has_truth": truth_path is not None, "truth_path": str(truth_path) if truth_path else "",
                "scored": scored,
                "truth_levels": schema.truth_levels(truth) if truth_path else {},
                "actual": {k: truth.get(k) for k in ("execution", "decision", "wall_h", "delta")},
                "cost": v.get("cost") or {},
                "leak": bool(v.get("leak_suspected")),
            })
    return out


def rejected(dirs: list[Path]) -> dict[str, Any]:
    """Verdict files the harness moved aside (invalid JSON or schema): not verdicts, but paid for."""
    n, usd = 0, 0.0
    for d in dirs:
        for p in Path(d).rglob(f"exp-*.verdict*.json{schema.REJECTED_SUFFIX}*"):
            try:
                body = json.loads(p.read_text())
            except ValueError:
                continue
            n += 1
            cost = (body.get("rejected") or {}).get("cost") or {}
            if isinstance(cost.get("usd"), (int, float)):
                usd += float(cost["usd"])
    return {"n": n, "cost_usd_sum": round(usd, 4)}


def _rate(values: list[str], hit: tuple[str, ...]) -> float | str:
    scorable = [x for x in values if x != "unscorable"]
    if not scorable:
        return ""
    return round(sum(x in hit for x in scorable) / len(scorable), 3)


def _mean(values: list[float]) -> float | str:
    return round(sum(values) / len(values), 4) if values else ""


def _slices(r: dict[str, Any], by: str | None) -> list[str]:
    if by is None:
        return [""]
    if by == "type":
        return list(dict.fromkeys(r.get("change_types") or [])) or ["(untyped)"]
    if by == "family":
        return [r.get("family") or "(unknown)"]
    raise ValueError(f"by must be one of {SLICES}")


def summarize(all_rows: list[dict[str, Any]], by: str | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        for sl in _slices(r, by):
            groups[(r["wma_skill"], r["backend"], r["model"], r["effort"], r["mode"], sl)].append(r)
    out = []
    for (skill, backend, model, effort, mode, sl), rs in sorted(groups.items()):
        # A verdict that read outside the fence may have seen its own answer: it costs money and
        # counts in n, but it says nothing about the skill, so it stays out of every rate.
        clean = [r for r in rs if not r["leak"]]
        rec = [r for r in clean if r["has_truth"]]
        # Recall on the cards that did not run / did not yield a candidate: an agent that always says
        # yes already has the base rate on L0_hit and L1_hit; only recall shows whether it catches anything.
        failed = [r for r in rec if r["truth_levels"].get("L0") is False]
        invalid = [r for r in rec if r["truth_levels"].get("L1") is False]
        usd = [float(r["cost"]["usd"]) for r in rs if isinstance(r["cost"].get("usd"), (int, float))]
        saved = sum(float(r["actual"].get("wall_h") or 0) for r in rec
                    if r["L3_answer"] in ("no", "defer") and r["actual"].get("decision") in ("reject", "abandon_line"))
        killed = sum(float(r["actual"].get("wall_h") or 0) for r in rec
                     if r["L3_answer"] in ("no", "defer") and r["actual"].get("decision") == "adopt")
        out.append({
            "wma_skill": skill, "backend": backend, "model": model, "effort": effort, "mode": mode, "slice": sl,
            "n": len(rs), "n_scored": len(rec), "n_leak_suspected": len(rs) - len(clean),
            "L0_hit": _rate([r["scored"].get("L0", "unscorable") for r in rec], ("hit",)),
            "L0_recall_failed": _rate([r["scored"].get("L0", "unscorable") for r in failed], ("hit",)),
            "L1_hit": _rate([r["scored"].get("L1", "unscorable") for r in rec], ("hit",)),
            "L1_recall_invalid": _rate([r["scored"].get("L1", "unscorable") for r in invalid], ("hit",)),
            "L2_coverage": _rate([r["scored"].get("L2", "unscorable") for r in rec], ("in_interval",)),
            "L2_width_mean": _mean([r["L2_width"] for r in rs if r["L2_width"] is not None]),
            "L2_width_over_noise": _mean([r["L2_width_over_noise"] for r in rs if r["L2_width_over_noise"] is not None]),
            "n_L2_scorable": sum(r["scored"].get("L2", "unscorable") != "unscorable" for r in rec),
            "L3_hit": _rate([r["scored"].get("L3", "unscorable") for r in rec], ("hit",)),
            "gpu_h_saved": round(saved, 3), "gpu_h_wrongly_killed": round(killed, 3),
            "cost_usd_sum": round(sum(usd), 4) if usd else "",
            "cost_usd_mean": _mean(usd),
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
