"""Per-card verdict funnel for WMA cells: requested → answered → delivered before the launch? → read → acted on.

    .venv/bin/python tools/wma-rca/uptake.py results/ptb/<batch> [...] --out <dir>

Writes uptake.json / uptake.md. Control cells appear with no requests. The uptake class is a
mechanical first pass; `timeline.py <cell>` is the hand-reading that refines it (see README).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcalib import (batch_cells, cell_arm, g, hhmm, launch_time, load_cards, load_locks,  # noqa: E402
                    load_queue, load_verdicts, median, num, parse_transcript, read_time, suggestion_text,
                    ts, write_outputs)

#: what a scientist does when a common suggestion is taken; matched on Bash calls after the read
ACTIONS = {
    "save_steps": re.compile(r"save[-_]steps|save_strategy", re.I),
    "score_checkpoint": re.compile(r"checkpoint-\d+|ckpt-\d+|epoch-?1", re.I),
    "generation_config": re.compile(r"generation_config", re.I),
    "concurrency": re.compile(r"max-connections", re.I),
    "full_set": re.compile(r"--limit -1|--limit 1319", re.I),
}
SUGGESTS = {
    "save_steps": re.compile(r"save[-_]steps|save_strategy|intermediate checkpoint", re.I),
    "score_checkpoint": re.compile(r"checkpoint-\d+|epoch-1|score .*checkpoints?|halfway", re.I),
    "generation_config": re.compile(r"generation_config", re.I),
    "concurrency": re.compile(r"max-connections|concurrency", re.I),
    "full_set": re.compile(r"1319|--limit -1|full (test )?set", re.I),
}


def card_rows(batch: str, cell: str, cell_dir: Path) -> list[dict]:
    calls, results = parse_transcript(cell_dir)
    cards, locks, verdicts = load_cards(cell_dir), load_locks(cell_dir), load_verdicts(cell_dir)
    requests, responses = load_queue(cell_dir)
    arm = cell_arm(cell_dir)
    rows = []
    for card_id, card in sorted(cards.items()):
        lock = locks.get(card_id) or {}
        locked = ts(lock.get("locked_at"))
        verdict = verdicts.get(card_id)
        rejected = bool(verdict and verdict.get("_rejected"))
        issued = ts(verdict.get("issued_at")) if verdict and not rejected else None
        launch = launch_time(card_id, card, locked, calls)
        read = read_time(card_id, issued, calls, results)
        wall_h = num(g(card, "result.wall_h")) or 0.0
        run_end = (launch + timedelta(hours=wall_h)) if launch and wall_h else None
        gate = lock.get("wma") or {}          # since 2026-09-03: lock waits and records
        suggestions = suggestion_text(verdict) if verdict and not rejected else ""
        suggested = {k: bool(p.search(suggestions)) for k, p in SUGGESTS.items()}
        acted = {}
        if read:
            # only what happens between the read and the next card's lock counts as a reaction to
            # this verdict; later cards have their own verdicts (an upper bound still: see README)
            later_locks = [ts(lk.get("locked_at")) for cid, lk in locks.items() if cid != card_id]
            window_end = min([t for t in later_locks if t and t > read] or [None], default=None)
            later = [(t, text) for t, tool, text in calls
                     if tool == "Bash" and t and t >= read and (window_end is None or t < window_end)]
            acted = {k: sum(1 for _t, text in later if p.search(text)) for k, p in ACTIONS.items()}
        taken = [k for k in suggested if suggested[k] and acted.get(k)]
        if issued is None:
            uptake = None
        elif read is None:
            uptake = "never-read"
        elif run_end and read >= run_end:
            uptake = "post-hoc"
        elif taken:
            uptake = "adopted"
        else:
            uptake = "ignored"
        rows.append({
            "batch": batch, "cell": cell, "arm": arm, "card": card_id,
            "family": g(card, "setup.method.family"), "planned_h": num(g(card, "setup.budget.planned_h")),
            "wall_h": wall_h, "locked": hhmm(locked), "requested": hhmm(ts(requests.get(card_id))),
            "response": (responses.get(card_id) or {}).get("state"),
            "error": ((responses.get(card_id) or {}).get("error") or "")[:80] or None,
            "verdict": "rejected" if rejected else ("delivered" if verdict else None),
            "issued": hhmm(issued), "launch": hhmm(launch),
            "verdict_before_launch": (issued < launch) if (issued and launch) else None,
            "read": hhmm(read) if read else ("NEVER" if issued else None),
            "read_lag_min": round((read - issued).total_seconds() / 60) if (read and issued) else None,
            "lock_wma_state": gate.get("state"), "lock_wma_waited_s": gate.get("waited_s"),
            "L0": g(verdict, "levels.L0_runs.answer") if verdict else None,
            "L1": g(verdict, "levels.L1_valid.answer") if verdict else None,
            "L2": g(verdict, "levels.L2_effect.direction") if verdict else None,
            "L2_interval": g(verdict, "levels.L2_effect.interval") if verdict else None,
            "L3": g(verdict, "levels.L3_worth_now.answer") if verdict else None,
            "suggested": [k for k, v in suggested.items() if v], "acted_after_read": taken,
            "uptake": uptake, "decision": g(card, "conclusion.decision"),
            "delta": num((g(card, "result.measurements") or [{}])[0].get("delta_vs_comparator"))
            if g(card, "result.measurements") else None,
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    wma = [r for r in rows if r["arm"] == "wma"]
    delivered = [r for r in wma if r["verdict"] == "delivered"]
    requested = [r for r in wma if r["requested"]]
    before = [r for r in delivered if r["verdict_before_launch"] is not None]
    gate_states = Counter(r["lock_wma_state"] for r in wma if r["lock_wma_state"])
    waits = [r["lock_wma_waited_s"] for r in wma if isinstance(r.get("lock_wma_waited_s"), (int, float))]
    by_cell = {}
    for r in wma:
        c = by_cell.setdefault(r["cell"], {"cards": 0, "requested": 0, "delivered": 0, "actions": 0,
                                           "last_requested_card": None})
        c["cards"] += 1
        if r["requested"]:
            c["requested"] += 1
            c["last_requested_card"] = r["card"]
        if r["verdict"] == "delivered":
            c["delivered"] += 1
        if r["uptake"] == "adopted":
            c["actions"] += 1
    return {
        "cards_wma_arm": len(wma), "requested": len(requested), "delivered": len(delivered),
        "requests_without_verdict": len([r for r in requested if r["verdict"] != "delivered"]),
        "verdict_before_launch": {"yes": sum(1 for r in before if r["verdict_before_launch"]),
                                  "of": len(before)},
        "read": Counter("never" if r["read"] == "NEVER" else "read" for r in delivered),
        "read_lag_min_median": median([r["read_lag_min"] for r in delivered if r["read_lag_min"] is not None]),
        "uptake": Counter(r["uptake"] for r in delivered),
        "L3": Counter(str(r["L3"]) for r in delivered),
        "L0_or_L1_no": [f"{r['cell']}/{r['card']}" for r in delivered if "no" in (str(r["L0"]), str(r["L1"]))],
        "suggested_checkpoint_scoring": sum(1 for r in delivered if "score_checkpoint" in r["suggested"]
                                            or "save_steps" in r["suggested"]),
        "acted_checkpoint_scoring": sum(1 for r in delivered if "score_checkpoint" in r["acted_after_read"]
                                        or "save_steps" in r["acted_after_read"]),
        "lock_gate_states": dict(gate_states), "lock_gate_wait_s_median": median(waits),
        "train_h_total": round(sum(r["wall_h"] for r in wma if r["family"] not in ("other", "decode-config")), 1),
        "train_h_with_verdict": round(sum(r["wall_h"] for r in delivered
                                          if r["family"] not in ("other", "decode-config")), 1),
        "by_cell": by_cell,
    }


def markdown(rows: list[dict], agg: dict) -> str:
    out = ["# Verdict uptake", "",
           f"WMA-arm cards {agg['cards_wma_arm']}; requested {agg['requested']}; delivered {agg['delivered']}; "
           f"requests without a verdict {agg['requests_without_verdict']}; verdict before launch "
           f"{agg['verdict_before_launch']['yes']}/{agg['verdict_before_launch']['of']}; "
           f"read {dict(agg['read'])}; uptake {dict(agg['uptake'])}; L3 {dict(agg['L3'])}; "
           f"L0/L1 no: {agg['L0_or_L1_no'] or 'none'}; checkpoint scoring suggested/acted "
           f"{agg['suggested_checkpoint_scoring']}/{agg['acted_checkpoint_scoring']}; training hours with a verdict "
           f"{agg['train_h_with_verdict']}/{agg['train_h_total']}; lock gate states {agg['lock_gate_states']} "
           f"(median wait {agg['lock_gate_wait_s_median']} s)", "",
           "| cell | card | family | locked | requested | response | verdict | issued | launch | before launch | read | lag min | gate | L3 | L2 | uptake | acted |",
           "|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|"]
    for r in rows:
        if r["arm"] != "wma":
            continue
        out.append(f"| {r['cell']} | {r['card']} | {r['family']} | {r['locked']} | {r['requested']} | {r['response']} | "
                   f"{r['verdict']} | {r['issued']} | {r['launch']} | {r['verdict_before_launch']} | {r['read']} | "
                   f"{r['read_lag_min'] if r['read_lag_min'] is not None else ''} | {r['lock_wma_state'] or ''} | "
                   f"{r['L3']} | {r['L2']} {r['L2_interval'] or ''} | {r['uptake'] or ''} | "
                   f"{','.join(r['acted_after_read'])} |")
    out += ["", "## Per cell", "", "| cell | cards | requested | delivered | actions | last requested card |", "|---|---:|---:|---:|---:|---|"]
    for cell, c in sorted(agg["by_cell"].items()):
        out.append(f"| {cell} | {c['cards']} | {c['requested']} | {c['delivered']} | {c['actions']} | {c['last_requested_card']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+", help="results/ptb/<batch> directories")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for batch, cell, cell_dir, _inflight in batch_cells(args.batches):
        rows.extend(card_rows(batch, cell, cell_dir))
    agg = aggregate(rows)
    write_outputs(Path(args.out), "uptake", {"rows": rows, "aggregates": agg}, markdown(rows, agg))
    print(markdown(rows, agg).splitlines()[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
