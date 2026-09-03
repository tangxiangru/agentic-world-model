"""Per-verdict rows and aggregates by skill hash: levels, interval width against the noise floor,
probes, history citations, checkpoint suggestions, cost and turns.

    .venv/bin/python tools/wma-rca/verdicts.py results/ptb/<batch> [...] [--inflight] --out <dir>

Reads exp-NN.verdict.json (and .rejected) beside the cards and the private transcripts under
wma_private/ for cost and turns (harvested cells) or <cell>.inflight/wma_private/ (--inflight).
Nothing from a transcript is printed beyond counts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcalib import batch_cells, g, load_cards, load_verdicts, median, noise_floor, num, write_outputs  # noqa: E402

CKPT = re.compile(r"save[-_]steps|save_strategy|checkpoint sweep|intermediate checkpoints?|checkpoint-\d+|"
                  r"epoch-?1 checkpoint|every \d+ steps|\bC5\b", re.I)


def transcript_meta(cell_dir: Path, card_id: str) -> dict:
    """Turns, cost and the verdict payload (when the file beside the card is missing or rejected)."""
    path = cell_dir / "wma_private" / f"{card_id}.transcript.jsonl.gz"
    if not path.is_file():
        return {}
    meta: dict = {}
    try:
        for line in gzip.open(path, "rt", errors="replace"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "result":
                meta = {"turns": row.get("num_turns"), "cost_usd": row.get("total_cost_usd"),
                        "wall_min": round((row.get("duration_ms") or 0) / 60000, 1)}
    except (OSError, ValueError):
        return {}
    return meta


def verdict_row(batch: str, cell: str, card_id: str, verdict: dict, card: dict, meta: dict) -> dict:
    levels = verdict.get("levels") or {}
    l2 = levels.get("L2_effect") or {}
    interval = l2.get("interval")
    width = None
    if isinstance(interval, list) and len(interval) == 2 and all(isinstance(x, (int, float)) for x in interval):
        width = float(interval[1]) - float(interval[0])
    n = num(g(card, "evaluation.protocol.n"))
    floor = noise_floor(n)
    probes = [p for p in (verdict.get("probes") or []) if isinstance(p, dict)]
    evidence = [e for e in (verdict.get("evidence") or []) if isinstance(e, dict)]
    l2_basis = set(l2.get("basis") or [])
    suggestions = json.dumps(verdict.get("suggestions") or {})
    return {
        "batch": batch, "cell": cell, "card": card_id, "skill": verdict.get("wma_skill"),
        "rejected": bool(verdict.get("_rejected")), "change_types": verdict.get("change_types") or [],
        "family": g(card, "setup.method.family"),
        "L0": (levels.get("L0_runs") or {}).get("answer"), "L0_conf": (levels.get("L0_runs") or {}).get("confidence"),
        "L1": (levels.get("L1_valid") or {}).get("answer"), "L1_conf": (levels.get("L1_valid") or {}).get("confidence"),
        "L2": l2.get("direction"), "L2_interval": interval, "L2_conf": l2.get("confidence"),
        "L3": (levels.get("L3_worth_now") or {}).get("answer"),
        "n": n, "floor": floor, "width": width,
        "width_over_floor": round(width / floor, 2) if width is not None and floor else None,
        "n_probes": len(probes), "probe_kinds": [p.get("kind") for p in probes],
        "probe_changed": [p.get("changed") for p in probes],
        "probe_changed_L01": any(p.get("changed") in ("L0", "L1") for p in probes),
        "cites_history": any("history" in str(e.get("path", "")) for e in evidence),
        "L2_cites_history": any("history" in str(e.get("path", "")) for e in evidence if e.get("id") in l2_basis),
        "suggests_checkpoint": bool(CKPT.search(suggestions)),
        "truth_delta": num((g(card, "result.measurements") or [{}])[0].get("delta_vs_comparator"))
        if g(card, "result.measurements") else None,
        "execution": g(card, "result.execution"),
        **meta,
    }


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["rejected"]:
            groups[str(r["skill"])].append(r)
    for skill, rs in groups.items():
        c34 = [r for r in rs if any(t in ("C3", "C4") for t in r["change_types"])]
        widths = [r for r in rs if r["width_over_floor"] is not None]
        covered = [r for r in rs if r["truth_delta"] is not None and isinstance(r["L2_interval"], list)
                   and len(r["L2_interval"]) == 2 and all(isinstance(x, (int, float)) for x in r["L2_interval"])]
        hits = [r for r in covered if r["L2_interval"][0] <= r["truth_delta"] <= r["L2_interval"][1]]
        out[skill] = {
            "n_verdicts": len(rs), "n_c34": len(c34),
            "L0": Counter(str(r["L0"]) for r in rs), "L1": Counter(str(r["L1"]) for r in rs),
            "L3": Counter(str(r["L3"]) for r in rs), "L2_direction": Counter(str(r["L2"]) for r in rs),
            "width_over_floor_median": median([r["width_over_floor"] for r in widths]),
            "width_over_3x_floor": f"{sum(1 for r in widths if r['width_over_floor'] > 3)}/{len(widths)}",
            "c34_width_over_floor_median": median([r["width_over_floor"] for r in c34 if r["width_over_floor"] is not None]),
            "L2_coverage_on_closed_cards": f"{len(hits)}/{len(covered)}",
            "L2_misses_above": sum(1 for r in covered if r["truth_delta"] > r["L2_interval"][1]),
            "L2_misses_below": sum(1 for r in covered if r["truth_delta"] < r["L2_interval"][0]),
            "L01_no": [f"{r['cell']}/{r['card']}" for r in rs if "no" in (str(r["L0"]), str(r["L1"]))],
            "probes_median": median([r["n_probes"] for r in rs]),
            "share_probe_changed_L01": round(sum(1 for r in rs if r["probe_changed_L01"]) / len(rs), 3),
            "c34_L2_cites_history": f"{sum(1 for r in c34 if r['L2_cites_history'])}/{len(c34)}",
            "suggests_checkpoint": f"{sum(1 for r in rs if r['suggests_checkpoint'])}/{len(rs)}",
            "c34_suggests_checkpoint": f"{sum(1 for r in c34 if r['suggests_checkpoint'])}/{len(c34)}",
            "turns_median": median([r["turns"] for r in rs if r.get("turns") is not None]),
            "cost_usd_median": median([r["cost_usd"] for r in rs if r.get("cost_usd") is not None]),
            "cost_usd_total": round(sum(r["cost_usd"] for r in rs if r.get("cost_usd")), 2),
            "change_types": Counter(t for r in rs for t in r["change_types"]),
        }
    out["_rejected"] = [f"{r['cell']}/{r['card']}" for r in rows if r["rejected"]]
    return out


def markdown(agg: dict) -> str:
    out = ["# Verdicts by skill hash", ""]
    for skill, a in agg.items():
        if not isinstance(a, dict):
            continue
        out.append(f"## skill `{skill}` — {a['n_verdicts']} verdicts ({a['n_c34']} C3/C4)")
        out.append("")
        for k, v in a.items():
            if k in ("n_verdicts", "n_c34"):
                continue
            out.append(f"- {k}: {dict(v) if isinstance(v, Counter) else v}")
        out.append("")
    out.append(f"rejected by the validator: {agg.get('_rejected') or 'none'}; "
               f"transcript-only (no verdict file): {agg.get('_transcript_only', 0)}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--inflight", action="store_true", help="also read <cell>.inflight snapshots (transcripts only)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for batch, cell, cell_dir, inflight in batch_cells(args.batches, inflight=args.inflight):
        cards = {} if inflight else load_cards(cell_dir)
        verdicts = {} if inflight else load_verdicts(cell_dir)
        seen = set(verdicts)
        for card_id, verdict in verdicts.items():
            rows.append(verdict_row(batch, cell, card_id, verdict, cards.get(card_id, {}), transcript_meta(cell_dir, card_id)))
        # verdicts only visible in the private transcript (in-flight cells, or a rejected file the card lacks)
        for path in sorted((cell_dir / "wma_private").glob("exp-*.transcript.jsonl.gz")):
            card_id = path.name.split(".")[0]
            if card_id in seen:
                continue
            rows.append({"batch": batch, "cell": cell, "card": card_id, "skill": None, "rejected": False,
                         "change_types": [], "family": None, "L0": None, "L1": None, "L2": None, "L2_interval": None,
                         "L3": None, "n": None, "floor": None, "width": None, "width_over_floor": None, "n_probes": 0,
                         "probe_kinds": [], "probe_changed": [], "probe_changed_L01": False, "cites_history": False,
                         "L2_cites_history": False, "suggests_checkpoint": False,
                         "truth_delta": None, "execution": None, "transcript_only": True,
                         **transcript_meta(cell_dir, card_id)})
    agg = aggregate([r for r in rows if not r.get("transcript_only")])
    agg["_transcript_only"] = sum(1 for r in rows if r.get("transcript_only"))
    write_outputs(Path(args.out), "verdicts", {"rows": rows, "aggregates": agg}, markdown(agg))
    for skill, a in agg.items():
        if isinstance(a, dict):
            print(f"{skill}: n={a['n_verdicts']} L3={dict(a['L3'])} coverage={a['L2_coverage_on_closed_cards']} "
                  f"width/floor med={a['width_over_floor_median']} cost med=${a['cost_usd_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
