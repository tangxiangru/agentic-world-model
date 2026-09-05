"""What the blocking lock cost and whether it held, per card, from harvested cells.

    python tools/wma-rca/gate.py results/ptb/<batch> [...] --out <dir>

Each `awm exp_protocol lock` call is paired with its own tool result: the wall time between them is
the gate's cost for that lock (the wait plus preflight), and the result says how it ended — a verdict
line, `review failed`, `recorded as a timeout`, or no world-model agent. A card answered through a
relock pays the gate again, which is why locks are counted per card, not per card-id.

Cells from before the verdict-in-lock protocol simply show locks with no gate lines; `state: none`.
Running cells are read from the operator's peeks with `inflight_gate.py` instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcalib import batch_cells, bash_command, cell_arm, median, parse_transcript, write_outputs  # noqa: E402

LOCK = re.compile(r"awm\s+exp_protocol\s+lock\b[^|;&]*?\b(exp-\d+)")
RELOCK = re.compile(r"--relock\b")
NO_WAIT = re.compile(r"--no-wma-wait\b")
ELAPSED = re.compile(r"(\d+\.\d) min elapsed")
VERDICT = re.compile(r"verdict: (L0_runs=[^\n]*)")
FAILED = re.compile(r"WMA review failed: ([^\n]*)")
TIMEOUT = re.compile(r"recorded as a timeout")
NOT_ATTACHED = re.compile(r"no world-model agent is attached")
LAUNCH = re.compile(r"(nohup|timeout \d+).*(train|sft|accelerate launch|torchrun)|--num-train-epochs", re.I)


def cell_rows(batch: str, cell: str, cell_dir: Path) -> list[dict]:
    calls, results = parse_transcript(cell_dir)
    if not calls:
        return []
    events = sorted([(t, "call", tool, text) for t, tool, text in calls if t]
                    + [(t, "result", "", text) for t, text in results if t], key=lambda e: e[0])
    rows: list[dict] = []
    arm = cell_arm(cell_dir)
    pending: tuple | None = None
    for when, kind, tool, text in events:
        if kind == "call":
            command = bash_command(text) if tool == "Bash" else ""
            hit = LOCK.search(command) if command else None
            if hit:
                pending = (when, hit.group(1), bool(RELOCK.search(command)), bool(NO_WAIT.search(command)))
            elif pending is None and command and LAUNCH.search(command) and rows:
                # time from the gate handing the launch back to the launch itself
                rows[-1].setdefault("launch_after_s", round((when - rows[-1]["returned_dt"]).total_seconds()))
            continue
        if pending is None:
            continue
        started, card, relock, no_wait = pending
        beats = ELAPSED.findall(text)
        verdict = VERDICT.search(text)
        state = ("delivered" if verdict else "failed" if FAILED.search(text)
                 else "timeout" if TIMEOUT.search(text) else "not_attached" if NOT_ATTACHED.search(text)
                 else "none")
        if state == "none" and not beats:
            # a lock that printed nothing about the WMA: pre-gate checkout, or a backgrounded lock
            # whose own result carries no verdict (the wait then lands in a later `tail`/`status` call)
            rows.append({"batch": batch, "cell": cell, "arm": arm, "card": card, "relock": relock,
                         "at": started.isoformat(), "at_dt": started, "returned_dt": when, "state": "none",
                         "gate_s": None, "heartbeats": 0, "verdict": None, "no_wma_wait": no_wait})
        else:
            rows.append({"batch": batch, "cell": cell, "arm": arm, "card": card, "relock": relock,
                         "at": started.isoformat(), "at_dt": started, "returned_dt": when, "state": state,
                         "gate_s": round((when - started).total_seconds()),
                         "heartbeats": len(beats), "verdict": verdict.group(1) if verdict else None,
                         "no_wma_wait": no_wait})
        pending = None
    return rows


def aggregate(rows: list[dict]) -> dict:
    gated = [r for r in rows if r["state"] in ("delivered", "failed", "timeout")]
    waits = [r["gate_s"] for r in gated if r["gate_s"] is not None]
    by_cell: dict[str, dict] = {}
    for r in rows:
        c = by_cell.setdefault(r["cell"], {"locks": 0, "relocks": 0, "gated": 0, "gate_h": 0.0,
                                           "cards": set(), "states": Counter()})
        c["locks"] += 1
        c["relocks"] += int(r["relock"])
        c["cards"].add(r["card"])
        c["states"][r["state"]] += 1
        if r["state"] in ("delivered", "failed", "timeout") and r["gate_s"]:
            c["gated"] += 1
            c["gate_h"] = round(c["gate_h"] + r["gate_s"] / 3600, 2)
    for c in by_cell.values():
        c["cards"] = len(c["cards"])
        c["states"] = dict(c["states"])
    launched = [r for r in rows if r.get("launch_after_s") is not None]
    return {"locks": len(rows), "relocks": sum(int(r["relock"]) for r in rows),
            "states": dict(Counter(r["state"] for r in rows)),
            "gate_s_median": median(waits), "gate_h_total": round(sum(waits) / 3600, 2),
            "delivered_before_a_launch": len([r for r in launched if r["state"] == "delivered"]),
            "no_wma_wait": sum(int(r["no_wma_wait"]) for r in rows), "by_cell": by_cell}


def markdown(rows: list[dict], agg: dict) -> str:
    out = ["# The lock gate: cost and outcome", "",
           f"locks {agg['locks']} ({agg['relocks']} of them relocks); states {agg['states']}; "
           f"median gate {agg['gate_s_median']} s; {agg['gate_h_total']} h of gate in total; "
           f"`--no-wma-wait` used {agg['no_wma_wait']}×", "",
           "| cell | card | relock | at | state | gate s | beats | verdict |", "|---|---|---|---|---|---:|---:|---|"]
    for r in rows:
        out.append(f"| {r['cell']} | {r['card']} | {'yes' if r['relock'] else ''} | {r['at'][11:16]} | "
                   f"{r['state']} | {r['gate_s'] if r['gate_s'] is not None else ''} | {r['heartbeats']} | "
                   f"{(r['verdict'] or '')[:70]} |")
    out += ["", "## Per cell", "", "| cell | cards | locks | relocks | gated | gate h | states |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for cell, c in sorted(agg["by_cell"].items()):
        out.append(f"| {cell} | {c['cards']} | {c['locks']} | {c['relocks']} | {c['gated']} | "
                   f"{c['gate_h']} | {c['states']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for batch, cell, cell_dir, _inflight in batch_cells(args.batches):
        rows.extend(cell_rows(batch, cell, cell_dir))
    agg = aggregate(rows)
    for r in rows:
        r.pop("at_dt", None)
        r.pop("returned_dt", None)
    write_outputs(Path(args.out), "gate", {"rows": rows, "aggregates": agg}, markdown(rows, agg))
    print(markdown(rows, agg).splitlines()[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
