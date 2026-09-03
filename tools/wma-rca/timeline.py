"""The lock → request → verdict → launch → read → close timeline of one cell, for hand-reading.

    .venv/bin/python tools/wma-rca/timeline.py results/ptb/<batch>/<cell>

Prints one line per event with the card id; verdict lines carry the four levels, never the text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcalib import (g, hhmm, launch_time, load_cards, load_locks, load_queue, load_verdicts,  # noqa: E402
                    parse_transcript, read_time, ts)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cell_dir = Path(sys.argv[1])
    calls, results = parse_transcript(cell_dir)
    cards, locks, verdicts = load_cards(cell_dir), load_locks(cell_dir), load_verdicts(cell_dir)
    requests, responses = load_queue(cell_dir)
    events: list[tuple] = []
    for card_id, card in cards.items():
        lock = locks.get(card_id) or {}
        locked = ts(lock.get("locked_at"))
        if locked:
            gate = lock.get("wma") or {}
            events.append((locked, card_id, "lock", f"family={g(card, 'setup.method.family')} planned_h={g(card, 'setup.budget.planned_h')}"
                           + (f" gate={gate.get('state')} waited={gate.get('waited_s')}s" if gate else "")))
        if requests.get(card_id):
            events.append((ts(requests[card_id]), card_id, "request", ""))
        resp = responses.get(card_id)
        if resp and resp.get("completed_at"):
            events.append((ts(resp["completed_at"]), card_id, "response", f"{resp.get('state')} {resp.get('error') or ''}"[:100]))
        verdict = verdicts.get(card_id)
        issued = ts(verdict.get("issued_at")) if verdict else None
        if issued:
            lv = verdict.get("levels") or {}
            events.append((issued, card_id, "verdict" + (" (rejected)" if verdict.get("_rejected") else ""),
                           f"L0={g(lv, 'L0_runs.answer')} L1={g(lv, 'L1_valid.answer')} L2={g(lv, 'L2_effect.direction')} "
                           f"{g(lv, 'L2_effect.interval')} L3={g(lv, 'L3_worth_now.answer')}"))
        launch = launch_time(card_id, card, locked, calls)
        if launch:
            events.append((launch, card_id, "launch", ""))
        read = read_time(card_id, issued, calls, results)
        if read:
            events.append((read, card_id, "read verdict", f"{round((read - issued).total_seconds() / 60)} min after issue"))
        if g(card, "result.wall_h") and launch:
            from datetime import timedelta
            events.append((launch + timedelta(hours=float(g(card, "result.wall_h"))), card_id, "run ends (lock+wall_h)",
                           f"delta={(g(card, 'result.measurements') or [{}])[0].get('delta_vs_comparator')} decision={g(card, 'conclusion.decision')}"))
    for when, card_id, kind, note in sorted(e for e in events if e[0]):
        print(f"{hhmm(when)}  {card_id:7s} {kind:24s} {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
