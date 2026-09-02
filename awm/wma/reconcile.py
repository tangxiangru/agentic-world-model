"""The verdict meets the outcome.

Truth comes from the card itself once it is closed, or — in replay — from a
truth card kept outside the session directory so the agent could not have
read it. Reconciling appends ``actual`` and ``scored`` to the verdict file;
a verdict with no truth yet scores as unscorable on every level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from awm.exp_protocol.schema import load_card, migrate_v1, now

from . import schema


class ReconcileError(ValueError):
    pass


def reconcile(card_path: Path, truth_path: Path | None = None) -> dict[str, Any]:
    card_path = Path(card_path)
    vpath = schema.verdict_path(card_path)
    if not vpath.is_file():
        raise ReconcileError(f"no verdict at {vpath}; nothing to reconcile")
    verdict = schema.load_verdict(vpath)
    source = Path(truth_path) if truth_path else card_path
    truth = schema.truth_from_card(migrate_v1(load_card(source)))
    verdict["reconciled_at"] = now()
    verdict["truth_source"] = str(source)
    verdict["actual"] = {k: truth[k] for k in ("execution", "output_checkpoint", "decision", "wall_h", "delta")}
    verdict["actual"]["n_measurements"] = len(truth["measurements"])
    verdict["scored"] = schema.score(verdict, truth)
    schema.dump_verdict(vpath, verdict)
    return verdict
