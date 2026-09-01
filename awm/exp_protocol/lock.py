"""Pin the pre-launch sections and the training script before the run; re-check at close.

This is the whole of the "written before the run cannot change after it"
rule: one JSON file beside the card. There is no state, no daemon, nothing
to resume. A lock that does not match at close is reported, not repaired —
a material change is a new card.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Report, get, now, plan_hash, sha256_file

LOCK_SCHEMA = "awm-exp-lock-v1"


def lock_path(card_path: Path) -> Path:
    card_path = Path(card_path)
    return card_path.with_name(card_path.stem + ".lock.json")


def preflight_path(card_path: Path) -> Path:
    card_path = Path(card_path)
    return card_path.with_name(card_path.stem + ".preflight.json")


def read_lock(card_path: Path) -> dict[str, Any] | None:
    p = lock_path(card_path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _script_entry(card: dict[str, Any]) -> dict[str, str] | None:
    script = get(card, "setup.command.script")
    if not script:
        return None
    p = Path(script)
    if not p.is_file():
        return {"path": str(script), "sha256": ""}
    return {"path": str(script), "sha256": sha256_file(p)}


def write_lock(card_path: Path, card: dict[str, Any], preflight_summary: dict[str, Any]) -> dict[str, Any]:
    info = {
        "schema_version": LOCK_SCHEMA,
        "card_id": card.get("card_id"),
        "locked_at": now(),
        "plan_sha256": plan_hash(card),
        "script": _script_entry(card),
        "preflight": dict(preflight_summary),
    }
    lock_path(card_path).write_text(json.dumps(info, indent=2) + "\n")
    return info


def verify_lock(card_path: Path, card: dict[str, Any]) -> Report:
    r = Report()
    info = read_lock(card_path)
    if info is None:
        r.error("lock", f"no lock file at {lock_path(card_path)}; was this card locked before the run?")
        return r
    if info.get("plan_sha256") != plan_hash(card):
        r.error("plan", "sections 0-4 differ from what was locked; a material change is a new card")
    locked_script = info.get("script")
    if locked_script and locked_script.get("sha256"):
        p = Path(locked_script["path"])
        if not p.is_file():
            r.warn("setup.command.script", f"{p} no longer exists; the locked hash cannot be re-checked")
        elif sha256_file(p) != locked_script["sha256"]:
            r.error("setup.command.script", f"{p} changed after the lock; the card names a script that did not run")
    return r
