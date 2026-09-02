"""Pin the pre-launch sections, the training script, and the data before the run; re-check at close.

This is the whole of the "written before the run cannot change after it
without leaving a trace" rule: one JSON file beside the card. There is no
state, no daemon, nothing to resume. The lock lives in the scientist's own
directory, so it is a trace, not a barrier: a mismatch at close is
reported, a deleted lock fails close loudly, and a second lock is refused
unless a reason is given and the previous hash is kept in the file.

Not covered, on purpose and documented in doc/spec/2026-09-01-exp-protocol-card-v2.md:
files the argv names other than ``setup.command.script`` (a config YAML, for
instance), and the card's own ``card_id`` / ``created_at``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Report, get, now, plan_hash, sha256_file

LOCK_SCHEMA = "awm-exp-lock-v1"


class LockExists(ValueError):
    """A lock is already there; re-locking needs a reason so the first hash survives."""


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


def _data_entries(card: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for d in get(card, "setup.data") or []:
        if not isinstance(d, dict) or not d.get("path"):
            continue
        p = Path(str(d["path"]))
        if p.is_file():
            out.append({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
        else:
            out.append({"path": str(p), "sha256": "", "bytes": None})
    return out


def write_lock(card_path: Path, card: dict[str, Any], preflight_summary: dict[str, Any], *,
               relock_reason: str | None = None,
               overrides: dict[str, str] | None = None) -> dict[str, Any]:
    previous = read_lock(card_path)
    history: list[dict[str, Any]] = []
    if previous is not None:
        if not relock_reason:
            raise LockExists(f"{lock_path(card_path)} exists; re-lock only with a reason")
        history = list(previous.get("relocked_from") or [])
        history.append({"plan_sha256": previous.get("plan_sha256"), "locked_at": previous.get("locked_at"),
                        "reason": relock_reason})
    info = {
        "schema_version": LOCK_SCHEMA,
        "card_id": card.get("card_id"),
        "locked_at": now(),
        "plan_sha256": plan_hash(card),
        "script": _script_entry(card),
        "data": _data_entries(card),
        "preflight": dict(preflight_summary),
        "overrides": dict(overrides or {}),
        "relocked_from": history,
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
    for i, entry in enumerate(info.get("data") or []):
        if not entry.get("sha256"):
            continue
        p = Path(entry["path"])
        if not p.is_file():
            r.warn(f"setup.data[{i}].path", f"{p} no longer exists; the locked hash cannot be re-checked")
        elif sha256_file(p) != entry["sha256"]:
            r.error(f"setup.data[{i}].path", f"{p} changed after the lock; the run did not train on the data the card names")
    return r
