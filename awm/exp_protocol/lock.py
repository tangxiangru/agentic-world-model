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

import contextlib
import fcntl
import json
import os
import tempfile
import uuid
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


@contextlib.contextmanager
def _card_guard(card_path: Path):
    """Serialize lock revisions and conditional annotations across processes."""
    path = lock_path(card_path).with_suffix(".guard")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _publish(card_path: Path, value: dict) -> None:
    path = lock_path(card_path)
    fd, temporary = tempfile.mkstemp(prefix=".lock-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _script_entry(card: dict[str, Any]) -> dict[str, str] | None:
    script = get(card, "setup.command.script")
    if not script:
        return None
    p = Path(script)
    if not p.is_absolute():
        p = Path(get(card, "setup.command.cwd") or ".") / p
    if not p.is_file():
        return {"path": str(script), "sha256": ""}
    return {"path": str(p.absolute()), "sha256": sha256_file(p)}


def _data_entries(card: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for d in get(card, "setup.data") or []:
        if not isinstance(d, dict) or not d.get("path"):
            continue
        p = Path(str(d["path"]))
        if not p.is_absolute():
            p = Path(get(card, "setup.command.cwd") or ".") / p
        if p.is_file():
            out.append({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
        else:
            out.append({"path": str(p), "sha256": "", "bytes": None})
    return out


def _config_entries(card: dict[str, Any]) -> list[dict[str, str]]:
    configs = get(card, "setup.command.configs") or []
    if not isinstance(configs, list) or any(not isinstance(p, str) or not p for p in configs):
        raise ValueError("setup.command.configs must be a list of config file paths")
    entries = []
    for name in configs:
        p = Path(name)
        if not p.is_absolute():
            p = Path(get(card, "setup.command.cwd") or ".") / p
        if not p.is_file():
            raise ValueError(f"declared config does not exist: {p}")
        entries.append({"path": str(p.absolute()), "sha256": sha256_file(p)})
    return entries


def plan_inputs(card: dict[str, Any]) -> dict[str, Any]:
    """Current bytes named by the formal plan, independent of any prior lock."""
    return {"script": _script_entry(card), "data": _data_entries(card), "configs": _config_entries(card)}


def write_lock(card_path: Path, card: dict[str, Any], preflight_summary: dict[str, Any], *,
               relock_reason: str | None = None,
               overrides: dict[str, str] | None = None) -> dict[str, Any]:
    with _card_guard(card_path):
        return _write_lock(card_path, card, preflight_summary, relock_reason=relock_reason, overrides=overrides)


def _write_lock(card_path: Path, card: dict[str, Any], preflight_summary: dict[str, Any], *,
                relock_reason: str | None, overrides: dict[str, str] | None) -> dict[str, Any]:
    previous = read_lock(card_path)
    history: list[dict[str, Any]] = []
    if previous is not None:
        if not relock_reason:
            raise LockExists(f"{lock_path(card_path)} exists; re-lock only with a reason")
        history = list(previous.get("relocked_from") or [])
        earlier = {"plan_sha256": previous.get("plan_sha256"), "locked_at": previous.get("locked_at"),
                   "lock_id": previous.get("lock_id"),
                   "overrides": dict(previous.get("overrides") or {}), "reason": relock_reason}
        if previous.get("wma") is not None:
            # each lock of a card asks the world-model agent again and waits for the answer; without
            # this the new lock would drop the earlier wait and the card's gate cost would be
            # unrecoverable from the artefacts (2026-09-03: relocks are how preconditions are answered)
            earlier["wma"] = previous["wma"]
        history.append(earlier)
    info = {
        "schema_version": LOCK_SCHEMA,
        "card_id": card.get("card_id"),
        "lock_id": uuid.uuid4().hex,
        "locked_at": now(),
        "plan_sha256": plan_hash(card),
        **plan_inputs(card),
        "preflight": dict(preflight_summary),
        "overrides": dict(overrides or {}),
        "relocked_from": history,
    }
    _publish(card_path, info)
    return info


def annotate_lock(card_path: Path, key: str, value: Any, *, expected_lock_id: str | None = None) -> dict[str, Any]:
    """Add one top-level field to an existing lock without touching what the lock pins.
    `verify_lock` compares plan/script/data hashes only, so an annotation never invalidates a lock."""
    with _card_guard(card_path):
        info = read_lock(card_path)
        if info is None:
            raise LockExists(f"{lock_path(card_path)} does not exist; nothing to annotate")
        if expected_lock_id is not None and info.get("lock_id") != expected_lock_id:
            raise LockExists("lock revision changed while waiting for WMA")
        info[key] = value
        _publish(card_path, info)
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
    for i, entry in enumerate([*(info.get("data") or []), *(info.get("configs") or [])]):
        if not entry.get("sha256"):
            continue
        p = Path(entry["path"])
        if not p.is_file():
            r.warn(f"setup.data[{i}].path", f"{p} no longer exists; the locked hash cannot be re-checked")
        elif sha256_file(p) != entry["sha256"]:
            r.error(f"setup.data[{i}].path", f"{p} changed after the lock; the run did not train on the data the card names")
    return r
