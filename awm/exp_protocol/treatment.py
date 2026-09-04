"""Frozen study modes, independent of whether a WMA sidecar is attached."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODES = ("single", "multi-self", "multi-joint")
DEFAULT = "multi-joint"


def describe(mode: str, *, explicit: bool) -> dict:
    if mode not in MODES:
        raise ValueError(f"decision_mode must be one of {MODES}")
    value = {"version": 1, "decision_mode": mode, "explicit": explicit}
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def identity(session: Path) -> dict:
    path = Path(session) / "awm_sandbox.json"
    if path.is_symlink():
        raise ValueError("study configuration must not be a symlink")
    if not path.exists():
        return describe(DEFAULT, explicit=False)
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError("study configuration must be an object")  # noqa: TRY004 - public JSON validation
    if "decision_mode" not in record:
        return describe(DEFAULT, explicit=False)
    result = describe(record["decision_mode"], explicit=True)
    if record.get("decision_mode_sha256") != result["sha256"]:
        raise ValueError("study mode configuration hash differs; restore the frozen setup")
    return result


def mode(session: Path) -> str:
    return identity(session)["decision_mode"]
