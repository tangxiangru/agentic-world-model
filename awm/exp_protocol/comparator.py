"""Opt-in comparator lifecycle; the shared receipt checker is stdlib-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import yaml

from .schema import get, now


def enabled(card: dict) -> bool:
    return get(card, "evaluation.comparator.defer_validation") is True


def declaration(card: dict) -> dict:
    return {"path": get(card, "evaluation.comparator.path"),
            "n": get(card, "evaluation.protocol.n"),
            "metric": get(card, "hypothesis.expected_effect.metric")}


@lru_cache(maxsize=None)
def _load(path: str):
    spec = importlib.util.spec_from_file_location("_awm_comparator_receipt", path)
    if spec is None or spec.loader is None:
        raise OSError(f"cannot load comparator receipt helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper():
    from .preflight import skill_dir
    return _load(str(skill_dir() / "hooks" / "comparator_receipt.py"))


def check_output(card: dict, *, allow_missing: bool = False) -> dict:
    d = declaration(card)
    try:
        return helper().inspect_output(d["path"], d["n"], d["metric"], allow_missing=allow_missing)
    except (OSError, ValueError, TypeError, ImportError, SyntaxError):
        return {"status": "fail", "detail": "cannot validate deferred comparator with the installed helper"}


def completion_state(card_path: Path, card: dict, info: dict | None) -> dict:
    required = enabled(card) or "deferred_comparator" in (info or {})
    if not required:
        return {"required": False, "valid": True, "outcome": "legacy"}
    try:
        state = helper().completion_state(card_path, info or {})
    except (OSError, ValueError, TypeError, ImportError, SyntaxError):
        state = {"valid": False, "outcome": "unverified", "detail": "cannot validate comparator completion receipt"}
    return {"required": True, **state}


def write_completion(card_path: Path, card: dict, info: dict) -> dict:
    from .lock import lock_path, read_lock, verify_lock
    from .schema import validate_plan, validate_result

    for report in (validate_plan(card), validate_result(card), verify_lock(card_path, card)):
        if not report.ok:
            raise ValueError(report.render())
    module = helper()
    raw = card_path.read_bytes()
    if yaml.safe_load(raw) != card:
        raise ValueError("card changed while close was validating it")
    marker = info.get("deferred_comparator")
    if not enabled(card) or not isinstance(marker, dict):
        raise ValueError("deferred comparator is not bound to this lock")
    if any(marker.get(k) != v for k, v in declaration(card).items()):
        raise ValueError("deferred declaration differs from its lock")
    execution = get(card, "result.execution")
    observation = None
    if execution == "completed":
        observation = check_output(card)
        if observation["status"] != "pass":
            raise ValueError(observation["detail"])
        outcome = "verified"
    elif execution in module.FAILURE_OUTCOMES:
        outcome = execution
    else:
        raise ValueError("deferred experiment has no completed/failed execution classification")
    proof = {"schema_version": module.RECEIPT_SCHEMA, "card_id": card["card_id"],
             "verified_at": now(), "locked_at": info["locked_at"],
             "plan_sha256": info["plan_sha256"], "card_sha256": hashlib.sha256(raw).hexdigest(),
             "declaration": marker, "outcome": outcome, "observation": observation}
    target = module.receipt_path(card_path)
    temporary = target.with_name(target.name + ".tmp")
    encoded = (json.dumps(proof, indent=2, allow_nan=False) + "\n").encode()
    temporary.write_bytes(encoded)
    temporary.replace(target)
    if read_lock(card_path) != info:
        raise ValueError("lock changed while comparator closure was being recorded")
    sealed = {**info, "deferred_close_sha256": hashlib.sha256(encoded).hexdigest()}
    target_lock = lock_path(card_path)
    temporary_lock = target_lock.with_name(target_lock.name + ".tmp")
    temporary_lock.write_text(json.dumps(sealed, indent=2) + "\n")
    temporary_lock.replace(target_lock)
    return proof
