"""Public, model-free candidate and decision records shared with the private WMA.

The scientist owns proposals and choices; comparison never grants launch permission.
Immutable request/action files are deliberately named without ``exp-*.verdict`` so
the existing frozen ledger cannot accidentally count their archived copies twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROPOSAL_SCHEMA = "awm-wma-proposal-set-v1"
COMPARISON_SCHEMA = "awm-wma-comparison-v1"
COMPARE_REQUEST_SCHEMA = "awm-wma-comparison-request-v1"
ACTION_SCHEMA = "awm-wma-action-v1"
DECISION_ID = re.compile(r"^decision-[0-9]+$")
CANDIDATE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
CARD_ID = re.compile(r"^exp-[0-9]+$")
ACTIONS = ("proceed", "repair", "probe", "decline", "abandon")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def proposal_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 100_000:
        raise ValueError(f"{name} must be a nonempty string (at most 100000 characters)")


def _texts(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError(f"{name} must contain 1–100 evidence strings")
    for text in value:
        _text(text, name)


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def validate_proposal(proposal: dict) -> None:
    if not isinstance(proposal, dict) or proposal.get("schema_version") != PROPOSAL_SCHEMA:
        raise ValueError(f"expected schema_version {PROPOSAL_SCHEMA}")
    if not DECISION_ID.fullmatch(str(proposal.get("decision_id", ""))):
        raise ValueError("decision_id must be decision-NN")
    context = proposal.get("situation")
    if not isinstance(context, dict):
        raise ValueError("situation must be an object")  # noqa: TRY004 - public JSON validation contract
    _number(context.get("remaining_h"), "situation.remaining_h")
    _text(context.get("incumbent"), "situation.incumbent")
    _texts(context.get("evidence"), "situation.evidence")
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("provide 1–3 real candidate briefs")
    ids = []
    for c in candidates:
        if not isinstance(c, dict) or not CANDIDATE_ID.fullmatch(str(c.get("candidate_id", ""))):
            raise ValueError("each candidate needs a safe candidate_id")
        ids.append(c["candidate_id"])
        for name in ("hypothesis", "parent_checkpoint", "change", "cost_basis", "uncertainty", "decision_test"):
            _text(c.get(name), f"{c['candidate_id']}.{name}")
        for name in ("train_h", "eval_h"):
            _number(c.get(name), f"{c['candidate_id']}.{name}")
        _texts(c.get("evidence"), f"{c['candidate_id']}.evidence")
        if any(k in c for k in ("result", "outcome", "measurements", "conclusion")):
            raise ValueError("candidate briefs must describe unobserved proposals, not their outcomes")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be distinct")
    if proposal.get("scientist_preference") not in ids:
        raise ValueError("scientist_preference must name a candidate before comparison")
    if len(ids) == 1:
        _text(proposal.get("singleton_reason"), "singleton_reason")
    if len(json.dumps(proposal, allow_nan=False).encode()) > 1_000_000:
        raise ValueError("proposal set exceeds 1 MB")


def validate_comparison(value: dict, proposal: dict) -> None:
    validate_proposal(proposal)
    if not isinstance(value, dict) or value.get("schema_version") != COMPARISON_SCHEMA:
        raise ValueError(f"expected schema_version {COMPARISON_SCHEMA}")
    if value.get("decision_id") != proposal["decision_id"] or value.get("proposal_sha256") != proposal_sha(proposal):
        raise ValueError("comparison does not match the frozen proposal")
    ids = {c["candidate_id"] for c in proposal["candidates"]}
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or any(not isinstance(x, str) for x in ranking) or len(ranking) != len(ids) or set(ranking) != ids:
        raise ValueError("ranking must include every candidate exactly once")
    assessments = value.get("candidate_assessments")
    if not isinstance(assessments, list) or len(assessments) != len(ids):
        raise ValueError("candidate_assessments must cover every candidate")
    seen = set()
    for c in assessments:
        if not isinstance(c, dict) or not isinstance(c.get("candidate_id"), str) or c.get("candidate_id") not in ids or c["candidate_id"] in seen:
            raise ValueError("invalid or duplicate candidate assessment")
        seen.add(c["candidate_id"])
        if c.get("feasibility") not in ("ready", "needs_check", "blocked"):
            raise ValueError("feasibility must be ready, needs_check, or blocked")
        for key in ("expected_effect", "opportunity_cost", "uncertainty"):
            _text(c.get(key), key)
    reasons = value.get("comparisons")
    if not isinstance(reasons, list) or len(reasons) != len(ids) - 1:
        raise ValueError("explain every adjacent pair in the proposed ranking")
    for i, r in enumerate(reasons):
        if not isinstance(r, dict) or (r.get("preferred"), r.get("alternative")) != (ranking[i], ranking[i + 1]):
            raise ValueError("comparison reasons must follow adjacent ranking pairs")
        _text(r.get("reason"), "comparison.reason")
        _text(r.get("uncertainty"), "comparison.uncertainty")
        _texts(r.get("evidence"), "comparison.evidence")
    suggestions = value.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("suggestions must be a list")  # noqa: TRY004 - public JSON validation contract
    for s in suggestions:
        if not isinstance(s, dict) or not isinstance(s.get("candidate_id"), str) or s.get("candidate_id") not in ids:
            raise ValueError("suggestion must name an existing candidate")
        if s.get("kind") not in ("required_fix", "optional_probe", "prefer_alternative"):
            raise ValueError("invalid suggestion kind")
        for key in ("action", "evidence_scope", "decision_if_observed"):
            _text(s.get(key), f"suggestion.{key}")


def safe_path(session: Path, relative: str) -> Path:
    root = Path(session).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"record path leaves the session: {relative}")
    return path


def write_once(path: Path, value: dict) -> None:
    """Never overwrite a request, result revision, or scientist action."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # link publishes the completed bytes atomically and refuses an existing name.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_records(paths):
    """A malformed historical file must not poison subsequent valid retries."""
    for path in paths:
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or not isinstance(value.get("created_at"), str):
                raise ValueError("record has no creation time")  # noqa: TRY004 - malformed JSON record
            yield path, value
        except (OSError, ValueError) as exc:
            warnings.warn(f"unreadable audit record retained at {path}: {exc}", RuntimeWarning, stacklevel=2)


def proposal_path(session: Path, decision_id: str) -> Path:
    if not DECISION_ID.fullmatch(decision_id):
        raise ValueError("decision_id must be decision-NN")
    return safe_path(session, f"memory/decisions/{decision_id}.proposal.json")


def create_proposal(session: Path) -> Path:
    from . import treatment
    if treatment.mode(session) == "single":
        raise ValueError("single mode does not create a candidate pool")
    directory = safe_path(session, "memory/decisions")
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while proposal_path(session, f"decision-{number:02d}").exists():
        number += 1
    did = f"decision-{number:02d}"
    value = {"schema_version": PROPOSAL_SCHEMA, "decision_id": did,
             "situation": {"remaining_h": 0, "incumbent": "", "evidence": []},
             "scientist_preference": "A", "candidates": [
                 {"candidate_id": cid, "hypothesis": "", "parent_checkpoint": "", "change": "",
                  "train_h": 0, "eval_h": 0, "cost_basis": "", "evidence": [],
                  "uncertainty": "", "decision_test": ""} for cid in ("A", "B")]}
    path = proposal_path(session, did)
    write_once(path, value)
    return path


def load_proposal(session: Path, decision_id: str) -> dict:
    value = json.loads(proposal_path(session, decision_id).read_text())
    validate_proposal(value)
    if value["decision_id"] != decision_id:
        raise ValueError("decision ID differs from the proposal filename")
    return value


def card_fingerprint(session: Path, card_id: str) -> dict:
    from . import schema, treatment

    if not CARD_ID.fullmatch(card_id):
        raise ValueError("card_id must be exp-NN")
    path = safe_path(session, f"memory/cards/{card_id}.yaml")
    card = schema.load_card(path)
    lock_path = path.with_suffix(".lock.json")
    locked = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    fields = {k: locked.get(k) for k in ("lock_id", "plan_sha256", "script", "data", "configs")}
    # Recompute every file that the lock actually pins. Do not trust stale stored hashes.
    files = []
    entries = [fields.get("script"), *(fields.get("data") or []), *(fields.get("configs") or [])]
    for e in entries:
        if not isinstance(e, dict) or not e.get("path"):
            continue
        logical = Path(e["path"])
        if logical.is_absolute() and logical.is_relative_to("/home/ben/task"):
            # PTB exposes the same tree at /home/ben/task in the scientist and
            # /session in the sidecar. Preserve logical identity; hash local bytes.
            p = Path(session) / logical.relative_to("/home/ben/task")
        elif not logical.is_absolute():
            p = Path(session) / logical
        else:
            p = logical
        files.append({"path": str(logical), "sha256": schema.sha256_file(p) if p.is_file() else None})
    return {"plan_sha256": schema.plan_hash(card), "lock_sha256": proposal_sha(fields),
            "files": files, "treatment": treatment.identity(session)}


def append_action(session: Path, card_id: str, action: str, reason: str, *,
                  suggestion: str | None = None, evidence: list[str] | None = None) -> Path:
    if action not in ACTIONS:
        raise ValueError("action must be one of " + ", ".join(ACTIONS))
    _text(reason, "reason")
    fingerprint = card_fingerprint(session, card_id)
    lp = safe_path(session, f"memory/cards/{card_id}.lock.json")
    if not lp.is_file():
        raise ValueError("lock and await review before recording an action")
    lock = json.loads(lp.read_text())
    wma = lock.get("wma") or {}
    if action == "proceed" and wma.get("state") not in ("delivered", "not_attached", "failed", "timeout", "skipped"):
        raise ValueError("the lock has not returned from WMA review")
    value = {"schema_version": ACTION_SCHEMA, "card_id": card_id, "created_at": now(),
             "action_id": uuid.uuid4().hex, "locked_at": lock.get("locked_at"),
             "action": action, "reason": reason, "suggestion": suggestion,
             "evidence": evidence or [], "fingerprint": fingerprint,
             "request_id": wma.get("request_id"), "review_state": wma.get("state"),
             "review_fingerprint": wma.get("fingerprint")}
    path = safe_path(session, f".wma/actions/{card_id}/{value['action_id']}.json")
    write_once(path, value)
    return path


def latest_action(session: Path, card_id: str) -> dict | None:
    if not CARD_ID.fullmatch(card_id):
        raise ValueError("card_id must be exp-NN")
    directory = safe_path(session, f".wma/actions/{card_id}")
    values = [v for _, v in read_records(directory.glob("*.json"))
              if v.get("schema_version") == ACTION_SCHEMA and v.get("card_id") == card_id]
    return max(values, key=lambda v: (v["created_at"], v.get("action", ""))) if values else None
