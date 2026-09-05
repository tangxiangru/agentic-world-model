"""Execute a recorded plan only after its current review and scientist decision.

This wrapper records process execution, not scientific completion. A detached
command's exit does not certify that its background training finished.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import decisions, lock, schema, treatment


def _choice(session: Path, card_id: str, card: dict) -> dict:
    records = [choice for _, choice in decisions.read_records(
        decisions.safe_path(session, "memory/decisions").glob("decision-*/choices/*.json"))]
    choices = [choice for choice in records if choice.get("card_id") == card_id]
    if not choices:
        raise ValueError("training needs a recorded candidate choice: awm wma choose ... --card " + card_id)
    choice = max(choices, key=lambda c: c["created_at"])
    current_choice = max((c for c in records if c.get("decision_id") == choice["decision_id"]),
                         key=lambda c: c["created_at"])
    if current_choice != choice:
        raise ValueError("the choice was superseded or declined; bind a current choice before launching")
    configured = treatment.identity(session)
    if choice.get("treatment", treatment.describe(treatment.DEFAULT, explicit=False)) != configured:
        raise ValueError("study mode changed after the candidate choice")
    if (configured["decision_mode"] == "multi-self") != (choice.get("comparison_state") == "not_requested"):
        raise ValueError("choice comparison state does not match this study mode")
    proposal = decisions.load_proposal(session, choice["decision_id"])
    if choice.get("proposal_sha256") != decisions.proposal_sha(proposal):
        raise ValueError("candidate set changed after the choice; compare and choose again")
    if choice.get("selected") not in [c["candidate_id"] for c in proposal["candidates"]]:
        raise ValueError("the recorded choice does not select a real candidate")
    if choice.get("bound_plan_sha256") != schema.plan_hash(card) or choice.get("bound_inputs") != lock.plan_inputs(card):
        raise ValueError("formal plan or inputs differ from the choice binding; record choose --card again after review")
    return choice


def execute(session: Path, card_id: str) -> int:
    session = Path(session).resolve()
    if not decisions.CARD_ID.fullmatch(card_id):
        raise ValueError("card_id must be exp-NN")
    path = decisions.safe_path(session, f"memory/cards/{card_id}.yaml")
    card = schema.load_card(path)
    if schema.get(card, "result.execution") not in (None, "", "not_run"):
        raise ValueError("this card already has an outcome; use a new pre-launch plan")
    validation = schema.validate_plan(card, session)
    integrity = lock.verify_lock(path, card)
    if not validation.ok or not integrity.ok:
        raise ValueError(validation.render() + "\n" + integrity.render())
    info = lock.read_lock(path) or {}
    review = info.get("wma") or {}
    if review.get("state") not in ("delivered", "not_attached", "failed", "timeout", "skipped"):
        raise ValueError("lock has not returned from review; wait before launching")
    current = decisions.card_fingerprint(session, card_id)
    if any(f["sha256"] is None for f in current["files"]):
        raise ValueError("a pinned input is now missing; repair and re-lock")
    if review.get("fingerprint") is not None and review["fingerprint"] != current:
        raise ValueError("plan or pinned inputs changed after review; re-lock and decide again")
    if review["state"] == "delivered":
        vp = path.with_suffix(".verdict.json")
        verdict = json.loads(vp.read_text())
        if verdict.get("request_id") != review.get("request_id") or verdict.get("review_fingerprint") != current:
            raise ValueError("the delivered verdict does not belong to this request and plan")
    action = decisions.latest_action(session, card_id)
    if not action or action.get("action") != "proceed":
        raise ValueError("record a proceed decision with awm wma act before launching")
    if (action.get("fingerprint") != current or action.get("request_id") != review.get("request_id")
            or action.get("locked_at") != info.get("locked_at")):
        raise ValueError("the proceed decision predates this lock or plan; record a current decision")
    action_id = action.get("action_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", action_id):
        raise ValueError("invalid proceed action ID")
    choice = None
    if (schema.get(card, "setup.method.family") in ("sft", "rft", "dpo", "grpo", "distill")
            and treatment.mode(session) != "single"):
        choice = _choice(session, card_id, card)
    argv = schema.get(card, "setup.command.argv")
    cwd = Path(schema.get(card, "setup.command.cwd"))
    if not cwd.is_absolute():
        cwd = session / cwd
    additions = schema.get(card, "setup.command.env") or {}
    if not isinstance(additions, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in additions.items()):
        raise ValueError("setup.command.env must map names to string values")
    if not cwd.is_dir():
        raise ValueError(f"command cwd does not exist: {cwd}")
    base = decisions.safe_path(session, f".wma/launches/{card_id}")
    start = {"schema_version": "awm-exp-launch-v1", "card_id": card_id,
             "action_id": action_id, "request_id": review.get("request_id"),
             "review_state": review["state"], "fingerprint": current,
             "treatment": treatment.identity(session),
             "choice": choice, "argv": argv, "cwd": str(cwd), "env_keys": sorted(additions),
             "started_at": decisions.now()}
    # One explicit proceed decision permits one launch, even under concurrent invocations.
    try:
        decisions.write_once(base / f"{action_id}.start.json", start)
    except FileExistsError as exc:
        raise ValueError("this proceed decision already launched; record a new action for a retry") from exc
    began = time.monotonic()
    error = None
    try:
        code = subprocess.run(argv, cwd=cwd, env={**os.environ, **additions}, check=False).returncode
    except OSError as exc:
        code, error = 127, str(exc)
    except KeyboardInterrupt:
        code, error = 130, "interrupted"
    decisions.write_once(base / f"{action_id}.exit.json", {
        "schema_version": "awm-exp-launch-exit-v1", "card_id": card_id,
        "action_id": action_id, "finished_at": decisions.now(),
        "command_exit_code": code, "wall_s": round(time.monotonic() - began, 6),
        "error": error, "scientific_completion": "not_assessed",
    })
    return code if code >= 0 else 128 - code
