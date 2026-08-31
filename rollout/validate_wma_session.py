#!/usr/bin/env python3
"""Attest that a labelled WMA cell actually used and finalized through the WMA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml


class ValidationError(RuntimeError):
    """The WMA session is incomplete, degraded, or not attributable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def read_object(path: Path) -> dict[str, Any]:
    if not regular(path):
        raise ValidationError(f"required regular file is missing or linked: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON object in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON top level is not an object: {path}")
    return value


def read_yaml_object(path: Path) -> dict[str, Any]:
    if not regular(path):
        raise ValidationError(f"required regular file is missing or linked: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"invalid YAML object in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"YAML top level is not an object: {path}")
    return value


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _next_event(
    rows: list[dict[str, Any]],
    card_id: str,
    after: int,
    event: str,
    label: str,
    predicate=lambda _row: True,
) -> tuple[int, dict[str, Any]]:
    for index in range(after + 1, len(rows)):
        row = rows[index]
        if (
            row.get("card_id") == card_id
            and row.get("event") == event
            and predicate(row)
        ):
            return index, row
    raise ValidationError(f"smoke lifecycle for {card_id} is missing {label}")


def _ping_from_event(session: Path, card_id: str, row: dict[str, Any]) -> dict[str, Any]:
    ping_id = row.get("ping_id")
    if not isinstance(ping_id, str):
        raise ValidationError(f"smoke lifecycle ping for {card_id} has no ping_id")
    expected = session / "wm" / "cards" / card_id / "pings" / f"{ping_id}.yaml"
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or Path(raw_path).resolve() != expected.resolve():
        raise ValidationError(f"smoke lifecycle ping path does not match {card_id}/{ping_id}")
    ping = read_yaml_object(expected)
    if (
        ping.get("card_id") != card_id
        or ping.get("ping_id") != ping_id
        or ping.get("kind") != row.get("kind")
        or ping.get("reply_required") is not True
    ):
        raise ValidationError(f"smoke lifecycle ping file disagrees with {card_id}/{ping_id}")
    return ping


def _reply_for_event(
    session: Path,
    card_id: str,
    ping_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    reply = read_yaml_object(
        session / "wm" / "cards" / card_id / "replies" / f"{ping_id}.yaml"
    )
    if (
        row.get("ping_id") != ping_id
        or reply.get("card_id") != card_id
        or reply.get("ping_id") != ping_id
        or reply.get("choice") != row.get("choice")
        or reply.get("by") != "scientist"
    ):
        raise ValidationError(f"smoke lifecycle reply file disagrees with {card_id}/{ping_id}")
    return reply


def _successful_call(
    call_audits: dict[int, dict[str, Any]],
    row: dict[str, Any],
    card_id: str,
    phase: str,
) -> None:
    audit = call_audits.get(int(row["seq"]))
    if audit is None:
        raise ValidationError(f"smoke lifecycle {phase} call for {card_id} has no audit")
    if (
        row.get("phase") != phase
        or audit.get("phase") != phase
        or audit.get("card_id") != card_id
        or audit.get("status") != "success"
    ):
        raise ValidationError(f"smoke lifecycle {phase} call for {card_id} is not correlated")
    for field in ("tool_event_count", "citation_count"):
        value = audit.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(
                f"smoke lifecycle {phase} call for {card_id} has no grounded {field}"
            )
        if row.get(field) != value:
            raise ValidationError(
                f"smoke lifecycle {phase} call event disagrees with its audit {field}"
            )


def _validate_smoke_card(
    session: Path,
    rows: list[dict[str, Any]],
    call_audits: dict[int, dict[str, Any]],
    card_id: str,
    expected_base_model: str,
    expected_base_checkpoint: Path,
) -> dict[str, Any]:
    position, _ = _next_event(rows, card_id, -1, "card_proposed", "card_proposed")
    position, brief_call = _next_event(
        rows,
        card_id,
        position,
        "wma_call",
        "successful brief wma_call",
        lambda row: row.get("phase") == "brief",
    )
    _successful_call(call_audits, brief_call, card_id, "brief")
    position, brief_ping_row = _next_event(
        rows,
        card_id,
        position,
        "ping",
        "brief ping",
        lambda row: row.get("kind") == "brief",
    )
    brief_ping = _ping_from_event(session, card_id, brief_ping_row)
    brief_id = str(brief_ping["ping_id"])
    position, brief_reply_row = _next_event(
        rows,
        card_id,
        position,
        "reply",
        "explicit scientist brief reply",
        lambda row: row.get("ping_id") == brief_id
        and row.get("choice") in ("accept", "override"),
    )
    brief_reply = _reply_for_event(session, card_id, brief_id, brief_reply_row)
    if brief_reply["choice"] == "override" and not brief_reply.get("why"):
        raise ValidationError(f"smoke lifecycle override for {card_id} has no reason")

    position, _ = _next_event(rows, card_id, position, "card_frozen", "card_frozen")
    position, _ = _next_event(rows, card_id, position, "parent_scored", "parent_scored")
    position, _ = _next_event(
        rows, card_id, position, "training_started", "training_started"
    )
    position, hook = _next_event(
        rows,
        card_id,
        position,
        "hook",
        "final checkpoint hook yield",
        lambda row: row.get("code") == 3
        and row.get("final") is True
        and isinstance(row.get("step"), int)
        and not isinstance(row.get("step"), bool)
        and row["step"] >= 1,
    )
    position, _ = _next_event(
        rows, card_id, position, "worker_spawned", "worker_spawned"
    )
    position, observation_row = _next_event(
        rows,
        card_id,
        position,
        "observation",
        "observation for the final hook",
        lambda row: row.get("step") == hook["step"]
        and isinstance(row.get("obs_id"), str),
    )
    obs_id = str(observation_row["obs_id"])
    values = observation_row.get("values")
    if not isinstance(values, dict) or not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values.values()
    ):
        raise ValidationError(f"smoke lifecycle observation {card_id}/{obs_id} is not finite")
    observation = read_object(
        session
        / "wm"
        / "cards"
        / card_id
        / "observations"
        / obs_id
        / "observation.json"
    )
    checkpoint = (observation.get("checkpoint") or {}).get("path")
    if (
        observation.get("card_id") != card_id
        or observation.get("obs_id") != obs_id
        or (observation.get("checkpoint") or {}).get("step") != hook["step"]
        or (observation.get("cause") or {}).get("final") is not True
        or not isinstance(checkpoint, str)
    ):
        raise ValidationError(f"smoke lifecycle observation file disagrees with {card_id}/{obs_id}")

    observation_phase = f"observation-{obs_id}"
    position, observation_call = _next_event(
        rows,
        card_id,
        position,
        "wma_call",
        "successful observation wma_call",
        lambda row: row.get("phase") == observation_phase,
    )
    _successful_call(call_audits, observation_call, card_id, observation_phase)
    position, decision_ping_row = _next_event(
        rows,
        card_id,
        position,
        "ping",
        "decision ping",
        lambda row: row.get("kind") == "decision",
    )
    decision_ping = _ping_from_event(session, card_id, decision_ping_row)
    if decision_ping.get("observation") != obs_id:
        raise ValidationError(f"smoke lifecycle decision ping does not name {card_id}/{obs_id}")
    decision_id = str(decision_ping["ping_id"])
    selection = f"select:{obs_id}"
    position, decision_reply_row = _next_event(
        rows,
        card_id,
        position,
        "reply",
        "explicit scientist decision reply",
        lambda row: row.get("ping_id") == decision_id and row.get("choice") == selection,
    )
    _reply_for_event(session, card_id, decision_id, decision_reply_row)
    position, _ = _next_event(
        rows,
        card_id,
        position,
        "decision_applied",
        "selected decision application",
        lambda row: row.get("ping_id") == decision_id and row.get("choice") == selection,
    )
    position, _ = _next_event(
        rows,
        card_id,
        position,
        "sealed",
        "seal for the selected observation",
        lambda row: row.get("obs_id") == obs_id and row.get("checkpoint") == checkpoint,
    )
    position, _ = _next_event(
        rows,
        card_id,
        position,
        "awaiting_review",
        "awaiting_review transition",
        lambda row: row.get("via") == "select" and row.get("obs_id") == obs_id,
    )
    position, adopted = _next_event(
        rows,
        card_id,
        position,
        "adopted",
        "adopted transition",
        lambda row: row.get("checkpoint") == checkpoint,
    )
    _next_event(
        rows,
        card_id,
        position,
        "card_closed",
        "finalized adoption",
        lambda row: row.get("how") == "finalize" and row.get("decision") == "adopt",
    )

    card_dir = session / "wm" / "cards" / card_id
    seal = read_object(card_dir / "seal.json")
    if (
        seal.get("card_id") != card_id
        or seal.get("obs_id") != obs_id
        or (seal.get("checkpoint") or {}).get("path") != checkpoint
        or seal.get("decision_ping") != decision_id
    ):
        raise ValidationError(f"smoke lifecycle seal disagrees with {card_id}/{obs_id}")
    state = read_object(card_dir / "state.json")
    state_seal = state.get("seal") or {}
    if (
        state.get("status") != "closed"
        or state.get("final_seen") is not True
        or state_seal.get("obs_id") != obs_id
        or state_seal.get("checkpoint") != checkpoint
    ):
        raise ValidationError(f"smoke lifecycle final state is incomplete for {card_id}")
    card = read_yaml_object(card_dir / "card.yaml")
    setup = card.get("setup") or {}
    parent = setup.get("parent_checkpoint") or {}
    result = card.get("result") or {}
    training = result.get("training_summary") or {}
    conclusion = card.get("conclusion") or {}
    if setup.get("base_model") != expected_base_model:
        raise ValidationError(
            f"smoke lifecycle {card_id} base model is not {expected_base_model}"
        )
    parent_path = parent.get("path")
    if (
        parent.get("origin") != "base_model"
        or not isinstance(parent_path, str)
        or Path(parent_path).resolve() != expected_base_checkpoint.resolve()
    ):
        raise ValidationError(
            f"smoke lifecycle {card_id} parent is not the expected official base checkpoint"
        )
    output_checkpoint = result.get("output_checkpoint")
    steps = training.get("steps")
    if (
        result.get("execution") != "completed"
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or not isinstance(output_checkpoint, str)
        or Path(output_checkpoint).resolve() != Path(checkpoint).resolve()
        or conclusion.get("decision") != "adopt"
    ):
        raise ValidationError(f"smoke lifecycle finalized card is incomplete for {card_id}")
    if adopted.get("submission") != str(session / "final_model"):
        raise ValidationError(f"smoke lifecycle adopted submission is wrong for {card_id}")
    return {
        "card_id": card_id,
        "brief_ping_id": brief_id,
        "decision_ping_id": decision_id,
        "observation_id": obs_id,
        "checkpoint": checkpoint,
        "base_model": expected_base_model,
        "base_checkpoint": str(expected_base_checkpoint.resolve()),
        "training_steps": steps,
    }


def validate(
    session: Path,
    *,
    require_smoke_lifecycle: bool = False,
    expected_base_model: str | None = None,
    expected_base_checkpoint: Path | None = None,
) -> dict[str, Any]:
    session = session.resolve()
    events_path = session / "wm" / "events.jsonl"
    if not regular(events_path) or events_path.stat().st_size == 0:
        raise ValidationError(f"WMA ledger is missing, empty, or linked: {events_path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(events_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValidationError(f"WMA ledger row {number} is not an object")
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid WMA ledger {events_path}: {exc}") from exc
    sequences = [row.get("seq") for row in rows]
    if not rows or any(isinstance(seq, bool) or not isinstance(seq, int) for seq in sequences):
        raise ValidationError("WMA ledger has missing or invalid sequence numbers")
    if sequences != sorted(set(sequences)):
        raise ValidationError("WMA ledger sequence is duplicate or non-monotonic")
    failures = [
        row for row in rows if row.get("event") in ("agent_failed", "agent_degraded")
    ]
    if failures:
        raise ValidationError("WMA ledger contains an agent_failed or agent_degraded event")

    successful_calls: list[dict[str, Any]] = []
    call_audits: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "wma_call" or not isinstance(row.get("card_id"), str):
            continue
        audit_value = row.get("path")
        if not isinstance(audit_value, str):
            raise ValidationError("wma_call event has no audit path")
        audit_path = Path(audit_value)
        expected_root = session / "wm" / "cards" / row["card_id"] / "wma-calls"
        if not inside(audit_path, expected_root):
            raise ValidationError(f"wma_call audit escapes its card directory: {audit_path}")
        audit = read_object(audit_path)
        if audit.get("status") != "success":
            raise ValidationError(f"wma_call audit is not successful: {audit_path}")
        successful_calls.append(row)
        call_audits[int(row["seq"])] = audit
    if not successful_calls:
        raise ValidationError("labelled WMA cell has no successful wma_call event")

    call_cards = {row["card_id"] for row in successful_calls}
    adopted_cards = {
        row.get("card_id")
        for row in rows
        if row.get("event") == "adopted" and isinstance(row.get("card_id"), str)
    }
    sealed_cards = {
        row.get("card_id")
        for row in rows
        if row.get("event") == "sealed"
        and isinstance(row.get("card_id"), str)
        and isinstance(row.get("checkpoint"), str)
    }
    finalized_adoptions = {
        row.get("card_id")
        for row in rows
        if row.get("event") == "card_closed"
        and row.get("how") == "finalize"
        and row.get("decision") == "adopt"
        and isinstance(row.get("card_id"), str)
    }
    qualified = sorted(call_cards & sealed_cards & adopted_cards & finalized_adoptions)
    if not qualified:
        raise ValidationError(
            "no card with a successful wma_call was sealed, adopted, and finalized"
        )
    submission = session / "final_model"
    if not submission.is_dir() or submission.is_symlink() or not any(submission.iterdir()):
        raise ValidationError("finalized WMA session has no non-empty real final_model")
    result = {
        "adopted_card_ids": qualified,
        "event_count": len(rows),
        "events_sha256": sha256_file(events_path),
        "successful_wma_call_count": len(successful_calls),
    }
    if require_smoke_lifecycle:
        if not expected_base_model or expected_base_checkpoint is None:
            raise ValidationError(
                "smoke lifecycle validation requires the expected base model and checkpoint"
            )
        failures: list[str] = []
        for card_id in qualified:
            try:
                result["smoke_lifecycle"] = _validate_smoke_card(
                    session,
                    rows,
                    call_audits,
                    card_id,
                    expected_base_model,
                    expected_base_checkpoint,
                )
                break
            except ValidationError as exc:
                failures.append(str(exc))
        else:
            raise ValidationError(
                "no adopted card completed the correlated smoke lifecycle: "
                + "; ".join(failures)
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--study-input", required=True, type=Path)
    parser.add_argument("--expected-base-model")
    parser.add_argument("--expected-base-checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        study = read_object(args.study_input)
        smoke = study.get("study_mode") == "smoke"
        evidence = validate(
            args.session,
            require_smoke_lifecycle=smoke,
            expected_base_model=args.expected_base_model,
            expected_base_checkpoint=args.expected_base_checkpoint,
        )
        if "wma_session" in study:
            raise ValidationError("study input already contains wma_session")
        study["wma_session"] = evidence
        atomic_json(args.record, evidence)
        atomic_json(args.study_input, study)
    except (OSError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
