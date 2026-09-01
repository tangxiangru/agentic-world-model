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


def read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    if not regular(path) or path.stat().st_size == 0:
        raise ValidationError(f"{label} is missing, empty, or linked: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValidationError(f"{label} row {number} is not an object")
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label} {path}: {exc}") from exc
    if not rows:
        raise ValidationError(f"{label} contains no rows: {path}")
    return rows


def _read_exit_code(path: Path, label: str) -> int:
    if not regular(path):
        raise ValidationError(f"{label} exit record is missing or linked: {path}")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValidationError(f"invalid {label} exit record {path}: {exc}") from exc


def validate_peer_session(
    session: Path,
    *,
    expected_arm: str,
    expected_wma_model: str,
    expected_memory_sides: str,
    study: dict[str, Any],
) -> dict[str, Any]:
    """Validate the upstream two-Claude-session `consult` protocol.

    A labelled C2/C3 cell is released only if the peer started with the exact
    evidence arm, completed normally, logged at least one validated consult,
    received the shipped outcome, and was independently model-attested.
    """

    session = session.resolve()
    wm = session / "wm"
    cfg = read_object(wm / "config.json")
    if cfg.get("schema_version") != "awm-wm-config-v2":
        raise ValidationError("peer WMA config must use awm-wm-config-v2")
    if cfg.get("session_dir") != str(session):
        raise ValidationError("peer WMA config session_dir does not name this task")
    if cfg.get("arm") != expected_arm:
        raise ValidationError(
            f"peer WMA arm {cfg.get('arm')!r} does not match {expected_arm!r}"
        )
    if cfg.get("wma_model") != expected_wma_model:
        raise ValidationError("peer WMA config does not pin the expected model")
    expected_sides = expected_memory_sides.split(",")
    if cfg.get("memory_sides") != expected_sides:
        raise ValidationError(
            f"peer WMA memory sides {cfg.get('memory_sides')!r} do not match {expected_sides!r}"
        )
    if cfg.get("base_model") != "google/gemma-3-4b-pt":
        raise ValidationError("peer WMA config does not name the pinned Gemma base")
    if "SendMessage" not in str(cfg.get("consult_api")):
        raise ValidationError("peer WMA config does not identify the SendMessage consult API")

    condition = study.get("condition")
    if condition == "c2":
        if expected_arm != "traj" or cfg.get("memory_root") is not None:
            raise ValidationError("C2 must expose raw trajectories only through arm=traj")
        if cfg.get("prior_runs_root") != "/home/ben/prior_runs":
            raise ValidationError("C2 peer does not point at the attested prior-run mount")
    elif condition == "c3":
        if expected_arm != "retrieval" or cfg.get("prior_runs_root") is not None:
            raise ValidationError("C3 must expose experiment cards only through arm=retrieval")
        if cfg.get("memory_root") != "/home/ben/wm-memory":
            raise ValidationError("C3 peer does not point at the attested card-memory mount")
    else:
        raise ValidationError(f"peer WMA validator requires C2 or C3, got {condition!r}")

    if _read_exit_code(wm / "wma-exit-code.txt", "WMA") != 0:
        raise ValidationError("WMA peer did not complete normally")
    stream = wm / "wma-session.jsonl"
    stream_rows = read_jsonl_objects(stream, "WMA stream")
    if not any(row.get("type") == "result" for row in stream_rows):
        raise ValidationError("WMA stream has no terminal result event")

    model_attestation = study.get("wma_model")
    if not isinstance(model_attestation, dict):
        raise ValidationError("study input has no independent WMA model attestation")
    if model_attestation.get("expected_model_id") != expected_wma_model:
        raise ValidationError("WMA model attestation names a different model")
    if model_attestation.get("reported_model_ids") != [expected_wma_model]:
        raise ValidationError("WMA model telemetry does not match the pinned model")
    if model_attestation.get("reported_providers") != ["vertex"]:
        raise ValidationError("WMA model telemetry is not Vertex-only")

    ledger_path = wm / "consults.jsonl"
    rows = read_jsonl_objects(ledger_path, "WMA consult ledger")
    sequences = [row.get("seq") for row in rows]
    if sequences != list(range(1, len(rows) + 1)):
        raise ValidationError("WMA consult ledger sequence is not contiguous from 1")
    consults = [row for row in rows if row.get("event") != "outcome"]
    outcomes = [row for row in rows if row.get("event") == "outcome"]
    if not consults:
        raise ValidationError("labelled WMA cell has no validated consult")
    if not outcomes:
        raise ValidationError("scientist did not tell the WMA what it shipped")
    for row in consults:
        if row.get("arm") != expected_arm or row.get("model") != expected_wma_model:
            raise ValidationError("consult ledger arm/model attribution changed within the cell")
        if not isinstance(row.get("verdict"), str) or not isinstance(
            row.get("suggestion"), str
        ):
            raise ValidationError("consult ledger row lacks a verdict or suggestion")
        value = row.get("path")
        if not isinstance(value, str):
            raise ValidationError("consult ledger row has no persisted response path")
        response_path = Path(value)
        if not inside(response_path, wm / "cards") or not regular(response_path):
            raise ValidationError(f"consult response escapes task/wm/cards: {response_path}")
        read_object(response_path)
        card_id = row.get("card_id")
        if not isinstance(card_id, str):
            raise ValidationError("consult ledger row has no card_id")
        card = read_object(wm / "cards" / card_id / "card.json")
        if card.get("card_id") != card_id:
            raise ValidationError(f"persisted card identity mismatch for {card_id}")
    consult_cards = {row.get("card_id") for row in consults}
    if not any(row.get("card_id") in consult_cards for row in outcomes):
        raise ValidationError("shipped outcome is not attached to a consulted card")

    submission = session / "final_model"
    if not submission.is_dir() or submission.is_symlink() or not any(submission.iterdir()):
        raise ValidationError("peer-WMA cell has no non-empty real final_model")
    return {
        "protocol": "peer-consult-v1",
        "arm": expected_arm,
        "wma_model": expected_wma_model,
        "memory_sides": expected_sides,
        "consult_count": len(consults),
        "outcome_count": len(outcomes),
        "card_ids": sorted(str(card_id) for card_id in consult_cards),
        "consults_sha256": sha256_file(ledger_path),
        "wma_stream_sha256": sha256_file(stream),
    }


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


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} is not a non-empty path")
    return Path(os.path.abspath(value))


def _validate_base_lineage(
    session: Path,
    rows: list[dict[str, Any]],
    qualified_cards: set[str],
    expected_base_model: str,
    expected_base_checkpoint: Path,
) -> dict[str, Any]:
    """Attest every executed card and the final incumbent back to one exact base."""
    base = Path(os.path.abspath(expected_base_checkpoint))
    if not base.is_dir() or base.is_symlink():
        raise ValidationError(
            "expected official base checkpoint is missing, not a directory, or linked"
        )
    base_resolved = base.resolve()

    training_started: dict[str, int] = {}
    finalized: dict[str, int] = {}
    adopted_events: list[dict[str, Any]] = []
    for row in rows:
        card_id = row.get("card_id")
        if not isinstance(card_id, str):
            continue
        if row.get("event") == "training_started":
            if card_id in training_started:
                raise ValidationError(f"base lineage has duplicate training_started for {card_id}")
            training_started[card_id] = int(row["seq"])
        elif row.get("event") == "card_closed" and row.get("how") == "finalize":
            if card_id in finalized:
                raise ValidationError(f"base lineage has duplicate finalization for {card_id}")
            finalized[card_id] = int(row["seq"])
        elif row.get("event") == "adopted" and isinstance(row.get("checkpoint"), str):
            adopted_events.append(row)

    if not adopted_events:
        raise ValidationError("base lineage has no adopted incumbent event")
    final_adopted = adopted_events[-1]
    final_card_id = str(final_adopted["card_id"])
    if final_card_id not in qualified_cards:
        raise ValidationError(
            f"final incumbent {final_card_id} lacks a successful WMA call, seal, and finalization"
        )
    if final_card_id not in training_started:
        raise ValidationError(f"final incumbent {final_card_id} has no training_started event")

    for card_id, started in training_started.items():
        if card_id not in finalized:
            raise ValidationError(
                f"base lineage card {card_id} has no terminal card_closed/finalize event"
            )
        if finalized[card_id] <= started:
            raise ValidationError(
                f"base lineage card {card_id} finalization does not follow training_started"
            )
    if not (
        training_started[final_card_id]
        < int(final_adopted["seq"])
        < finalized[final_card_id]
    ):
        raise ValidationError(
            f"final incumbent {final_card_id} adoption is not between training and finalization"
        )

    incumbent = read_object(session / "wm" / "incumbent.json")
    incumbent_card_id = incumbent.get("card_id")
    incumbent_checkpoint = _absolute_path(
        incumbent.get("checkpoint"), "wm/incumbent.json checkpoint"
    )
    adopted_checkpoint = _absolute_path(
        final_adopted.get("checkpoint"), "final adopted checkpoint"
    )
    if (
        incumbent_card_id != final_card_id
        or incumbent_checkpoint != adopted_checkpoint
        or incumbent_checkpoint.resolve() != adopted_checkpoint.resolve()
    ):
        raise ValidationError("wm/incumbent.json does not match the final adopted event")
    if final_adopted.get("submission") != str(session / "final_model"):
        raise ValidationError("final adopted event does not name this session's final_model")

    lineage_ids = set(training_started)
    lineage_ids.add(final_card_id)
    card_root = session / "wm" / "cards"
    cards: dict[str, dict[str, Any]] = {}
    parent_paths: dict[str, Path] = {}
    parent_origins: dict[str, str] = {}
    output_paths: dict[str, Path] = {}
    for card_id in lineage_ids:
        card_path = card_root / card_id / "card.yaml"
        if not inside(card_path, card_root):
            raise ValidationError(f"base lineage card path escapes the session: {card_id}")
        card = read_yaml_object(card_path)
        if card.get("card_id") != card_id:
            raise ValidationError(f"base lineage card file disagrees with {card_id}")
        setup = card.get("setup")
        if not isinstance(setup, dict):
            raise ValidationError(f"base lineage card {card_id} has no setup")
        if setup.get("base_model") != expected_base_model:
            raise ValidationError(
                f"base lineage card {card_id} base model is not {expected_base_model}"
            )
        parent = setup.get("parent_checkpoint")
        if not isinstance(parent, dict):
            raise ValidationError(f"base lineage card {card_id} has no parent checkpoint")
        parent_path = _absolute_path(
            parent.get("path"), f"base lineage card {card_id} parent checkpoint"
        )
        origin = parent.get("origin")
        if not isinstance(origin, str) or not origin:
            raise ValidationError(f"base lineage card {card_id} has no parent origin")
        if not parent_path.is_dir() or parent_path.is_symlink():
            raise ValidationError(
                f"base lineage card {card_id} parent checkpoint is missing or linked"
            )
        result = card.get("result")
        output = result.get("output_checkpoint") if isinstance(result, dict) else None
        output_path = _absolute_path(
            output, f"base lineage card {card_id} output checkpoint"
        )
        if not inside(output_path, session):
            raise ValidationError(
                f"base lineage card {card_id} output checkpoint escapes the session"
            )
        if not output_path.is_dir() or output_path.is_symlink():
            raise ValidationError(
                f"base lineage card {card_id} output checkpoint is missing or linked"
            )
        output_paths[card_id] = output_path
        cards[card_id] = card
        parent_paths[card_id] = parent_path
        parent_origins[card_id] = origin

    output_owners: dict[Path, list[str]] = {}
    for card_id, output_path in output_paths.items():
        output_owners.setdefault(output_path, []).append(card_id)
    duplicated_outputs = {
        path: owners for path, owners in output_owners.items() if len(owners) != 1
    }
    if duplicated_outputs:
        raise ValidationError("base lineage has ambiguous reused output checkpoint paths")

    parents: dict[str, str | None] = {}
    for card_id, parent_path in parent_paths.items():
        if parent_path == base and parent_path.resolve() == base_resolved:
            if parent_origins[card_id] != "base_model":
                raise ValidationError(
                    f"base lineage root {card_id} does not declare origin base_model"
                )
            parents[card_id] = None
            continue
        owners = output_owners.get(parent_path, [])
        if len(owners) != 1:
            raise ValidationError(
                f"base lineage card {card_id} parent does not reference the official base "
                "or one executed card output checkpoint"
            )
        if parent_origins[card_id] != owners[0]:
            raise ValidationError(
                f"base lineage card {card_id} parent origin does not name {owners[0]}"
            )
        parents[card_id] = owners[0]

    for card_id in lineage_ids:
        chain: set[str] = set()
        cursor: str | None = card_id
        while cursor is not None:
            if cursor in chain:
                raise ValidationError(f"base lineage contains a cycle through {cursor}")
            chain.add(cursor)
            cursor = parents[cursor]

    for card_id, parent_id in parents.items():
        if parent_id is None:
            continue
        child_started = training_started.get(card_id)
        parent_finalized = finalized.get(parent_id)
        if (
            child_started is None
            or parent_finalized is None
            or parent_finalized >= child_started
        ):
            raise ValidationError(
                f"base lineage card {card_id} does not follow an earlier finalized card output"
            )

    final_output = output_paths.get(final_card_id)
    if (
        final_output is None
        or final_output != incumbent_checkpoint
        or final_output.resolve() != incumbent_checkpoint.resolve()
        or not final_output.is_dir()
        or final_output.is_symlink()
    ):
        raise ValidationError(
            f"final incumbent {final_card_id} does not match its real output checkpoint"
        )
    conclusion = cards[final_card_id].get("conclusion")
    if not isinstance(conclusion, dict) or conclusion.get("decision") != "adopt":
        raise ValidationError(f"final incumbent {final_card_id} card is not an adoption")

    ordered = sorted(
        lineage_ids,
        key=lambda card_id: (training_started.get(card_id, int(final_adopted["seq"])), card_id),
    )
    return {
        "base_model": expected_base_model,
        "base_checkpoint": str(base),
        "executed_card_ids": sorted(
            training_started, key=lambda card_id: training_started[card_id]
        ),
        "final_card_id": final_card_id,
        "final_checkpoint": str(final_output),
        "cards": [
            {
                "card_id": card_id,
                "parent": parents[card_id] or "base_model",
                "output_checkpoint": (
                    str(output_paths[card_id]) if card_id in output_paths else None
                ),
            }
            for card_id in ordered
        ],
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
    if (expected_base_model is None) != (expected_base_checkpoint is None):
        raise ValidationError(
            "base lineage validation requires both expected base model and checkpoint"
        )
    if expected_base_model is not None and expected_base_checkpoint is not None:
        result["base_lineage"] = _validate_base_lineage(
            session,
            rows,
            set(qualified),
            expected_base_model,
            expected_base_checkpoint,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--study-input", required=True, type=Path)
    parser.add_argument("--expected-arm", choices=("traj", "retrieval"))
    parser.add_argument("--expected-wma-model")
    parser.add_argument("--expected-memory-sides", default="train")
    parser.add_argument("--expected-base-model")
    parser.add_argument("--expected-base-checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        study = read_object(args.study_input)
        if regular(args.session.resolve() / "wm" / "config.json"):
            if not args.expected_arm or not args.expected_wma_model:
                raise ValidationError(
                    "peer validation requires --expected-arm and --expected-wma-model"
                )
            evidence = validate_peer_session(
                args.session,
                expected_arm=args.expected_arm,
                expected_wma_model=args.expected_wma_model,
                expected_memory_sides=args.expected_memory_sides,
                study=study,
            )
        else:
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
