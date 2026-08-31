#!/usr/bin/env python3
"""Attest that a labelled WMA cell actually used and finalized through the WMA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


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


def validate(session: Path) -> dict[str, Any]:
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
    return {
        "adopted_card_ids": qualified,
        "event_count": len(rows),
        "events_sha256": sha256_file(events_path),
        "successful_wma_call_count": len(successful_calls),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--study-input", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        evidence = validate(args.session)
        study = read_object(args.study_input)
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
