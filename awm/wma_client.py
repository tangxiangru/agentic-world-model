"""Thin scientist-side client for a private online WMA sidecar.

The scientist may enqueue reviews and read their status, but this module ships
without ``awm.wma`` or ``skills/wma``.  The sidecar owns the model, skill,
history, budget, and verdict execution policy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

REQUEST_SCHEMA = "awm-wma-review-request-v1"
CARD_ID = re.compile(r"^exp-[0-9]+$")


def _control_dir(session_dir: Path) -> Path:
    return Path(session_dir).resolve() / ".wma"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def enqueue(session_dir: Path, card_ids: list[str]) -> tuple[str, Path]:
    """Atomically enqueue one batch without exposing WMA implementation details."""

    session_dir = Path(session_dir).resolve()
    if not session_dir.is_dir():
        raise ValueError(f"session directory does not exist: {session_dir}")
    if not card_ids:
        raise ValueError("at least one experiment card is required")
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("card ids must be distinct within one review request")
    for card_id in card_ids:
        if not CARD_ID.fullmatch(card_id):
            raise ValueError(f"invalid card id: {card_id}")
        card = session_dir / "memory" / "cards" / f"{card_id}.yaml"
        if not card.is_file():
            raise ValueError(f"no such card: {card}")

    request_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex[:8]}"
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "card_ids": card_ids,
    }
    path = _control_dir(session_dir) / "requests" / f"{request_id}.json"
    _atomic_json(path, request)
    return request_id, path


def _review(args: argparse.Namespace) -> int:
    if not args.background:
        print("online WMA review is asynchronous; pass --background")
        return 2
    try:
        request_id, path = enqueue(Path(args.dir), list(args.card_id))
    except ValueError as exc:
        print(f"not queued: {exc}")
        return 2
    print(
        f"WMA review queued for {', '.join(args.card_id)} (request {request_id}); "
        f"keep working. `awm wma status --dir {args.dir}` shows verdict progress; request: {path}"
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    session = Path(args.dir).resolve()
    cards = session / "memory" / "cards"
    if not cards.is_dir():
        print(f"no cards under {session}")
        return 2
    control = _control_dir(session)
    responses: dict[str, dict] = {}
    for path in sorted((control / "responses").glob("*.json")):
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for card_id in response.get("card_ids") or []:
            responses[str(card_id)] = response
    queued_ids: set[str] = set()
    for directory in (control / "requests", control / "processing"):
        for path in directory.glob("*.json"):
            try:
                queued_ids.update(json.loads(path.read_text(encoding="utf-8")).get("card_ids") or [])
            except (OSError, ValueError):
                continue
    for card in sorted(cards.glob("exp-*.yaml")):
        card_id = card.stem
        verdict = card.with_name(f"{card_id}.verdict.json")
        if verdict.is_file():
            print(f"{card_id}: verdict ready")
        elif card_id in queued_ids:
            print(f"{card_id}: review queued/running")
        elif card_id in responses:
            print(f"{card_id}: review {responses[card_id].get('state', 'unknown')}")
        else:
            print(f"{card_id}: no verdict")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    wma = sub.add_parser("wma", help="request a private asynchronous WMA review")
    commands = wma.add_subparsers(dest="cmd", required=True)
    review = commands.add_parser("review", help="queue one or more locked cards for WMA review")
    review.add_argument("--dir", required=True)
    review.add_argument("card_id", nargs="+")
    review.add_argument("--background", action="store_true")
    review.set_defaults(func=_review)
    status = commands.add_parser("status", help="show queued and completed WMA reviews")
    status.add_argument("--dir", required=True)
    status.set_defaults(func=_status)
