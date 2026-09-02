"""Private online-WMA request worker used beside a PTB scientist sandbox."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import Budget, get_backend
from .review import ReviewError, review

REQUEST_SCHEMA = "awm-wma-review-request-v1"
RESPONSE_SCHEMA = "awm-wma-review-response-v1"
CARD_ID = re.compile(r"^exp-[0-9]+$")


@dataclass(frozen=True)
class Config:
    session_dir: Path
    skill_dir: Path
    history_dir: Path | None
    backend: str
    model: str
    effort: str
    budget: Budget
    jobs: int
    private_output_dir: Path | None = None
    scratch_dir: Path | None = None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rank(verdicts: list[dict[str, Any]]) -> list[str]:
    order = {"yes": 0, "defer": 1, "no": 2}
    return [
        verdict["card_id"]
        for verdict in sorted(
            verdicts,
            key=lambda verdict: (
                order.get(verdict["levels"]["L3_worth_now"].get("answer"), 3),
                -float(verdict["levels"]["L3_worth_now"].get("confidence") or 0),
            ),
        )
    ]


def process_request(path: Path, config: Config) -> dict[str, Any]:
    """Review one atomic request and return an auditable response."""

    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"unsupported request schema: {request.get('schema_version')!r}")
    request_id = str(request.get("request_id") or "")
    card_ids = request.get("card_ids")
    if (
        not request_id
        or not isinstance(card_ids, list)
        or not card_ids
        or len(card_ids) != len(set(card_ids))
        or any(not isinstance(card_id, str) or not CARD_ID.fullmatch(card_id) for card_id in card_ids)
    ):
        raise ValueError("request needs a request_id and distinct exp-NN card_ids")

    def one(card_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        lock = config.session_dir / "memory" / "cards" / f"{card_id}.lock.json"
        if not lock.is_file():
            return card_id, None, f"locked card required: {lock}"
        try:
            verdict = review(
                config.session_dir,
                card_id,
                get_backend(config.backend, config.model, config.effort),
                mode="online",
                budget=config.budget,
                model=config.model,
                skill_dir=config.skill_dir,
                history_dir=config.history_dir,
                effort=config.effort,
                expose_skill=False,
                transcript_dir=config.private_output_dir,
                allowed_roots=[config.scratch_dir] if config.scratch_dir else None,
            )
            return card_id, verdict, None
        except ReviewError as exc:
            return card_id, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, min(config.jobs, len(card_ids)))) as pool:
        results = list(pool.map(one, card_ids))
    verdicts = [verdict for _, verdict, error in results if verdict is not None and error is None]
    errors = {card_id: error for card_id, _, error in results if error}
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_id": request_id,
        "card_ids": card_ids,
        "state": "completed" if not errors else "partial" if verdicts else "failed",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": config.backend,
        "model": config.model,
        "effort": config.effort,
        "ranking": _rank(verdicts),
        "errors": errors,
    }


def run(config: Config, *, poll_seconds: float = 1.0, once: bool = False) -> int:
    control = config.session_dir / ".wma"
    requests = control / "requests"
    processing = control / "processing"
    processed = control / "processed"
    responses = control / "responses"
    stop = control / "stop"
    for directory in (requests, processing, processed, responses):
        directory.mkdir(parents=True, exist_ok=True)

    while True:
        pending = sorted(requests.glob("*.json"))
        for request_path in pending:
            claimed = processing / request_path.name
            try:
                os.replace(request_path, claimed)
            except FileNotFoundError:
                continue
            try:
                response = process_request(claimed, config)
            except (OSError, ValueError, TypeError) as exc:
                response = {
                    "schema_version": RESPONSE_SCHEMA,
                    "request_id": claimed.stem,
                    "card_ids": [],
                    "state": "failed",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "backend": config.backend,
                    "model": config.model,
                    "effort": config.effort,
                    "ranking": [],
                    "errors": {"request": str(exc)},
                }
            _atomic_json(responses / claimed.name, response)
            os.replace(claimed, processed / claimed.name)
        if once or (stop.exists() and not any(requests.glob("*.json"))):
            return 0
        time.sleep(max(0.05, poll_seconds))


def _budget(value: str) -> Budget:
    budget = Budget()
    for item in value.split(","):
        key, separator, raw = item.strip().partition("=")
        if not separator or key not in {"cpu", "gpu", "wall", "turns"}:
            raise ValueError(f"invalid budget item: {item!r}")
        if key == "turns":
            budget.max_turns = int(raw)
        else:
            setattr(budget, f"{key}_min", float(raw))
    return budget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--skill-dir", required=True, type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--backend", default="claude", choices=("claude", "codex", "heuristic"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--budget", default="cpu=10,gpu=0,wall=15,turns=40")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    config = Config(
        session_dir=args.session.resolve(),
        skill_dir=args.skill_dir.resolve(),
        history_dir=args.history.resolve() if args.history else None,
        backend=args.backend,
        model=args.model,
        effort=args.effort,
        budget=_budget(args.budget),
        jobs=args.jobs,
        private_output_dir=args.private_output.resolve() if args.private_output else None,
        scratch_dir=args.scratch_dir.resolve() if args.scratch_dir else None,
    )
    return run(config, poll_seconds=args.poll_seconds, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
