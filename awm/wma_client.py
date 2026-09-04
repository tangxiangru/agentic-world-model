"""Thin scientist-side client for a private online WMA sidecar.

The scientist may enqueue reviews and read their status, but this module ships
without ``awm.wma`` or ``skills/wma``.  The sidecar owns the model, skill,
history, budget, and verdict execution policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "awm-wma-review-request-v1"
CARD_ID = re.compile(r"^exp-[0-9]+$")
NOT_ATTACHED = ("no world-model agent is attached to this cell; no verdict will come — "
                "carry on with the next step of the protocol")
#: how long a blocking review waits before giving the launch back to the scientist. The sidecar's
#: own wall budget is 15 min per verdict; a second request queued behind a running one can take two.
DEFAULT_WAIT_MIN = 20.0


def say(line: str) -> None:
    """Print one progress line and flush it. Scientists background the wait (`nohup awm exp_protocol
    lock … > logs/lock.log &`) and tail the file; block-buffered stdout showed them an empty log for
    the whole wait (w10r04, 2026-09-03)."""
    print(line, flush=True)


class NoSidecar(RuntimeError):
    """No sidecar opened a queue in this session: the control arm, or a sidecar that never started."""


def _control_dir(session_dir: Path) -> Path:
    return Path(session_dir).resolve() / ".wma"


def sidecar_attached(session_dir: Path) -> bool:
    """The sidecar creates ``.wma/requests`` when it starts, before the scientist does; a cell without
    that directory has no world-model agent, and a request written there would wait forever."""
    return (_control_dir(session_dir) / "requests").is_dir()


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
    if not sidecar_attached(session_dir):
        raise NoSidecar(NOT_ATTACHED)
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
    # Freeze the exact plan/file state before another process can begin review.
    from .exp_protocol import decisions
    request["fingerprints"] = {cid: decisions.card_fingerprint(session_dir, cid) for cid in card_ids}
    decisions.write_once(decisions.safe_path(session_dir, f".wma/review-requests/{request_id}.json"), request)
    path = _control_dir(session_dir) / "requests" / f"{request_id}.json"
    _atomic_json(path, request)
    return request_id, path


def verdict_path(session_dir: Path, card_id: str) -> Path:
    return Path(session_dir).resolve() / "memory" / "cards" / f"{card_id}.verdict.json"


def _verdict_version(path: Path) -> tuple[int, int] | None:
    """A cheap identity for one delivered verdict.

    Re-locking asks for a fresh review while the previous verdict remains beside
    the card.  Size alone is insufficient (two verdicts may serialize equally),
    so include nanosecond mtime.  The sidecar rewrites the verdict on delivery.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _response_for(session_dir: Path, request_id: str) -> dict | None:
    path = _control_dir(session_dir) / "responses" / f"{request_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise_verdict(verdict: dict[str, Any]) -> str:
    """One line the scientist can act on: the four levels and the first precondition. The file's
    shape is the sidecar's; this reads it generically and never needs the private package."""
    levels = verdict.get("levels") or {}

    def level(name: str) -> str:
        entry = levels.get(name) or {}
        answer = entry.get("answer")
        if answer is None:
            answer = entry.get("direction")
        text = f"{name}={answer}"
        if entry.get("interval") is not None:
            text += f" {entry['interval']}"
        if entry.get("confidence") is not None:
            text += f"@{entry['confidence']}"
        return text

    parts = [level(n) for n in ("L0_runs", "L1_valid", "L2_effect", "L3_worth_now")]
    suggestions = verdict.get("suggestions") or {}
    preconditions = suggestions.get("preconditions") or []
    if preconditions:
        first = preconditions[0]
        if isinstance(first, dict):
            first = first.get("text") or first.get("what") or json.dumps(first)
        parts.append(f"first precondition: {str(first)[:160]}")
    return "; ".join(parts)


def wait_for_verdict(session_dir: Path, card_id: str, request_id: str, *,
                     timeout_s: float = DEFAULT_WAIT_MIN * 60, poll_s: float = 5.0,
                     heartbeat_s: float = 30.0, out: Callable[[str], None] = say,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep,
                     prior_verdict: tuple[int, int] | None = None,
                     require_request_match: bool = False) -> dict[str, Any]:
    """Block until the verdict file exists, the sidecar's response reports a failure, or the
    timeout passes. Returns ``{"state": delivered|failed|timeout, "waited_s", "verdict_path", "error"}``.
    The heartbeat keeps a tool runner from mistaking the wait for a hang."""
    started = clock()
    next_beat = started + heartbeat_s
    target = verdict_path(session_dir, card_id)
    while True:
        current_verdict = _verdict_version(target)
        if current_verdict is not None and current_verdict != prior_verdict:
            matches = not require_request_match
            if require_request_match:
                try:
                    matches = json.loads(target.read_text()).get("request_id") == request_id
                except (OSError, ValueError, AttributeError):
                    matches = False
            if matches:
                return {"state": "delivered", "waited_s": round(clock() - started, 1),
                        "verdict_path": str(target), "error": None}
        response = _response_for(session_dir, request_id)
        if response is not None and response.get("state") in ("failed", "partial"):
            errors = response.get("errors") or {}
            error = errors.get(card_id) or errors.get("request")
            if error or response.get("state") == "failed":
                return {"state": "failed", "waited_s": round(clock() - started, 1),
                        "verdict_path": None, "error": str(error or "review failed")}
        elapsed = clock() - started
        if elapsed >= timeout_s:
            return {"state": "timeout", "waited_s": round(elapsed, 1), "verdict_path": None,
                    "error": f"no verdict after {timeout_s / 60:.0f} min"}
        if clock() >= next_beat:
            out(f"waiting for the WMA verdict on {card_id}: {elapsed / 60:.1f} min elapsed")
            next_beat = clock() + heartbeat_s
        sleep(poll_s)


def review_and_wait(session_dir: Path, card_id: str, *, timeout_min: float = DEFAULT_WAIT_MIN,
                    out: Callable[[str], None] = say, **wait_kwargs: Any) -> dict[str, Any]:
    """The blocking form of a review: enqueue one card, wait, print the verdict line. The result
    dict is what `lock` records; ``state`` is ``not_attached`` on the control arm."""
    if not math.isfinite(timeout_min) or timeout_min <= 0:
        raise ValueError("review timeout must be finite and positive")
    session_dir = Path(session_dir).resolve()
    prior_verdict = _verdict_version(verdict_path(session_dir, card_id))
    try:
        request_id, _ = enqueue(session_dir, [card_id])
    except NoSidecar:
        out(NOT_ATTACHED)
        return {"state": "not_attached", "waited_s": 0.0, "verdict_path": None, "error": None,
                "request_id": None, "requested_at": None}
    requested_at = datetime.now(timezone.utc).isoformat()
    out(f"WMA review requested for {card_id} (request {request_id}); waiting up to "
        f"{timeout_min:.0f} min — prepare the launch meanwhile, do not start it")
    result = wait_for_verdict(session_dir, card_id, request_id, timeout_s=timeout_min * 60, out=out,
                              prior_verdict=prior_verdict, require_request_match=True, **wait_kwargs)
    result["request_id"] = request_id
    result["requested_at"] = requested_at
    # This is the enqueue-time fingerprint, never a post-hoc snapshot of an edited plan.
    request = json.loads((_control_dir(session_dir) / "review-requests" / (request_id + ".json")).read_text())
    result["fingerprint"] = request["fingerprints"][card_id]
    if result["state"] == "delivered":
        try:
            verdict = json.loads(Path(result["verdict_path"]).read_text(encoding="utf-8"))
            out(f"verdict: {summarise_verdict(verdict)}")
        except (OSError, ValueError) as exc:
            out(f"verdict: written but unreadable ({exc}); open {result['verdict_path']}")
        out(f"verdict file: {result['verdict_path']} — read it before launching")
    elif result["state"] == "failed":
        out(f"WMA review failed: {result['error']} — no verdict for this card; you may launch")
    else:
        out(f"WMA verdict did not arrive ({result['error']}); recorded as a timeout; you may launch")
    return result


def _proposal_command(args: argparse.Namespace) -> int:
    from .exp_protocol import decisions, treatment
    if treatment.mode(Path(args.dir)) == "single":
        say("single mode uses a formal card directly; no candidate-set request is configured")
        return 2
    path = decisions.create_proposal(Path(args.dir))
    say(f"wrote {path}; fill real candidates and your preference before comparing")
    return 0


def compare_and_wait(session_dir: Path, decision_id: str, *, timeout_min: float = DEFAULT_WAIT_MIN,
                     out: Callable[[str], None] = say, poll_s: float = 1.0) -> dict:
    from .exp_protocol import decisions, treatment

    if not math.isfinite(timeout_min) or timeout_min <= 0:
        raise ValueError("comparison timeout must be finite and positive")
    session = Path(session_dir).resolve()
    configured = treatment.identity(session)
    if configured["decision_mode"] == "single":
        raise ValueError("candidate comparison is not configured in single mode")
    proposal = decisions.load_proposal(session, decision_id)
    request_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    base = decisions.safe_path(session, f".wma/comparisons/{request_id}")
    request = {"schema_version": decisions.COMPARE_REQUEST_SCHEMA, "request_id": request_id,
               "created_at": decisions.now(), "decision_id": decision_id,
               "proposal": proposal, "proposal_sha256": decisions.proposal_sha(proposal),
               "treatment": configured}
    decisions.write_once(base / "request.json", request)
    if configured["decision_mode"] == "multi-self":
        result = {"state": "not_requested", "request_id": request_id, "waited_s": 0.0,
                  "treatment": configured, "sidecar_attached": sidecar_attached(session)}
        decisions.write_once(base / "completion.json", result)
        out("multi-self mode: briefs frozen for your choice; no joint WMA call was requested")
        return result
    if not sidecar_attached(session):
        result = {"state": "not_attached", "request_id": request_id, "waited_s": 0.0}
        decisions.write_once(base / "completion.json", result)
        out(NOT_ATTACHED)
        return result
    _atomic_json(_control_dir(session) / "requests" / f"{request_id}.json", request)
    started = time.monotonic()
    heartbeat = started
    out(f"comparing {len(proposal['candidates'])} candidate(s) together; this does not authorize a launch")
    while True:
        response = _response_for(session, request_id)
        if response is not None:
            result = {"state": response.get("state"), "request_id": request_id,
                      "waited_s": round(time.monotonic() - started, 1), "errors": response.get("errors", {})}
            if response.get("state") == "completed":
                value = json.loads((base / "comparison.json").read_text())
                decisions.validate_comparison(value, proposal)
                result["comparison_path"] = str(base / "comparison.json")
                out("joint ranking: " + " > ".join(value["ranking"]))
                for pair in value["comparisons"]:
                    out(f"  {pair['preferred']} before {pair['alternative']}: {pair['reason']}; uncertainty: {pair['uncertainty']}")
            else:
                out(f"comparison failed: {response.get('errors')}; record your own choice and continue with formal lock review")
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout_min * 60:
            result = {"state": "timeout", "request_id": request_id, "waited_s": round(elapsed, 1)}
            out("comparison timed out; record your own choice and retain formal lock review")
            break
        if time.monotonic() - heartbeat >= 30:
            out(f"waiting for joint comparison: {elapsed / 60:.1f} min")
            heartbeat = time.monotonic()
        time.sleep(poll_s)
    decisions.write_once(base / "completion.json", result)
    return result


def _compare_command(args: argparse.Namespace) -> int:
    try:
        result = compare_and_wait(Path(args.dir), args.decision_id, timeout_min=args.timeout_min)
    except (ValueError, OSError) as exc:
        say(f"not compared: {exc}")
        return 2
    return 0 if result["state"] in ("completed", "not_attached", "not_requested") else 3


def record_choice(session: Path, decision_id: str, candidate_id: str | None, reason: str,
                  card_id: str | None = None) -> Path:
    from .exp_protocol import decisions, treatment

    configured = treatment.identity(session)
    if configured["decision_mode"] == "single":
        raise ValueError("single mode uses a formal card without a candidate-set choice")
    proposal = decisions.load_proposal(session, decision_id)
    if (candidate_id is not None and candidate_id not in [c["candidate_id"] for c in proposal["candidates"]]) or not reason.strip():
        raise ValueError("choose an existing candidate and explain why")
    if candidate_id is None and card_id:
        raise ValueError("declining all candidates cannot bind a training card")
    if card_id and not decisions.CARD_ID.fullmatch(card_id):
        raise ValueError("card must be exp-NN")
    requests = []
    for path, r in decisions.read_records(decisions.safe_path(session, ".wma/comparisons").glob("*/request.json")):
        if r.get("decision_id") == decision_id:
            requests.append((path, r))
    if not requests:
        raise ValueError("compare the proposal set first, including in the no-WMA control")
    path, request = max(requests, key=lambda x: x[1]["created_at"])
    if request.get("treatment", treatment.describe(treatment.DEFAULT, explicit=False)) != configured:
        raise ValueError("study mode changed after the candidate request")
    if request["proposal_sha256"] != decisions.proposal_sha(proposal):
        raise ValueError("proposal changed since comparison; compare the new version first")
    completion = path.parent / "completion.json"
    if not completion.is_file():
        raise ValueError("wait for comparison to return before recording the final choice")
    result = json.loads(completion.read_text())
    allowed = ({"not_requested"} if configured["decision_mode"] == "multi-self"
               else {"completed", "not_attached", "failed", "timeout"})
    if result.get("state") not in allowed or result.get("request_id") != request["request_id"]:
        raise ValueError("completion does not match this study mode and request")
    record = {"schema_version": "awm-wma-choice-v1", "created_at": decisions.now(),
              "decision_id": decision_id, "request_id": request["request_id"],
              "proposal_sha256": request["proposal_sha256"],
              "scientist_preference": proposal["scientist_preference"], "selected": candidate_id,
              "reason": reason, "card_id": card_id, "comparison_state": result["state"],
              "treatment": configured}
    record["bound_plan_sha256"] = None
    record["bound_inputs"] = None
    if card_id:
        from .exp_protocol import lock, schema
        card_path = decisions.safe_path(session, f"memory/cards/{card_id}.yaml")
        if card_path.is_file():
            card = schema.load_card(card_path)
            record["bound_plan_sha256"] = schema.plan_hash(card)
            record["bound_inputs"] = lock.plan_inputs(card)
    target = decisions.safe_path(session, f"memory/decisions/{decision_id}/choices/{uuid.uuid4().hex}.json")
    decisions.write_once(target, record)
    return target


def _choose_command(args: argparse.Namespace) -> int:
    try:
        say(str(record_choice(Path(args.dir), args.decision_id, args.candidate, args.reason, args.card)))
        return 0
    except (ValueError, OSError) as exc:
        say(f"choice not recorded: {exc}")
        return 2


def _act_command(args: argparse.Namespace) -> int:
    from .exp_protocol import decisions
    try:
        path = decisions.append_action(Path(args.dir), args.card_id, args.action, args.reason,
                                       suggestion=args.suggestion, evidence=args.evidence)
        say(str(path))
        return 0
    except (ValueError, OSError) as exc:
        say(f"action not recorded: {exc}")
        return 2


def register_decisions(commands: argparse._SubParsersAction) -> None:
    from .exp_protocol import decisions

    propose = commands.add_parser("propose", help="prepare real candidate briefs before selecting a training")
    propose.add_argument("--dir", required=True)
    propose.set_defaults(func=_proposal_command)
    compare = commands.add_parser("compare", help="one joint private review of a frozen candidate set")
    compare.add_argument("--dir", required=True)
    compare.add_argument("decision_id")
    compare.add_argument("--timeout-min", type=float, default=DEFAULT_WAIT_MIN)
    compare.set_defaults(func=_compare_command)
    choose = commands.add_parser("choose", help="record the scientist's final choice and reason")
    choose.add_argument("--dir", required=True)
    choose.add_argument("decision_id")
    selection = choose.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidate")
    selection.add_argument("--decline", action="store_true", help="decline every supplied candidate with a reason")
    choose.add_argument("--reason", required=True)
    choose.add_argument("--card")
    choose.set_defaults(func=_choose_command)
    act = commands.add_parser("act", help="record a current-plan action after review and before launch")
    act.add_argument("--dir", required=True)
    act.add_argument("card_id")
    act.add_argument("--action", choices=decisions.ACTIONS, required=True)
    act.add_argument("--reason", required=True)
    act.add_argument("--suggestion")
    act.add_argument("--evidence", action="append", default=[])
    act.set_defaults(func=_act_command)


def _review(args: argparse.Namespace) -> int:
    if args.background:
        try:
            request_id, path = enqueue(Path(args.dir), list(args.card_id))
        except NoSidecar as exc:
            print(str(exc))          # not an error: the control arm's whole answer
            return 0
        except ValueError as exc:
            print(f"not queued: {exc}")
            return 2
        print(
            f"WMA review queued for {', '.join(args.card_id)} (request {request_id}); "
            f"keep working. `awm wma status --dir {args.dir}` shows verdict progress; request: {path}"
        )
        return 0
    if len(args.card_id) != 1:
        print("a blocking review takes one card; batch several with --background")
        return 2
    try:
        result = review_and_wait(Path(args.dir), args.card_id[0], timeout_min=args.timeout_min)
    except ValueError as exc:
        print(f"not queued: {exc}")
        return 2
    return 0 if result["state"] in ("delivered", "not_attached") else 3


def _status(args: argparse.Namespace) -> int:
    session = Path(args.dir).resolve()
    cards = session / "memory" / "cards"
    if not cards.is_dir():
        print(f"no cards under {session}")
        return 2
    control = _control_dir(session)
    if not sidecar_attached(session):
        print(NOT_ATTACHED)
        return 0
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
    wma = sub.add_parser("wma", help="ask the private world-model agent about a locked card")
    commands = wma.add_subparsers(dest="cmd", required=True)
    register_decisions(commands)
    review = commands.add_parser(
        "review",
        help="ask the WMA about a locked card and wait for the verdict; --background queues without waiting")
    review.add_argument("--dir", required=True)
    review.add_argument("card_id", nargs="+")
    review.add_argument("--background", action="store_true",
                        help="queue (several cards allowed) and return at once; for extra or batch reviews")
    review.add_argument("--timeout-min", type=float, default=DEFAULT_WAIT_MIN,
                        help=f"how long the blocking form waits (default {DEFAULT_WAIT_MIN:.0f} min)")
    review.set_defaults(func=_review)
    status = commands.add_parser("status", help="show queued and completed WMA reviews")
    status.add_argument("--dir", required=True)
    status.set_defaults(func=_status)
