"""Standard-library comparator evidence and portable completion receipts.

Shared by the CLI and the standalone Stop hook. This is reproducibility
evidence, not protection against an actor who can rewrite every input file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

RECEIPT_SCHEMA = "awm-deferred-comparator-v1"
FAILURE_OUTCOMES = ("failed", "killed", "not_run")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def receipt_path(card_path: Path) -> Path:
    return Path(card_path).with_name(Path(card_path).stem + ".comparator.json")


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def inspect_output(path: str, n: int, metric: str, *, allow_missing: bool = False) -> dict:
    """Read actual evaluated counts, not dataset population or requested limits."""
    def fail(detail):
        return {"status": "fail", "detail": detail}

    if not isinstance(path, str) or not Path(path).is_absolute() or not _integer(n):
        return fail("deferred comparator needs an absolute path and positive integer protocol n")
    if not isinstance(metric, str) or not metric.strip():
        return fail("deferred comparator needs a named expected-effect metric")
    p = Path(path)
    if not p.exists():
        if allow_missing:
            return {"status": "warn", "detail": f"{p}: measurement deferred; close must verify actual n and {metric}"}
        return fail(f"{p}: deferred comparator output is missing")
    try:
        raw = p.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError, UnicodeError):
        return fail(f"{p}: comparator must be a readable JSON report")
    if not isinstance(payload, dict) or not payload:
        return fail(f"{p}: comparator report must be a nonempty object")
    if "status" in payload and payload["status"] not in ("success", "completed"):
        return fail(f"{p.name}: evaluator did not report successful completion")

    counts = []
    for key in ("n", "num_samples"):
        if key in payload:
            counts.append((key, payload[key]))
    if "samples" in payload:
        samples = payload["samples"]
        if isinstance(samples, list) and any(isinstance(s, dict) and s.get("error") for s in samples):
            return fail(f"{p.name}: evaluator reported sample errors")
        counts.append(("samples", len(samples) if isinstance(samples, list) else samples))
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return fail(f"{p.name}: malformed results object")
    for key in ("total_samples", "completed_samples"):
        if key in results:
            counts.append((f"results.{key}", results[key]))
    scores = results.get("scores", [])
    if not isinstance(scores, list) or any(not isinstance(s, dict) for s in scores):
        return fail(f"{p.name}: malformed score records")
    values = []
    if metric in payload:
        values.append(payload[metric])
    for record in [payload, *scores]:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            return fail(f"{p.name}: malformed metrics object")
        if metric in metrics:
            value = metrics[metric]
            values.append(value.get("value") if isinstance(value, dict) else value)
            if "scored_samples" in record:
                counts.append(("scored_samples", record["scored_samples"]))
            if "unscored_samples" in record and (
                type(record["unscored_samples"]) is not int or record["unscored_samples"] != 0
            ):
                return fail(f"{p.name}: metric contains unscored samples")
    if not counts:
        return fail(f"{p.name}: actual sample count is absent; a requested limit or stderr estimate is not evidence")
    for source, actual in counts:
        if not _integer(actual):
            return fail(f"{p.name}: {source} must record a positive integer count, not {type(actual).__name__}")
        if actual != n:
            return fail(f"{p.name}: {source}={actual!r}, expected actual n={n}")
    if not values or any(not _number(v) for v in values):
        return fail(f"{p.name}: finite metric {metric!r} is absent or invalid")
    if any(not math.isclose(v, values[0], rel_tol=1e-12, abs_tol=1e-12) for v in values):
        return fail(f"{p.name}: conflicting values for metric {metric!r}")
    return {"status": "pass", "detail": f"{p.name}: actual n={n}, {metric}={values[0]}; remaining protocol identity needs manual verification",
            "n": n, "metric": metric, "value": values[0], "sha256": hashlib.sha256(raw).hexdigest()}


def completion_state(card_path: Path, lock_info: dict) -> dict:
    """Validate a close receipt without importing YAML or the AWM package."""
    def invalid(detail):
        return {"valid": False, "outcome": "unverified", "detail": detail}

    marker = lock_info.get("deferred_comparator")
    if not isinstance(marker, dict):
        return invalid("no deferred-comparator lock declaration")
    if (not isinstance(marker.get("path"), str) or not Path(marker["path"]).is_absolute()
            or not _integer(marker.get("n")) or not isinstance(marker.get("metric"), str)
            or not marker["metric"].strip() or not isinstance(marker.get("session_dir"), str)
            or not Path(marker["session_dir"]).is_absolute()):
        return invalid("malformed deferred-comparator declaration")
    if (not isinstance(lock_info.get("card_id"), str)
            or not isinstance(lock_info.get("locked_at"), str) or not lock_info["locked_at"]
            or not isinstance(lock_info.get("plan_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", lock_info["plan_sha256"])):
        return invalid("malformed comparator lock identity")
    try:
        proof_bytes = receipt_path(card_path).read_bytes()
        if lock_info.get("deferred_close_sha256") != hashlib.sha256(proof_bytes).hexdigest():
            return invalid("comparator receipt is not the close record sealed by this lock")
        proof = json.loads(proof_bytes)
        if not isinstance(proof, dict) or proof.get("schema_version") != RECEIPT_SCHEMA:
            return invalid("missing or unknown comparator receipt schema")
        if not isinstance(proof.get("verified_at"), str) or not proof["verified_at"]:
            return invalid("comparator receipt has no verification time")
        if (proof.get("card_id") != lock_info.get("card_id")
                or proof.get("plan_sha256") != lock_info.get("plan_sha256")
                or proof.get("locked_at") != lock_info.get("locked_at")
                or proof.get("declaration") != marker
                or proof.get("card_sha256") != digest(card_path)):
            return invalid("card, plan or lock changed since comparator closure")
    except (OSError, ValueError, UnicodeError):
        return invalid("comparator close receipt is missing or unreadable")
    outcome = proof.get("outcome")
    if outcome in FAILURE_OUTCOMES:
        if proof.get("observation") is not None:
            return invalid("failed closure cannot certify a metric")
        return {"valid": True, "outcome": outcome, "detail": "failed/unrun experiment closed without a verified comparison"}
    obs = proof.get("observation")
    if (outcome != "verified" or not isinstance(obs, dict) or obs.get("status") != "pass"
            or not _integer(obs.get("n")) or obs["n"] != marker["n"]
            or obs.get("metric") != marker["metric"] or not _number(obs.get("value"))
            or not isinstance(obs.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", obs["sha256"])):
        return invalid("invalid verified-comparator observation")

    original = Path(marker["path"])
    original_session = Path(marker["session_dir"])
    current_session = Path(card_path).resolve().parents[2]
    candidates = []
    try:
        candidates.append(current_session / original.relative_to(original_session))
    except ValueError:
        pass
    candidates.append(original)
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            try:
                if digest(candidate) != obs["sha256"]:
                    return invalid("comparator evidence changed after closure")
            except OSError:
                return invalid("comparator evidence cannot be re-read")
            return {"valid": True, "outcome": outcome, "evidence_available": True,
                    "detail": "verified comparator receipt and evidence hash"}
    if current_session == original_session.resolve():
        return invalid("comparator evidence is missing from the live session")
    return {"valid": True, "outcome": outcome, "evidence_available": False,
            "detail": "historical verification receipt; original evidence unavailable after relocation"}
