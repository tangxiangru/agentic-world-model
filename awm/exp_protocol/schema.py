"""Experiment card v2: constants, loading, and validation.

A card is one experiment — one situation, one problem, one hypothesis, one
intervention, one result. Sections 0–4 (situation, problem, hypothesis, setup,
evaluation) are written before the launch command runs; sections 5–6 (result,
conclusion) after. ``validate_plan`` checks the former, ``validate_result`` the
latter. Both run on a laptop; neither touches a GPU or the network.

Errors are things the protocol refuses; warnings are things a careful scientist
would want to know. ``Report.ok`` means no errors.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CARD_SCHEMA = "awm-experiment-card-v2"
PLAN_SECTIONS = ("situation", "problem", "hypothesis", "setup", "evaluation")
RESULT_SECTIONS = ("result", "conclusion")
CARD_ID_RE = re.compile(r"^exp-\d{2,}$")
METHOD_FAMILIES = ("sft", "rft", "dpo", "grpo", "distill", "merge", "decode-config", "other")
EXECUTIONS = ("completed", "failed", "killed", "not_run")
VERDICTS = ("supported", "contradicted", "inconclusive")
MECHANISM_VERDICTS = ("supported", "contradicted", "not_tested")
DECISIONS = ("adopt", "reject", "iterate", "abandon_line")
DIRECTIONS = ("higher", "lower")
CONTAMINATION = ("passed", "failed", "not_run")
KEEP_POLICIES = ("all", "last", "best")
#: "reach 85 %" is a target, not a hypothesis (experiment-card-v1.md, "Hypothesis").
TARGET_RE = re.compile(r"\b(reach|hit|achieve|target)\b.*\d", re.IGNORECASE)


class CardError(ValueError):
    """A card file cannot be read as a card at all."""


@dataclass
class Problem:
    field: str
    message: str
    severity: str = "error"  # error | warning


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, field_: str, message: str) -> None:
        self.problems.append(Problem(field_, message, "error"))

    def warn(self, field_: str, message: str) -> None:
        self.problems.append(Problem(field_, message, "warning"))

    def render(self) -> str:
        lines = [f"{p.severity.upper():7} {p.field}: {p.message}" for p in self.problems]
        return "\n".join(lines) if lines else "ok"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_card(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CardError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CardError(f"{path}: a card is a YAML mapping")
    return data


def dump_card(path: Path, card: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))


def get(card: dict[str, Any], dotted: str) -> Any:
    """``get(card, "setup.command.argv")`` → the value, or None when any step is missing."""
    node: Any = card
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def plan_hash(card: dict[str, Any]) -> str:
    """Hash of the pre-launch sections only; what ``lock`` records and ``close`` re-checks."""
    payload = json.dumps({k: card.get(k) for k in PLAN_SECTIONS}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def minimal_card(card_id: str, created_at: str | None = None) -> dict[str, Any]:
    """Every pre-launch section present, every required leaf ``None``: the file ``new`` writes."""
    return {
        "schema_version": CARD_SCHEMA,
        "card_id": card_id,
        "created_at": created_at or now(),
        "situation": {"elapsed_h": None, "remaining_h": None, "incumbent": None, "trigger": None,
                      "trigger_evidence": [], "alternatives_rejected": [], "pitfalls_hit": [],
                      "smoke_runs": []},
        "problem": {"statement": None, "evidence": [], "affected_share": None,
                    "failure_examples": [], "watch_set": None},
        "hypothesis": {"claim": None, "mechanism": None,
                       "expected_effect": {"metric": None, "direction": None, "against": None,
                                           "magnitude": None},
                       "falsified_if": None},
        "setup": {"parent_checkpoint": {"path": None, "origin": None, "hash": None},
                  "base_model": None, "data": [],
                  "method": {"family": None, "framework": None, "peft": None, "hyperparams": {},
                             "target_format": None, "stop_token": None, "answer_marker": None},
                  "command": {"argv": [], "cwd": None, "script": None, "env": {}, "log": None},
                  "output_dir": None,
                  "checkpoints": {"every_steps": None, "keep": None},
                  "budget": {"gpu": None, "planned_h": None}},
        "evaluation": {"protocol": {"command": [], "dev_set": None, "n": None, "seed": None},
                       "comparator": {"ref": None, "value": None, "path": None},
                       "diagnostic": None},
    }


# ----------------------------------------------------------------- helpers

def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _inside(path: Any, root: Path | None) -> bool:
    if root is None or not isinstance(path, str) or not path.startswith("/"):
        return True
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _require(report: Report, card: dict[str, Any], dotted: str, kind: str = "str") -> Any:
    value = get(card, dotted)
    missing = value is None or (kind in ("str", "list") and not value)
    if missing:
        report.error(dotted, "required")
        return None
    if kind == "str" and not isinstance(value, str):
        report.error(dotted, "must be a string")
        return None
    if kind == "list" and not isinstance(value, list):
        report.error(dotted, "must be a list")
        return None
    if kind == "num" and not _is_num(value):
        report.error(dotted, "must be a number")
        return None
    if kind == "int" and not (isinstance(value, int) and not isinstance(value, bool)):
        report.error(dotted, "must be an integer")
        return None
    return value


# ----------------------------------------------------------- validate_plan

def validate_plan(card: dict[str, Any], session_dir: Path | None = None) -> Report:
    """Check sections 0–4. Existence of files is preflight's job; shape and consistency are ours."""
    r = Report()
    if card.get("schema_version") != CARD_SCHEMA:
        r.error("schema_version", f"must be {CARD_SCHEMA}")
    card_id = card.get("card_id")
    if not isinstance(card_id, str) or not CARD_ID_RE.match(card_id):
        r.error("card_id", "must look like exp-NN")
    for section in PLAN_SECTIONS:
        if not isinstance(card.get(section), dict):
            r.error(section, "section missing")
    if r.errors:
        return r

    # 0. situation
    _require(r, card, "situation.trigger")
    elapsed = _require(r, card, "situation.elapsed_h", "num")
    if elapsed is not None and elapsed < 0:
        r.error("situation.elapsed_h", "must be >= 0")
    remaining = get(card, "situation.remaining_h")
    if remaining is not None and not _is_num(remaining):
        r.error("situation.remaining_h", "must be a number")
    for i, alt in enumerate(get(card, "situation.alternatives_rejected") or []):
        if not isinstance(alt, dict) or not alt.get("option") or not alt.get("reason"):
            r.error(f"situation.alternatives_rejected[{i}]", "needs option and reason")
    for i, hit in enumerate(get(card, "situation.pitfalls_hit") or []):
        if not isinstance(hit, dict) or not hit.get("what"):
            r.error(f"situation.pitfalls_hit[{i}]", "needs what")
        elif hit.get("cost_h") is not None and not _is_num(hit["cost_h"]):
            r.error(f"situation.pitfalls_hit[{i}].cost_h", "must be a number")
    if not get(card, "situation.alternatives_rejected"):
        r.warn("situation.alternatives_rejected", "no alternatives recorded; was this the only option?")

    # 1. problem
    _require(r, card, "problem.statement")
    for i, ev in enumerate(get(card, "problem.evidence") or []):
        if not isinstance(ev, dict) or not ev.get("path") or not ev.get("locator"):
            r.error(f"problem.evidence[{i}]", "needs path and locator")
        elif not _inside(ev["path"], session_dir):
            r.warn(f"problem.evidence[{i}].path", "outside the session dir")
    examples = get(card, "problem.failure_examples") or []
    if examples and not 3 <= len(examples) <= 10:
        r.warn("problem.failure_examples", f"{len(examples)} given; 3–10 is the convention")
    watch = get(card, "problem.watch_set")
    if watch is not None and (not isinstance(watch, dict) or not watch.get("path") or not watch.get("n")):
        r.error("problem.watch_set", "needs path and n")

    # 2. hypothesis
    claim = _require(r, card, "hypothesis.claim")
    if claim and TARGET_RE.search(claim):
        r.warn("hypothesis.claim", "reads as a score target, not a hypothesis")
    direction = get(card, "hypothesis.expected_effect.direction")
    if direction is not None and direction not in DIRECTIONS:
        r.error("hypothesis.expected_effect.direction", f"must be one of {DIRECTIONS}")
    if not get(card, "hypothesis.falsified_if"):
        r.warn("hypothesis.falsified_if", "not stated; what would prove this wrong?")

    # 3. setup
    _require(r, card, "setup.parent_checkpoint.path")
    origin = _require(r, card, "setup.parent_checkpoint.origin")
    if origin and origin != "base_model" and not CARD_ID_RE.match(origin):
        r.error("setup.parent_checkpoint.origin", "must be base_model or a card id")
    data = _require(r, card, "setup.data", "list")
    for i, d in enumerate(data or []):
        where = f"setup.data[{i}]"
        if not isinstance(d, dict):
            r.error(where, "must be a mapping")
            continue
        for key in ("path", "source"):
            if not d.get(key):
                r.error(f"{where}.{key}", "required")
        n = d.get("n_examples")
        if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
            r.error(f"{where}.n_examples", "must be a positive integer")
        cc = d.get("contamination_check")
        if cc is not None and cc not in CONTAMINATION:
            r.error(f"{where}.contamination_check", f"must be one of {CONTAMINATION}")
        if cc == "failed":
            r.error(f"{where}.contamination_check", "failed: this data may not be trained on")
        if d.get("path") and not _inside(d["path"], session_dir):
            r.warn(f"{where}.path", "outside the session dir")
    family = _require(r, card, "setup.method.family")
    if family and family not in METHOD_FAMILIES:
        r.error("setup.method.family", f"must be one of {METHOD_FAMILIES}")
    argv = _require(r, card, "setup.command.argv", "list")
    if argv and not all(isinstance(a, str) and a for a in argv):
        r.error("setup.command.argv", "must be a list of non-empty strings")
    cwd = _require(r, card, "setup.command.cwd")
    if cwd and not _inside(cwd, session_dir):
        r.warn("setup.command.cwd", "outside the session dir")
    out = _require(r, card, "setup.output_dir")
    if out and not _inside(out, session_dir):
        r.warn("setup.output_dir", "outside the session dir")
    keep = get(card, "setup.checkpoints.keep")
    if keep is None:
        r.error("setup.checkpoints.keep", "required: all | last | best | <positive int>")
    elif not (keep in KEEP_POLICIES or (isinstance(keep, int) and not isinstance(keep, bool) and keep > 0)):
        r.error("setup.checkpoints.keep", "must be all | last | best | <positive int>")

    # 4. evaluation
    n = _require(r, card, "evaluation.protocol.n", "int")
    if n is not None and n <= 0:
        r.error("evaluation.protocol.n", "must be positive")
    comp = get(card, "evaluation.comparator") or {}
    if isinstance(comp, dict) and comp.get("value") is not None and not comp.get("path"):
        r.error("evaluation.comparator.path", "a comparator value needs the path of the eval it came from")
    return r


# --------------------------------------------------------- validate_result

def validate_result(card: dict[str, Any]) -> Report:
    """Check sections 5–6 against each other and against what section 4 promised."""
    r = Report()
    for section in RESULT_SECTIONS:
        if not isinstance(card.get(section), dict):
            r.error(section, "section missing")
    if r.errors:
        return r
    execution = _require(r, card, "result.execution")
    if execution and execution not in EXECUTIONS:
        r.error("result.execution", f"must be one of {EXECUTIONS}")
    measurements = get(card, "result.measurements") or []
    if not isinstance(measurements, list):
        r.error("result.measurements", "must be a list")
        measurements = []
    for i, m in enumerate(measurements):
        for key in ("metric", "value", "n", "path"):
            if not isinstance(m, dict) or m.get(key) is None:
                r.error(f"result.measurements[{i}].{key}", "required")
    for i, ck in enumerate(get(card, "result.checkpoints_kept") or []):
        if not isinstance(ck, dict) or not ck.get("path"):
            r.error(f"result.checkpoints_kept[{i}]", "needs path")

    verdict = _require(r, card, "conclusion.verdict")
    if verdict and verdict not in VERDICTS:
        r.error("conclusion.verdict", f"must be one of {VERDICTS}")
    mv = _require(r, card, "conclusion.mechanism_verdict")
    if mv and mv not in MECHANISM_VERDICTS:
        r.error("conclusion.mechanism_verdict", f"must be one of {MECHANISM_VERDICTS}")
    _require(r, card, "conclusion.summary")
    decision = _require(r, card, "conclusion.decision")
    if decision and decision not in DECISIONS:
        r.error("conclusion.decision", f"must be one of {DECISIONS}")

    if verdict in ("supported", "contradicted") and not measurements:
        r.error("conclusion.verdict", f"{verdict} needs at least one measurement under the protocol")
    if mv in ("supported", "contradicted") and get(card, "result.diagnostic_result.value") is None:
        r.error("conclusion.mechanism_verdict", "only a diagnostic result can support or contradict a mechanism")
    if decision == "adopt" and not get(card, "result.output_checkpoint"):
        r.error("result.output_checkpoint", "adopt needs the checkpoint that becomes the incumbent")
    if decision == "iterate" and not get(card, "conclusion.next_step"):
        r.warn("conclusion.next_step", "iterate: name the change in next_step")
    return r
