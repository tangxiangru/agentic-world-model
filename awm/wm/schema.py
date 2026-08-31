"""Schemas, validation, and file helpers for the world-model runtime.

Everything the runtime persists is YAML or JSON with a ``schema_version``.
Validation here is deliberately strict about the things the spec makes
load-bearing — paths inside the session directory, evidence that resolves,
enumerations — and permissive about free text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CARD_SCHEMA = "awm-experiment-card-v1"
CONTRACT_SCHEMA = "awm-evaluation-contract-v1"
RULE_SCHEMA = "awm-rule-v1"
PING_SCHEMA = "awm-ping-v1"
REPLY_SCHEMA = "awm-reply-v1"
STATE_SCHEMA = "awm-wm-state-v1"
OBSERVATION_SCHEMA = "awm-observation-v1"
SEAL_SCHEMA = "awm-seal-v1"
CONFIG_SCHEMA = "awm-wm-config-v1"

CARD_STATES = ("draft", "frozen", "running", "awaiting_review", "closed")
PING_KINDS = ("brief", "notice", "yield_request", "decision")
BRIEF_OPTIONS = ("accept", "amend", "override", "withdraw")
YIELD_OPTIONS = ("accept", "reject")
DECISION_OPTIONS = ("continue", "more_eval", "abort")  # plus select:<obs-id>
CARD_DECISIONS = ("adopt", "reject", "iterate", "abandon_line")
EXECUTIONS = ("completed", "failed", "killed", "not_run")
VERDICTS = ("supported", "contradicted", "inconclusive")
METHOD_FAMILIES = ("sft", "rft", "dpo", "grpo", "distill", "merge", "decode-config", "other")
ARMS = ("null", "retrieval", "llm", "predictor")

HOOK_CONTINUE = 0
HOOK_YIELD = 3
HOOK_ABORT = 4


class WMError(ValueError):
    """A card, reply, or runtime call violated the protocol."""


# ---------------------------------------------------------------- utilities

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WMError(f"missing {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise WMError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise WMError(f"{path}: top level must be a mapping")
    return value


def dump_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100))
    os.replace(tmp, path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise WMError(f"missing {path}")
    return json.loads(path.read_text())


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def count_jsonl(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _req(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, "", [], {}):
        raise WMError(f"{where}.{key} is required")
    return mapping[key]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WMError(f"{where} must be a mapping")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise WMError(f"{where} must be a list")
    return value


CARD_ID_RE = re.compile(r"^exp-\d{2,}$")


# ---------------------------------------------------------------- card

def validate_card(card: dict[str, Any], session_dir: Path) -> list[dict[str, Any]]:
    """Validate sections 1-4 of a card and run the mechanical grounding checks.

    Returns the grounding report: one entry per check with ``passed`` and
    ``detail``. Raises ``WMError`` on schema violations or any failed check
    that the spec says ``propose`` must reject.
    """
    where = "card"
    if card.get("schema_version") != CARD_SCHEMA:
        raise WMError(f"{where}.schema_version must be {CARD_SCHEMA}")
    card_id = str(_req(card, "card_id", where))
    if not CARD_ID_RE.match(card_id):
        raise WMError(f"{where}.card_id must look like exp-NN, got {card_id!r}")
    for section in ("problem", "hypothesis", "setup", "evaluation"):
        _mapping(_req(card, section, where), f"{where}.{section}")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, fatal: bool = True) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if fatal and not passed:
            raise WMError(f"grounding check failed: {name}: {detail}")

    # problem
    problem = card["problem"]
    _req(problem, "statement", "problem")
    evidence = _list(_req(problem, "evidence", "problem"), "problem.evidence")
    for i, ev in enumerate(evidence):
        ev = _mapping(ev, f"problem.evidence[{i}]")
        p = Path(str(_req(ev, "path", f"problem.evidence[{i}]")))
        _req(ev, "locator", f"problem.evidence[{i}]")
        check(f"evidence[{i}].path", p.is_file() and inside(p, session_dir),
              f"{p} {'exists inside session dir' if p.is_file() and inside(p, session_dir) else 'missing or outside session dir'}")
    examples = _list(_req(problem, "failure_examples", "problem"), "problem.failure_examples")
    check("failure_examples.count", 3 <= len(examples) <= 10, f"{len(examples)} examples (need 3-10)")
    for i, ex in enumerate(examples):
        ex = _mapping(ex, f"problem.failure_examples[{i}]")
        for key in ("id", "source", "question", "gold", "model_output", "failure"):
            _req(ex, key, f"problem.failure_examples[{i}]")
        src = Path(str(ex["source"]))
        check(f"failure_examples[{i}].source", src.is_file() and inside(src, session_dir), str(src))
    watch = _mapping(_req(problem, "watch_set", "problem"), "problem.watch_set")
    wpath = Path(str(_req(watch, "path", "problem.watch_set")))
    wn = int(_req(watch, "n", "problem.watch_set"))
    check("watch_set.path", wpath.is_file() and inside(wpath, session_dir), str(wpath))
    actual = count_jsonl(wpath)
    check("watch_set.count", actual == wn, f"file has {actual} items, card says {wn}")
    ids = {str(r.get("id")) for r in read_jsonl(wpath)}
    check("watch_set.ids", all("id" in r for r in read_jsonl(wpath)), "every watch item has an id")
    check("failure_examples.in_watch_set",
          all(str(ex["id"]) in ids for ex in examples),
          "every failure example id is in the watch set", fatal=False)

    # hypothesis
    hyp = card["hypothesis"]
    for key in ("claim", "mechanism", "falsified_if"):
        _req(hyp, key, "hypothesis")
    eff = _mapping(_req(hyp, "expected_effect", "hypothesis"), "hypothesis.expected_effect")
    for key in ("metric", "direction", "against"):
        _req(eff, key, "hypothesis.expected_effect")
    if eff["direction"] not in ("higher", "lower"):
        raise WMError("hypothesis.expected_effect.direction must be higher|lower")
    if re.search(r"\b(reach|hit|achieve|target)\b.*\d", str(hyp["claim"]), re.IGNORECASE):
        check("hypothesis.not_a_target", False, "claim reads as a score target, not a hypothesis", fatal=False)

    # setup
    setup = card["setup"]
    parent = _mapping(_req(setup, "parent_checkpoint", "setup"), "setup.parent_checkpoint")
    ppath = Path(str(_req(parent, "path", "setup.parent_checkpoint")))
    _req(parent, "origin", "setup.parent_checkpoint")
    check("parent_checkpoint.path", ppath.is_dir(), f"{ppath} is a directory")
    for i, d in enumerate(_list(_req(setup, "data", "setup"), "setup.data")):
        d = _mapping(d, f"setup.data[{i}]")
        for key in ("path", "source", "n_examples", "selection", "contamination_check"):
            _req(d, key, f"setup.data[{i}]")
        dpath = Path(str(d["path"]))
        check(f"data[{i}].path", dpath.exists() and inside(dpath, session_dir), str(dpath))
        check(f"data[{i}].contamination", d["contamination_check"] != "failed",
              f"contamination_check={d['contamination_check']}")
    method = _mapping(_req(setup, "method", "setup"), "setup.method")
    if _req(method, "family", "setup.method") not in METHOD_FAMILIES:
        raise WMError(f"setup.method.family must be one of {METHOD_FAMILIES}")
    _mapping(_req(method, "hyperparams", "setup.method"), "setup.method.hyperparams")
    cmd = _mapping(_req(setup, "command", "setup"), "setup.command")
    argv = _list(_req(cmd, "argv", "setup.command"), "setup.command.argv")
    if not all(isinstance(a, str) for a in argv):
        raise WMError("setup.command.argv must be a list of strings")
    cwd = Path(str(_req(cmd, "cwd", "setup.command")))
    check("command.cwd", cwd.is_dir() and inside(cwd, session_dir), str(cwd))
    log = Path(str(_req(cmd, "log", "setup.command")))
    check("command.log", inside(log, session_dir), f"{log} inside session dir")
    out_dir = Path(str(_req(setup, "output_dir", "setup")))
    check("output_dir", inside(out_dir, session_dir), f"{out_dir} inside session dir")
    resume = setup.get("resume_argv")
    if resume is not None:
        resume = _list(resume, "setup.resume_argv")
        check("resume_argv.placeholder", any("{checkpoint}" in str(a) for a in resume),
              "resume_argv contains the {checkpoint} placeholder")
    progress = _mapping(_req(setup, "progress", "setup"), "setup.progress")
    _req(progress, "unit", "setup.progress")
    total = _req(progress, "total", "setup.progress")
    check("progress.total", isinstance(total, int) and total > 0, f"total={total}")

    # evaluation
    ev = card["evaluation"]
    proto = _mapping(_req(ev, "protocol", "evaluation"), "evaluation.protocol")
    n = _req(proto, "n", "evaluation.protocol")
    check("evaluation.n", isinstance(n, int) and n > 0, f"n={n}")
    _req(proto, "dev_set", "evaluation.protocol")
    comp = _mapping(_req(ev, "comparator", "evaluation"), "evaluation.comparator")
    _req(comp, "ref", "evaluation.comparator")
    return checks


PLAN_SECTIONS = ("problem", "hypothesis", "setup", "evaluation")


def plan_hash(card: dict[str, Any]) -> str:
    """Hash of sections 1-4 only; what freeze pins and finalize re-checks."""
    return sha256_obj({k: card.get(k) for k in PLAN_SECTIONS})


def validate_result(result: dict[str, Any], card_id: str) -> None:
    """Validate sections 5-6 of a completed card (same file as sections 1-4)."""
    if result.get("schema_version") != CARD_SCHEMA:
        raise WMError(f"schema_version must be {CARD_SCHEMA}")
    if result.get("card_id") != card_id:
        raise WMError(f"card_id must be {card_id}")
    res = _mapping(_req(result, "result", "result"), "result.result")
    if _req(res, "execution", "result.result") not in EXECUTIONS:
        raise WMError(f"result.result.execution must be one of {EXECUTIONS}")
    _list(res.get("measurements", []), "result.result.measurements")
    for i, m in enumerate(res.get("measurements", [])):
        m = _mapping(m, f"result.measurements[{i}]")
        for key in ("metric", "value", "n", "path"):
            _req(m, key, f"result.measurements[{i}]")
    con = _mapping(_req(result, "conclusion", "result"), "result.conclusion")
    if _req(con, "verdict", "result.conclusion") not in VERDICTS:
        raise WMError(f"conclusion.verdict must be one of {VERDICTS}")
    if _req(con, "decision", "result.conclusion") not in CARD_DECISIONS:
        raise WMError(f"conclusion.decision must be one of {CARD_DECISIONS}")
    _req(con, "summary", "result.conclusion")
    if con["verdict"] in ("supported", "contradicted") and not res.get("measurements"):
        raise WMError("supported/contradicted needs at least one measurement")


# ---------------------------------------------------------------- contract

def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise WMError(f"contract.schema_version must be {CONTRACT_SCHEMA}")
    evaluators = _list(_req(contract, "evaluators", "contract"), "contract.evaluators")
    names = set()
    for i, e in enumerate(evaluators):
        e = _mapping(e, f"contract.evaluators[{i}]")
        for key in ("name", "kind", "metric", "direction", "n"):
            _req(e, key, f"contract.evaluators[{i}]")
        if e["kind"] not in ("official", "custom"):
            raise WMError("evaluator.kind must be official|custom")
        if e["kind"] == "custom":
            _req(e, "items", f"contract.evaluators[{i}]")
        if e["name"] in names:
            raise WMError(f"duplicate evaluator name {e['name']}")
        names.add(e["name"])
    sy = _mapping(_req(contract, "standing_yields", "contract"), "contract.standing_yields")
    cadence = _mapping(_req(sy, "cadence", "contract.standing_yields"), "contract.standing_yields.cadence")
    if cadence.get("kind") != "fractions":
        raise WMError("only cadence.kind: fractions is supported")
    vals = _list(_req(cadence, "values", "cadence"), "cadence.values")
    if any(not (0 < float(v) <= 1.0) for v in vals):
        raise WMError("cadence.values must be in (0, 1]")
    for name in _list(_req(sy, "evaluators", "contract.standing_yields"), "standing_yields.evaluators"):
        if name not in names:
            raise WMError(f"standing_yields names unknown evaluator {name}")
    for name in _list(contract.get("on_request", []), "contract.on_request"):
        if name not in names:
            raise WMError(f"on_request names unknown evaluator {name}")
    if contract.get("rule_schema", RULE_SCHEMA) != RULE_SCHEMA:
        raise WMError(f"rule_schema must be {RULE_SCHEMA}")
    for i, r in enumerate(_list(contract.get("rules", []), "contract.rules")):
        r = _mapping(r, f"contract.rules[{i}]")
        for key in ("id", "op", "evaluator", "field", "comparator", "threshold", "count", "action"):
            _req(r, key, f"contract.rules[{i}]")
        if r["op"] != "consecutive_threshold":
            raise WMError("only op: consecutive_threshold is supported")
        if r["comparator"] not in ("lt", "gt", "le", "ge"):
            raise WMError("rule.comparator must be lt|gt|le|ge")
        if r["action"] not in ("abort", "select_best", "continue"):
            raise WMError("rule.action must be abort|select_best|continue")
        if r["evaluator"] not in names:
            raise WMError(f"rule {r['id']} names unknown evaluator {r['evaluator']}")
    sel = _mapping(_req(contract, "selection", "contract"), "contract.selection")
    if _req(sel, "evaluator", "contract.selection") not in names:
        raise WMError("selection.evaluator unknown")
    if sel.get("completion_policy", "best_observation") != "best_observation":
        raise WMError("only completion_policy: best_observation is supported")


def evaluator_by_name(contract: dict[str, Any], name: str) -> dict[str, Any]:
    for e in contract["evaluators"]:
        if e["name"] == name:
            return e
    raise WMError(f"unknown evaluator {name}")
