"""Verdict v1: shape, validation, truth extraction, and scoring.

A verdict is one JSON file beside the card, ``exp-NN.verdict.json``. Four
levels, each with its own answer and confidence, each basis entry naming an
evidence id. ``truth_from_card`` reads what actually happened out of a
closed card; ``score`` holds each level to it. Nothing here calls a model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from awm.exp_protocol.schema import Report, get, now

VERDICT_SCHEMA = "awm-wma-verdict-v1"
LEVELS = ("L0_runs", "L1_valid", "L2_effect", "L3_worth_now")
YES_NO = ("yes", "no", "unknown")
L3_ANSWERS = ("yes", "no", "defer")
MODES = ("offline", "online")
PROBE_KINDS = ("static_check", "unit_test", "data_probe", "dry_run", "sample_probe")
CHANGED = ("L0", "L1", "L2", "L3", "none")
#: L2 direction: flat is for a card that expects no change (a packaging or baseline card).
DIRECTIONS = (None, "higher", "lower", "flat")
#: An unusable verdict is moved to <verdict>.rejected (never *.json, so the ledger ignores it).
REJECTED_SUFFIX = ".rejected"
#: L0 = "did it run": a run killed by the deadline ran; a failed or never-launched one did not.
RAN = ("completed", "killed")
WORTH = {"adopt": True, "reject": False, "abandon_line": False}


TAG_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
VERDICT_FILE_RE = re.compile(r"^(exp-\d+)\.verdict(?:\.([A-Za-z0-9_-]+))?\.json$")


def verdict_path(card_path: Path, tag: str | None = None) -> Path:
    """``exp-NN.verdict.json``, or ``exp-NN.verdict.<tag>.json`` when several agents review one card."""
    card_path = Path(card_path)
    if tag is not None:
        if not TAG_RE.match(tag):
            raise ValueError(f"tag must match {TAG_RE.pattern}, got {tag!r}")
        return card_path.with_name(f"{card_path.stem}.verdict.{tag}.json")
    return card_path.with_name(card_path.stem + ".verdict.json")


def card_path_for(verdict_path: Path) -> Path:
    """The card a verdict file belongs to, tagged or not."""
    verdict_path = Path(verdict_path)
    m = VERDICT_FILE_RE.match(verdict_path.name)
    if not m:
        raise ValueError(f"{verdict_path.name} is not a verdict file name")
    return verdict_path.with_name(m.group(1) + ".yaml")


def load_verdict(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def dump_verdict(path: Path, verdict: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2) + "\n")


def reject_verdict(path: Path, reason: str, **measured: Any) -> Path:
    """Move an unusable verdict file aside so the card counts as unreviewed, keeping the text and what it cost.

    A rejected file is not a verdict: the ledger never reads it, a replay pass retries the sample, and the
    round's spend can still be added up from it.
    """
    path = Path(path)
    target = path.with_name(path.name + REJECTED_SUFFIX)
    n = 1
    while target.exists():
        target = path.with_name(f"{path.name}{REJECTED_SUFFIX}.{n}")
        n += 1
    raw = path.read_text()
    try:
        body: dict[str, Any] = {"verdict": json.loads(raw)}
    except ValueError:
        body = {"raw": raw}
    body["rejected"] = {"reason": reason, "at": now(), **measured}
    target.write_text(json.dumps(body, indent=2) + "\n")
    path.unlink()
    return target


def skill_sha(skill_dir: Path) -> str:
    """Twelve hex chars of the skill's SKILL.md; the ledger groups by it."""
    return hashlib.sha256((Path(skill_dir) / "SKILL.md").read_bytes()).hexdigest()[:12]


def empty_verdict(card_id: str) -> dict[str, Any]:
    return {
        "schema_version": VERDICT_SCHEMA, "card_id": card_id, "wma_skill": "", "backend": "",
        "mode": "offline", "issued_at": now(),
        "levels": {
            "L0_runs": {"answer": "unknown", "confidence": 0.0, "basis": []},
            "L1_valid": {"answer": "unknown", "confidence": 0.0, "basis": []},
            "L2_effect": {"metric": None, "direction": None, "interval": None, "confidence": 0.0, "basis": []},
            "L3_worth_now": {"answer": "defer", "confidence": 0.0, "expected_cost_h": None, "basis": []},
        },
        "evidence": [], "probes": [], "suggestions": {"preconditions": [], "cheaper_variants": []},
        "cost": {"cpu_min": 0, "gpu_min": 0, "wall_min": 0},
    }


# --------------------------------------------------------------- validate

def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_verdict(v: dict[str, Any]) -> Report:
    r = Report()
    if v.get("schema_version") != VERDICT_SCHEMA:
        r.error("schema_version", f"must be {VERDICT_SCHEMA}")
    if not v.get("card_id"):
        r.error("card_id", "required")
    if v.get("mode") not in MODES:
        r.error("mode", f"must be one of {MODES}")
    levels = v.get("levels")
    if not isinstance(levels, dict):
        r.error("levels", "required")
        return r
    evidence = v.get("evidence") or []
    ids = {e.get("id") for e in evidence if isinstance(e, dict)}
    for i, e in enumerate(evidence):
        if not isinstance(e, dict) or not e.get("id") or not e.get("path"):
            r.error(f"evidence[{i}]", "needs id and path")
    for name in LEVELS:
        lv = levels.get(name)
        where = f"levels.{name}"
        if not isinstance(lv, dict):
            r.error(where, "required")
            continue
        conf = lv.get("confidence")
        if not _num(conf) or not 0 <= conf <= 1:
            r.error(f"{where}.confidence", "must be a number in [0, 1]")
        basis = lv.get("basis")
        if not isinstance(basis, list):
            r.error(f"{where}.basis", "must be a list of evidence ids")
        elif any(b not in ids for b in basis):
            r.error(f"{where}.basis", "names an evidence id that is not in evidence[]")
        if name == "L2_effect":
            iv = lv.get("interval")
            if iv is not None and not (isinstance(iv, list) and len(iv) == 2 and all(_num(x) for x in iv)
                                       and iv[0] <= iv[1]):
                r.error(f"{where}.interval", "must be [lo, hi] with lo <= hi, or null")
            if lv.get("direction") not in DIRECTIONS:
                r.error(f"{where}.direction", "must be higher | lower | flat | null")
        elif name == "L3_worth_now":
            if lv.get("answer") not in L3_ANSWERS:
                r.error(f"{where}.answer", f"must be one of {L3_ANSWERS}")
        else:
            if lv.get("answer") not in YES_NO:
                r.error(f"{where}.answer", f"must be one of {YES_NO}")
    for i, p in enumerate(v.get("probes") or []):
        if not isinstance(p, dict) or p.get("kind") not in PROBE_KINDS:
            r.error(f"probes[{i}].kind", f"must be one of {PROBE_KINDS}")
        elif p.get("changed") not in CHANGED:
            r.error(f"probes[{i}].changed", f"must be one of {CHANGED}")
    sug = v.get("suggestions")
    if not isinstance(sug, dict) or not all(isinstance(sug.get(k), list) for k in ("preconditions", "cheaper_variants")):
        r.error("suggestions", "needs preconditions[] and cheaper_variants[]")
    cost = v.get("cost")
    if not isinstance(cost, dict) or not all(_num(cost.get(k, 0)) for k in ("cpu_min", "gpu_min", "wall_min")):
        r.error("cost", "cpu_min / gpu_min / wall_min must be numbers")
    return r


# ------------------------------------------------------------------ truth

def truth_from_card(card: dict[str, Any]) -> dict[str, Any]:
    """What actually happened, read from a closed card's sections 5-6. Open card → all None."""
    ms = [m for m in (get(card, "result.measurements") or []) if isinstance(m, dict)]
    delta = None
    if ms:
        d = ms[0].get("delta_vs_comparator")
        if _num(d):
            delta = float(d)
        else:
            comp = get(card, "evaluation.comparator.value")
            if _num(ms[0].get("value")) and _num(comp):
                delta = float(ms[0]["value"]) - float(comp)
    return {
        "execution": get(card, "result.execution"),
        "output_checkpoint": get(card, "result.output_checkpoint"),
        "measurements": ms,
        "decision": get(card, "conclusion.decision"),
        "wall_h": get(card, "result.wall_h"),
        "delta": delta,
    }


# ------------------------------------------------------------------ score

def truth_levels(truth: dict[str, Any]) -> dict[str, bool | None]:
    """What L0 and L1 actually were: ran (completed or killed) / yielded a scorable candidate. None = open."""
    execution = truth.get("execution")
    if execution is None:
        return {"L0": None, "L1": None}
    return {"L0": execution in RAN,
            "L1": execution == "completed" and bool(truth.get("output_checkpoint")) and bool(truth.get("measurements"))}


def score(verdict: dict[str, Any], truth: dict[str, Any]) -> dict[str, str]:
    lv = verdict["levels"]
    out: dict[str, str] = {}

    def yes_no(level: str, actual: bool | None) -> str:
        ans = lv[level].get("answer")
        if ans not in ("yes", "no") or actual is None:
            return "unscorable"
        return "hit" if (ans == "yes") == actual else "miss"

    actual = truth_levels(truth)
    out["L0"] = yes_no("L0_runs", actual["L0"])
    out["L1"] = yes_no("L1_valid", actual["L1"])

    iv = lv["L2_effect"].get("interval")
    delta = truth.get("delta")
    if iv is None or delta is None:
        out["L2"] = "unscorable"
    elif delta < iv[0]:
        out["L2"] = "below"
    elif delta > iv[1]:
        out["L2"] = "above"
    else:
        out["L2"] = "in_interval"

    ans3 = lv["L3_worth_now"].get("answer")
    worth = WORTH.get(truth.get("decision"))
    if worth is None or ans3 not in L3_ANSWERS:
        out["L3"] = "unscorable"
    else:
        said_worth = ans3 == "yes"          # defer counts as "not now"
        out["L3"] = "hit" if said_worth == worth else "miss"
    return out
