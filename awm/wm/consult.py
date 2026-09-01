"""The single API between the two agents: ``consult``.

The scientist messages the world-model agent (WMA) at any time with whatever
it has — a plan, a plan plus results, a question. The WMA answers with one
response of fixed shape, updated on every consult:

    card         its structured understanding of the experiment (awm-experiment-card-v1, sections 1-4)
    verdict      SURE_WONT_WORK | SURE_WILL_WORK | CANNOT_DECIDE, with confidence and a predicted
                 final result (delta vs the parent, with a spread) and the past experiments it rests on
    eval_plan    when the scientist should evaluate, with what protocol, and the number to beat
    suggestion   TERMINATE | KEEP_RUNNING | ADJUST, with the reason
    reasons      short, every claim citing a file

This module holds the schema, its validator, the default evaluation plan every
arm falls back to, and the ledger every consult is appended to. The reasoning
itself happens in the WMA's own Claude Code session (``wma/CLAUDE.md``).
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from .schema import CARD_SCHEMA, WMError, dump_json, load_json, now

RESPONSE_SCHEMA = "awm-consult-response-v1"
VERDICTS = ("SURE_WONT_WORK", "SURE_WILL_WORK", "CANNOT_DECIDE")
SUGGESTIONS = ("TERMINATE", "KEEP_RUNNING", "ADJUST")
STAGES = ("plan", "running", "shipped")

# Confidence at which CANNOT_DECIDE becomes a SURE_* verdict. Fixed across arms so the
# arms differ in evidence, not in nerve.
SURE_THRESHOLD = 0.75


def default_eval_plan(total_steps: int | None, n: int = 150, parent_value: float | None = None,
                      fractions: tuple[float, ...] = (0.25, 0.5, 0.75)) -> dict[str, Any]:
    """What every arm recommends when the corpus has nothing to say: look at 25/50/75 %."""
    points = [{"step": round(total_steps * f), "fraction": f} for f in fractions] if total_steps else []
    return {
        "points": points,
        "protocol": {"command": ["python", "evaluate.py", "--model-path", "<checkpoint>", "--limit", str(n)], "n": n},
        "comparator": {"ref": "parent", "value": parent_value,
                       "note": "evaluate the checkpoint you start from with the same --limit first; that is the number to beat"},
        "number_to_beat": None,
        "basis": "default schedule (no comparable past experiments)" if not total_steps else
                 "default schedule at 25/50/75 % of the planned steps",
    }


def validate_response(resp: dict[str, Any]) -> list[str]:
    """Return the list of problems; empty means the response follows the contract."""
    problems: list[str] = []
    if resp.get("schema_version") != RESPONSE_SCHEMA:
        problems.append(f"schema_version must be {RESPONSE_SCHEMA}")
    if resp.get("stage") not in STAGES:
        problems.append(f"stage must be one of {STAGES}")
    card = resp.get("card")
    if not isinstance(card, dict):
        problems.append("card must be a mapping")
    else:
        if card.get("schema_version") != CARD_SCHEMA:
            problems.append(f"card.schema_version must be {CARD_SCHEMA}")
        for sec in ("problem", "hypothesis", "setup", "evaluation"):
            if not isinstance(card.get(sec), dict):
                problems.append(f"card.{sec} must be a mapping")
        if not isinstance(card.get("gaps", []), list):
            problems.append("card.gaps must be a list of questions")
    v = resp.get("verdict")
    if not isinstance(v, dict):
        problems.append("verdict must be a mapping")
    else:
        if v.get("label") not in VERDICTS:
            problems.append(f"verdict.label must be one of {VERDICTS}")
        conf = v.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            problems.append("verdict.confidence must be a number in [0, 1]")
        elif v.get("label") in ("SURE_WONT_WORK", "SURE_WILL_WORK") and conf < SURE_THRESHOLD:
            problems.append(f"a SURE_* verdict needs confidence >= {SURE_THRESHOLD}; use CANNOT_DECIDE")
        pred = v.get("prediction")
        if pred is not None:
            if not isinstance(pred, dict):
                problems.append("verdict.prediction must be a mapping or null")
            else:
                for k in ("metric", "delta_mean", "delta_sd"):
                    if k not in pred:
                        problems.append(f"verdict.prediction.{k} missing")
                if isinstance(pred.get("delta_sd"), (int, float)) and pred["delta_sd"] < 0:
                    problems.append("verdict.prediction.delta_sd must be >= 0")
        if not isinstance(v.get("based_on", []), list):
            problems.append("verdict.based_on must be a list of {path, locator, observation}")
        elif v.get("label") in ("SURE_WONT_WORK", "SURE_WILL_WORK") and not v.get("based_on"):
            problems.append("a SURE_* verdict must cite the past experiments it rests on")
    ep = resp.get("eval_plan")
    if not isinstance(ep, dict) or not isinstance(ep.get("points"), list) or not isinstance(ep.get("protocol"), dict):
        problems.append("eval_plan must have points[] and protocol{}")
    sg = resp.get("suggestion")
    if not isinstance(sg, dict) or sg.get("label") not in SUGGESTIONS:
        problems.append(f"suggestion.label must be one of {SUGGESTIONS}")
    elif not sg.get("reason"):
        problems.append("suggestion.reason is required")
    elif sg["label"] == "ADJUST" and not sg.get("change"):
        problems.append("ADJUST needs suggestion.change")
    reasons = resp.get("reasons")
    if not isinstance(reasons, list):
        problems.append("reasons must be a list of {claim, path, locator}")
    else:
        for i, r in enumerate(reasons):
            if not isinstance(r, dict) or not r.get("claim") or not r.get("path"):
                problems.append(f"reasons[{i}] needs claim and path")
    return problems


def lint_citations(resp: dict[str, Any], roots: list[Path]) -> tuple[dict[str, Any], list[str]]:
    """Drop citations whose path does not exist under an allowed root; report what was dropped."""
    dropped: list[str] = []

    def ok(p: Any) -> bool:
        if not isinstance(p, str) or not p:
            return False
        path = Path(p)
        if not path.exists():
            return False
        rp = path.resolve()
        return any(str(rp).startswith(str(Path(r).resolve()) + "/") or rp == Path(r).resolve() for r in roots)

    v = resp.get("verdict") or {}
    kept = []
    for e in v.get("based_on") or []:
        (kept if ok(e.get("path")) else dropped).append(e if ok(e.get("path")) else str(e.get("path")))
    v["based_on"] = kept
    kept = []
    for r in resp.get("reasons") or []:
        (kept if ok(r.get("path")) else dropped).append(r if ok(r.get("path")) else str(r.get("path")))
    resp["reasons"] = kept
    if v.get("label") in ("SURE_WONT_WORK", "SURE_WILL_WORK") and not v["based_on"]:
        v["label"] = "CANNOT_DECIDE"
        v["confidence"] = min(float(v.get("confidence", 0.5)), SURE_THRESHOLD - 0.01)
        v.setdefault("notes", []).append("downgraded: the cited evidence did not resolve under the allowed roots")
    return resp, dropped


class ConsultLedger:
    """``consults.jsonl``: one row per consult — what the scientist asked, what the WMA answered, when."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def append(self, **row: Any) -> dict[str, Any]:
        with self.path.open("a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            seq = sum(1 for line in fh if line.strip()) + 1
            entry = {"seq": seq, "at": now(), **row}
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
        return entry

    def rows(self) -> list[dict[str, Any]]:
        out = []
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def for_card(self, card_id: str) -> list[dict[str, Any]]:
        return [r for r in self.rows() if r.get("card_id") == card_id]


def log_consult(wm_dir: Path, response: dict[str, Any], *, request: str, roots: list[Path],
                arm: str, model: str | None) -> dict[str, Any]:
    """Validate, lint, persist the card and the response, append to the ledger. Returns the entry."""
    problems = validate_response(response)
    if problems:
        raise WMError("consult response does not follow the contract: " + "; ".join(problems))
    response, dropped = lint_citations(response, roots)
    card = response["card"]
    card_id = card.get("card_id") or f"exp-{len({r.get('card_id') for r in ConsultLedger(wm_dir / 'consults.jsonl').rows()}) + 1:02d}"
    card["card_id"] = card_id
    cdir = wm_dir / "cards" / card_id
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(ConsultLedger(wm_dir / "consults.jsonl").for_card(card_id)) + 1
    dump_json(cdir / f"consult-{n:02d}.json", {"request": request, "response": response, "at": now()})
    dump_json(cdir / "card.json", card)
    entry = ConsultLedger(wm_dir / "consults.jsonl").append(
        card_id=card_id, consult_n=n, stage=response["stage"], arm=arm, model=model,
        verdict=response["verdict"]["label"], confidence=response["verdict"].get("confidence"),
        prediction=response["verdict"].get("prediction"), suggestion=response["suggestion"]["label"],
        eval_points=[p.get("step") for p in response["eval_plan"].get("points", [])],
        n_based_on=len(response["verdict"].get("based_on") or []), citations_dropped=len(dropped),
        request_chars=len(request), path=str(cdir / f"consult-{n:02d}.json"))
    return entry


def record_outcome(wm_dir: Path, card_id: str, *, final_value: float | None, shipped: str | None,
                   note: str | None = None) -> dict[str, Any]:
    """The realised result, stored next to the predictions that preceded it."""
    ledger = ConsultLedger(wm_dir / "consults.jsonl")
    preds = [r.get("prediction") for r in ledger.for_card(card_id) if r.get("prediction")]
    entry = ledger.append(card_id=card_id, stage="shipped", event="outcome", final_value=final_value,
                          shipped=shipped, note=note, predictions_made=len(preds),
                          last_prediction=preds[-1] if preds else None)
    cdir = wm_dir / "cards" / card_id
    cdir.mkdir(parents=True, exist_ok=True)
    out = load_json(cdir / "card.json", default={}) if (cdir / "card.json").is_file() else {}
    out["outcome"] = {"final_value": final_value, "shipped": shipped, "note": note, "at": now()}
    dump_json(cdir / "card.json", out)
    return entry
