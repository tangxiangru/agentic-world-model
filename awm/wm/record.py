"""The single API of the recorder mode: ``record``.

In recorder mode the world-model agent gives no advice. The scientist explores
on its own, with no access to past trajectories, and messages the WMA around
each launch: the plan before it runs, results as they arrive, the shipped
checkpoint at the end. The WMA answers with one response of fixed shape:

    card        its structured understanding of the experiment
                (awm-experiment-card-v1), written at launch time — before the
                outcome exists — and extended with results as they are reported
    questions   at most three questions back to the scientist, allowed only to
                close reproducibility gaps, never to steer
    ack         one line saying what was recorded

No verdict, no prediction, no eval plan, no suggestion. A response carrying any
of those keys is rejected. The point of the run is the record: a list of cards
whose lineage resolves to the base model, each sufficient to re-produce its
checkpoint.

**What makes a recipe sufficient** is `SUFFICIENCY`: another agent, given the
cards (and the snapshots under ``wm/cards/<id>/snapshot/``) and nothing else —
not the session transcript — must be able to rerun the path from the base
model and get the same kind of checkpoint and the same kind of measurement.
Concretely, per card: where it started (parent lineage down to the base
model), exactly what data (source + selection rule + count + how it was
built), exactly what method (family, framework versions, the effective
hyperparameters including defaults, seed, precision), the exact launch
command with a snapshot of the script it names, how it was measured (exact
eval command and n), and what happened (execution status, measurements with
value/n/source). Unknown is recorded as a question, never guessed.

This module holds the schema, its validator, the sufficiency checker, and the
ledger every record is appended to. The reasoning itself happens in the WMA's
own Claude Code session (``wma/CLAUDE.md``).
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from .schema import CARD_SCHEMA, WMError, dump_json, load_json, now

RESPONSE_SCHEMA = "awm-record-response-v1"
STAGES = ("plan", "running", "closed")
FORBIDDEN_KEYS = ("verdict", "prediction", "eval_plan", "suggestion", "advice")
MAX_QUESTIONS = 3

# dotted card field -> why the recipe is not reproducible without it.
# `plan`-stage fields must be present when the card is first recorded;
# `result`-stage fields once the scientist has reported results (stage
# running/closed with measurements) or closed the card.
SUFFICIENCY: dict[str, tuple[str, str]] = {
    "setup.parent_checkpoint.path": ("plan", "where the run starts; lineage must resolve to the base model"),
    "setup.parent_checkpoint.origin": ("plan", "base_model or the exp-NN that produced the parent"),
    "setup.data": ("plan", "what the model was trained on: source, selection rule, n, how it was built"),
    "setup.method.family": ("plan", "sft / grpo / dpo / merge / decode-config / ..."),
    "setup.method.framework": ("plan", "trainer and versions actually used"),
    "setup.method.hyperparams": ("plan", "the effective values, defaults included: lr, steps or epochs, batch, seed, precision"),
    "setup.command.argv": ("plan", "the exact launch argv"),
    "setup.command.script": ("plan", "the script the argv names; snapshot it — the scientist edits in place"),
    "evaluation.protocol.command": ("result", "the exact eval command, so the measurement is repeatable"),
    "evaluation.protocol.n": ("result", "how many items the measurement used"),
    "result.execution": ("result", "completed / failed / killed — a crash is an outcome too"),
    "result.measurements": ("result", "value, n, and the eval output file each number came from"),
}


def _get(card: dict[str, Any], dotted: str) -> Any:
    cur: Any = card
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value or all(_empty(v) for v in (value.values() if isinstance(value, dict) else value))
    return False


def check_sufficiency(card: dict[str, Any], stage: str) -> list[str]:
    """Dotted fields the recipe still needs; empty means reproducible at this stage."""
    want_result = stage == "closed" or (stage == "running" and not _empty(_get(card, "result.measurements")))
    missing = []
    for field, (phase, _why) in SUFFICIENCY.items():
        if phase == "result" and not want_result:
            continue
        if _empty(_get(card, field)):
            missing.append(field)
    # a completed run must name the checkpoint it produced — that is the object
    # the post-run sweep evaluates, and the label every card exists to earn
    if want_result and _get(card, "result.execution") == "completed" and _empty(_get(card, "result.output_checkpoint")):
        missing.append("result.output_checkpoint")
    return missing


def validate_response(resp: dict[str, Any]) -> list[str]:
    """Return the list of problems; empty means the response follows the contract."""
    problems: list[str] = []
    if resp.get("schema_version") != RESPONSE_SCHEMA:
        problems.append(f"schema_version must be {RESPONSE_SCHEMA}")
    if resp.get("stage") not in STAGES:
        problems.append(f"stage must be one of {STAGES}")
    for key in FORBIDDEN_KEYS:
        if key in resp or isinstance(resp.get("card"), dict) and key in resp["card"]:
            problems.append(f"'{key}' is advisory content; the recorder never sends it")
    card = resp.get("card")
    if not isinstance(card, dict):
        problems.append("card must be a mapping")
    else:
        if card.get("schema_version") != CARD_SCHEMA:
            problems.append(f"card.schema_version must be {CARD_SCHEMA}")
        for sec in ("problem", "setup", "evaluation"):
            if not isinstance(card.get(sec), dict):
                problems.append(f"card.{sec} must be a mapping")
    questions = resp.get("questions", [])
    if not isinstance(questions, list) or any(not isinstance(q, str) or not q.strip() for q in questions):
        problems.append("questions must be a list of non-empty strings")
    elif len(questions) > MAX_QUESTIONS:
        problems.append(f"at most {MAX_QUESTIONS} questions per record; keep the rest for the next report")
    if not isinstance(resp.get("ack"), str) or not resp["ack"].strip():
        problems.append("ack must be one line saying what was recorded")
    return problems


class RecordLedger:
    """``records.jsonl``: one row per record — what the scientist reported, what was recorded, when."""

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


def log_record(wm_dir: Path, response: dict[str, Any], *, request: str,
               model: str | None) -> dict[str, Any]:
    """Validate, check sufficiency, persist the card and the response, append to the ledger."""
    problems = validate_response(response)
    if problems:
        raise WMError("record response does not follow the contract: " + "; ".join(problems))
    card = response["card"]
    ledger = RecordLedger(wm_dir / "records.jsonl")
    card_id = card.get("card_id") or f"exp-{len({r.get('card_id') for r in ledger.rows()}) + 1:02d}"
    card["card_id"] = card_id
    missing = check_sufficiency(card, response["stage"])
    response["missing"] = missing
    cdir = wm_dir / "cards" / card_id
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(ledger.for_card(card_id)) + 1
    dump_json(cdir / f"record-{n:02d}.json", {"request": request, "response": response, "at": now()})
    dump_json(cdir / "card.json", card)
    entry = ledger.append(
        card_id=card_id, record_n=n, stage=response["stage"], model=model,
        missing=missing, n_questions=len(response.get("questions", [])),
        n_measurements=len(card.get("result", {}).get("measurements") or []) if isinstance(card.get("result"), dict) else 0,
        request_chars=len(request), path=str(cdir / f"record-{n:02d}.json"))
    return entry


HASH_LIMIT_BYTES = 256 * 1024 * 1024  # weight shards above this are recorded by size only


def archive_checkpoint(wm_dir: Path, session_dir: Path, card_id: str, src: Path) -> dict[str, Any]:
    """Preserve a card's checkpoint under ``wm/checkpoints/<card>/`` before the
    scientist overwrites or deletes it.

    Every archived checkpoint gets the official test-set evaluation after the
    run (the run harness sweeps ``wm/checkpoints/*``), which is what turns each
    card into a labelled data point rather than only the shipped one. Copies
    with reflink where the filesystem supports it, byte copy otherwise; writes
    a manifest with the source path and per-file hashes (size-only above
    ``HASH_LIMIT_BYTES``).
    """
    import shutil
    import subprocess

    from .schema import inside, sha256_file

    src = Path(src).resolve()
    if not src.is_dir():
        raise WMError(f"{src} is not a directory")
    if not inside(src, Path(session_dir).resolve()):
        raise WMError(f"{src} is outside the session directory {session_dir}")
    if not (src / "config.json").is_file():
        raise WMError(f"{src} has no config.json; archive the checkpoint directory itself")
    dest = wm_dir / "checkpoints" / card_id
    if dest.exists():
        raise WMError(f"{dest} already exists; one archived checkpoint per card")
    dest.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["cp", "-R", "--reflink=auto", str(src), str(dest)],
                        capture_output=True, text=True).returncode
    if rc != 0:  # BSD cp has no --reflink; fall back to a plain copy
        shutil.copytree(src, dest)
    files = []
    for f in sorted(x for x in dest.rglob("*") if x.is_file()):
        size = f.stat().st_size
        entry: dict[str, Any] = {"path": str(f.relative_to(dest)), "bytes": size}
        if size <= HASH_LIMIT_BYTES:
            entry["sha256"] = sha256_file(f)
        files.append(entry)
    manifest = {"card_id": card_id, "source": str(src), "at": now(), "files": files,
                "bytes_total": sum(f["bytes"] for f in files)}
    dump_json(dest.parent / f"{card_id}.MANIFEST.json", manifest)
    card_file = wm_dir / "cards" / card_id / "card.json"
    if card_file.is_file():
        card = load_json(card_file, default={})
        card.setdefault("result", {})["archived_checkpoint"] = str(dest)
        dump_json(card_file, card)
    return manifest


def record_outcome(wm_dir: Path, card_id: str, *, final_value: float | None, shipped: str | None,
                   note: str | None = None) -> dict[str, Any]:
    """What the scientist shipped and scored, stored on the card it adopted."""
    ledger = RecordLedger(wm_dir / "records.jsonl")
    entry = ledger.append(card_id=card_id, stage="closed", event="outcome",
                          final_value=final_value, shipped=shipped, note=note)
    cdir = wm_dir / "cards" / card_id
    cdir.mkdir(parents=True, exist_ok=True)
    out = load_json(cdir / "card.json", default={}) if (cdir / "card.json").is_file() else {}
    out["outcome"] = {"final_value": final_value, "shipped": shipped, "note": note, "at": now()}
    dump_json(cdir / "card.json", out)
    return entry
