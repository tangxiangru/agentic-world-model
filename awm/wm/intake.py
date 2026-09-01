"""Intake: turn a scientist's plan into an experiment card, asking for what cannot be read.

The scientist does not write cards. It writes a short plan — what it will
train, the launch command, where checkpoints land, how many steps, how to
evaluate — and the world-model agent drafts the card from that plan plus the
workspace. Anything the plan and the workspace do not settle becomes a
*question* back to the scientist; nothing is guessed.

This module is the deterministic core every arm shares: parse the launch
command, look up files, list what is still missing. The autonomous arms call
their model on top of it to read scripts and infer more before asking.
"""

from __future__ import annotations

import contextlib
import re
import shlex
from pathlib import Path
from typing import Any

from .schema import CARD_SCHEMA, count_jsonl, inside, now

# field -> question the scientist gets when the field cannot be determined
REQUIRED: dict[str, str] = {
    "problem.statement": "What problem in the current model is this run meant to address? One or two sentences.",
    "hypothesis.claim": "What do you expect this run to change, and against what (the base model, or a previous run)? One sentence.",
    "setup.command.argv": "What is the exact launch command (full argv, as you will run it)?",
    "setup.output_dir": "Where will the trainer save checkpoints (the output directory, absolute path inside your task dir)?",
    "setup.progress.total": "How many optimizer steps is the run planned for (the number the checkpoint fractions are taken from)?",
    "setup.resume_argv": "Does the command accept --resume_from_checkpoint <path> to resume? Answer `yes`, or give the exact resume command.",
    "setup.parent_checkpoint.path": "Which checkpoint does the run start from (a local directory or the base model id)?",
    "setup.data": "Which training data file(s) does the run read (paths), and where do they come from?",
    "evaluation.protocol.n": "How many benchmark items should each intermediate evaluation use (evaluate.py --limit N)?",
}

CMD_HEAD = re.compile(r"^\s*(python3?|accelerate|torchrun|deepspeed|bash|sh|uv run|\./)\S*", re.MULTILINE)
FENCE = re.compile(r"```(?:bash|sh|shell|console)?\n(.*?)```", re.DOTALL)
MODEL_FLAG = re.compile(r"--(?:model(?:_name)?(?:_or_path)?|base_model|model-name)(?:=|\s+)(\S+)")
OUT_FLAG = re.compile(r"--(?:output[_-]dir|out|save_dir)(?:=|\s+)(\S+)")
STEPS_FLAG = re.compile(r"--(?:max[_-]steps|total[_-]steps|num[_-]steps)(?:=|\s+)(\d+)")
DATA_FLAG = re.compile(r"--(?:train[_-]file|data(?:_path|_file|set)?|train[_-]data)(?:=|\s+)(\S+)")
LIMIT_FLAG = re.compile(r"--limit(?:=|\s+)(\d+)")
HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
KV_LINE = re.compile(r"^\s*([a-z_.]+)\s*[:=]\s*(.+?)\s*$", re.MULTILINE)


def find_command(plan: str) -> list[str] | None:
    """The launch argv: the first fenced shell block, else the first command-looking line."""
    for block in FENCE.findall(plan):
        lines = [ln.strip().rstrip("\\") for ln in block.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if lines:
            try:
                return shlex.split(" ".join(lines))
            except ValueError:
                continue
    for m in CMD_HEAD.finditer(plan):
        line = plan[m.start():].split("\n", 1)[0].strip()
        try:
            return shlex.split(line)
        except ValueError:
            continue
    return None


def get(card: dict[str, Any], dotted: str) -> Any:
    cur: Any = card
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def put(card: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = card
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _parse_answer_value(field: str, text: str) -> Any:
    t = text.strip()
    if field == "setup.command.argv":
        return shlex.split(t)
    if field == "setup.resume_argv":
        if t.lower() in ("yes", "y", "true"):
            return "yes"
        return shlex.split(t)
    if field in ("setup.progress.total", "evaluation.protocol.n"):
        m = re.search(r"\d+", t)
        return int(m.group()) if m else None
    if field == "setup.data":
        return [p for p in re.split(r"[,\s]+", t) if p]
    return t


def parse_answers(text: str | None, mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Answers arrive as a mapping (``--answer-file``) or as ``field: value`` lines (``--answer``)."""
    out: dict[str, Any] = {}
    if mapping:
        for k, v in mapping.items():
            out[str(k)] = v
    if text:
        for m in KV_LINE.finditer(text):
            k, v = m.group(1), m.group(2)
            if k in REQUIRED or k.startswith(("problem.", "hypothesis.", "setup.", "evaluation.")):
                out[k] = _parse_answer_value(k, v)
        if not out and text.strip():
            out["_free_text"] = text.strip()
    return out


def draft_card(card_id: str, plan: str, answers: dict[str, Any], session_dir: Path,
               base_model: str | None = None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Deterministic draft: what the plan and workspace settle, and the questions for the rest."""
    sd = Path(session_dir)
    card: dict[str, Any] = {
        "schema_version": CARD_SCHEMA, "card_id": card_id, "created_at": now(),
        "problem": {}, "hypothesis": {}, "setup": {"method": {"family": "other", "hyperparams": {}}},
        "evaluation": {"protocol": {}},
        "intake": {"plan": plan.strip()[:4000], "answers": {k: v for k, v in answers.items() if k != "_free_text"}},
    }
    text = plan + "\n" + str(answers.get("_free_text", ""))

    # problem / hypothesis: the plan's own words, refined by answers
    paras = [p.strip() for p in re.split(r"\n\s*\n", plan.strip()) if p.strip() and "```" not in p]
    if paras:
        put(card, "problem.statement", paras[0][:600])
    for sent in re.split(r"(?<=[.!?])\s+", " ".join(paras)):
        if re.search(r"\b(expect|should|hypothes|will (?:improve|reduce|raise|fix|help))\b", sent, re.IGNORECASE):
            put(card, "hypothesis.claim", sent.strip()[:400])
            break

    # command and what it implies
    argv = find_command(text)
    if argv:
        put(card, "setup.command.argv", argv)
        put(card, "setup.command.cwd", str(sd))
        script = next((a for a in argv if a.endswith(".py")), None)
        if script:
            put(card, "setup.command.script", str((sd / script).resolve()) if not script.startswith("/") else script)
        put(card, "setup.command.log", str(sd / "logs" / f"{card_id}.log"))
        joined = " ".join(argv)
        for rx, field in ((OUT_FLAG, "setup.output_dir"), (MODEL_FLAG, "setup.parent_checkpoint.path")):
            m = rx.search(joined)
            if m:
                val = m.group(1)
                if field == "setup.output_dir" and not val.startswith("/"):
                    val = str((sd / val).resolve())
                put(card, field, val)
        m = STEPS_FLAG.search(joined)
        if m:
            put(card, "setup.progress", {"unit": "optimizer_step", "total": int(m.group(1))})
        if "--resume_from_checkpoint" in joined:
            put(card, "setup.resume_argv", [a for a in argv if not a.startswith("--resume_from_checkpoint")]
                + ["--resume_from_checkpoint", "{checkpoint}"])
        for m in DATA_FLAG.finditer(joined):
            _add_data(card, m.group(1), sd)
        fam = _family(joined + " " + (Path(get(card, "setup.command.script") or "").name))
        put(card, "setup.method.family", fam)
        put(card, "setup.method.framework", _framework(joined))
    m = LIMIT_FLAG.search(text)
    if m:
        put(card, "evaluation.protocol.n", int(m.group(1)))

    # explicit answers win
    for field, val in answers.items():
        if field == "_free_text":
            continue
        if field == "setup.resume_argv" and val == "yes":
            argv = get(card, "setup.command.argv") or []
            val = [a for a in argv if not a.startswith("--resume_from_checkpoint")] + ["--resume_from_checkpoint", "{checkpoint}"]
        if field == "setup.data":
            card["setup"]["data"] = []
            for p in (val if isinstance(val, list) else [val]):
                _add_data(card, str(p), sd)
            continue
        if field == "setup.progress.total":
            put(card, "setup.progress", {"unit": "optimizer_step", "total": int(val)})
            continue
        put(card, field, val)

    # defaults that are facts, not guesses
    if base_model and not get(card, "setup.parent_checkpoint.path"):
        put(card, "setup.parent_checkpoint.path", base_model)
    if get(card, "setup.parent_checkpoint.path") and not get(card, "setup.parent_checkpoint.origin"):
        put(card, "setup.parent_checkpoint.origin", "base_model")
    if base_model:
        put(card, "setup.base_model", base_model)
    n = get(card, "evaluation.protocol.n")
    if n:
        card["evaluation"]["protocol"].setdefault("dev_set", f"official --limit {n}")
        card["evaluation"]["protocol"].setdefault("command", ["python", "evaluate.py", "--limit", str(n)])
    if not card["setup"].get("data"):
        for p in re.findall(r"(\S+\.jsonl?)\b", text):
            _add_data(card, p, sd)

    questions = [{"field": f, "question": q} for f, q in REQUIRED.items() if _missing(card, f)]
    return card, questions


def _missing(card: dict[str, Any], field: str) -> bool:
    if field == "setup.data":
        return not card["setup"].get("data")
    v = get(card, field)
    return v in (None, "", [], {})


def _add_data(card: dict[str, Any], path: str, sd: Path) -> None:
    p = Path(path) if path.startswith("/") else (sd / path)
    entry = {"path": str(p.resolve()) if p.exists() else path, "source": "local" if p.exists() else path,
             "n_examples": None, "built_by": None, "build_command": [], "selection": "as given in the plan",
             "contamination_check": "not_run", "mixture_weight": 1.0}
    if p.is_file() and p.suffix in (".jsonl", ".json", ".txt"):
        with contextlib.suppress(OSError, UnicodeDecodeError):
            entry["n_examples"] = count_jsonl(p)
    if p.exists() and not inside(p, sd):
        entry["source"] = "outside-session"
    card["setup"].setdefault("data", []).append(entry)


def _family(text: str) -> str:
    t = text.lower()
    for key, fam in (("dpo", "dpo"), ("grpo", "grpo"), ("rft", "rft"), ("reject", "rft"), ("distill", "distill"),
                     ("merge", "merge"), ("sft", "sft"), ("train", "sft")):
        if key in t:
            return fam
    return "other"


def _framework(text: str) -> str:
    t = text.lower()
    if "trl" in t or "sft" in t:
        return "trl"
    if "accelerate" in t:
        return "accelerate"
    if "torchrun" in t or "deepspeed" in t:
        return "torch"
    return "custom"


def is_hf_id(s: str) -> bool:
    return bool(HF_ID.match(s or ""))
