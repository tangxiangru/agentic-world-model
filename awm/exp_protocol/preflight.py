"""Pre-flight: the checks a training launch must pass, each one a pitfall someone already fell into.

A check reads the card and the files it names and returns pass / warn /
fail / skip with one line of detail. Nothing here starts a process. The
catalogue in ``skills/exp_protocol/pitfalls.yaml`` names which check covers
which pitfall; pitfalls without a check are printed as reminders.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from awm import paths

from .schema import get, now

#: Keys that hold the training target in a jsonl row, most specific first.
TARGET_KEYS = ("completion", "target", "output", "response", "answer")
SAMPLE_ROWS = 500
CHARS_PER_TOKEN = 4  # a coarse estimate; the check says so in its detail


@dataclass
class CheckResult:
    check: str
    status: str  # pass | warn | fail | skip
    detail: str


@dataclass
class Context:
    card: dict[str, Any]
    session_dir: Path | None
    _rows: dict[str, list[dict[str, Any]]]

    def rows(self, path: str) -> list[dict[str, Any]]:
        """The first SAMPLE_ROWS rows of a jsonl file, parsed; cached per path."""
        if path not in self._rows:
            out: list[dict[str, Any]] = []
            p = Path(path)
            if p.is_file() and p.suffix == ".jsonl":
                with p.open() as fh:
                    for i, line in enumerate(fh):
                        if i >= SAMPLE_ROWS:
                            break
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            out.append(row)
            self._rows[path] = out
        return self._rows[path]


CHECKS: dict[str, tuple[str, Callable[[Context], CheckResult]]] = {}


def check(check_id: str, description: str):
    def register(fn: Callable[[Context], CheckResult]):
        CHECKS[check_id] = (description, fn)
        return fn
    return register


def _data_paths(ctx: Context) -> list[str]:
    return [str(d["path"]) for d in (get(ctx.card, "setup.data") or []) if isinstance(d, dict) and d.get("path")]


def target_text(row: dict[str, Any]) -> str | None:
    for key in TARGET_KEYS:
        if isinstance(row.get(key), str):
            return row[key]
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get("content"), str):
        return msgs[-1]["content"]
    if isinstance(row.get("text"), str):
        return row["text"]
    return None


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


# ----------------------------------------------------------------- checks

@check("data_files_exist", "every setup.data[].path is a file")
def data_files_exist(ctx: Context) -> CheckResult:
    if not _data_paths(ctx):
        return CheckResult("data_files_exist", "skip", "no data paths in the card")
    missing = [p for p in _data_paths(ctx) if not Path(p).is_file()]
    if missing:
        return CheckResult("data_files_exist", "fail", "missing: " + ", ".join(missing))
    return CheckResult("data_files_exist", "pass", f"{len(_data_paths(ctx))} file(s) present")


@check("data_n_examples_match", "n_examples equals the line count of each jsonl file")
def data_n_examples_match(ctx: Context) -> CheckResult:
    bad = []
    seen = 0
    for d in get(ctx.card, "setup.data") or []:
        p = Path(str(d.get("path", "")))
        if p.suffix != ".jsonl" or not p.is_file():
            continue
        seen += 1
        actual = _count_lines(p)
        if actual != d.get("n_examples"):
            bad.append(f"{p.name}: card says {d.get('n_examples')}, file has {actual}")
    if not seen:
        return CheckResult("data_n_examples_match", "skip", "no jsonl data files to count")
    if bad:
        return CheckResult("data_n_examples_match", "fail", "; ".join(bad))
    return CheckResult("data_n_examples_match", "pass", f"{seen} file(s) match")


@check("command_resolves", "cwd is a directory and the script is a file")
def command_resolves(ctx: Context) -> CheckResult:
    cwd = get(ctx.card, "setup.command.cwd")
    script = get(ctx.card, "setup.command.script")
    problems = []
    if cwd and not Path(cwd).is_dir():
        problems.append(f"cwd {cwd} is not a directory")
    if script and not Path(script).is_file():
        problems.append(f"script {script} is not a file")
    if problems:
        return CheckResult("command_resolves", "fail", "; ".join(problems))
    if not script:
        return CheckResult("command_resolves", "warn", "no setup.command.script named; the lock cannot hash what will run")
    return CheckResult("command_resolves", "pass", "cwd and script resolve")


@check("output_dir_creatable", "output_dir exists or its parent does")
def output_dir_creatable(ctx: Context) -> CheckResult:
    out = get(ctx.card, "setup.output_dir")
    if not out:
        return CheckResult("output_dir_creatable", "skip", "no output_dir")
    p = Path(out)
    if p.is_dir() or p.parent.is_dir():
        return CheckResult("output_dir_creatable", "pass", str(p))
    return CheckResult("output_dir_creatable", "warn", f"neither {p} nor its parent exists yet")


@check("stop_token_consistent", "training targets end with the declared stop token")
def stop_token_consistent(ctx: Context) -> CheckResult:
    tok = get(ctx.card, "setup.method.stop_token")
    if not tok:
        return CheckResult("stop_token_consistent", "warn",
                           "setup.method.stop_token not declared; the eos-mismatch pitfall cannot be checked")
    total = ok = 0
    for path in _data_paths(ctx):
        for row in ctx.rows(path):
            t = target_text(row)
            if t is None:
                continue
            total += 1
            ok += t.rstrip().endswith(tok)
    if total == 0:
        return CheckResult("stop_token_consistent", "skip", "no jsonl rows with a recognisable target field")
    frac = ok / total
    status = "pass" if frac >= 0.95 else "fail"
    return CheckResult("stop_token_consistent", status, f"{ok}/{total} sampled targets end with {tok!r}")


@check("answer_marker_single", "each target contains the answer marker exactly once")
def answer_marker_single(ctx: Context) -> CheckResult:
    marker = get(ctx.card, "setup.method.answer_marker")
    if not marker:
        return CheckResult("answer_marker_single", "warn",
                           "setup.method.answer_marker not declared; the double-format pitfall cannot be checked")
    total = bad = 0
    for path in _data_paths(ctx):
        for row in ctx.rows(path):
            t = target_text(row)
            if t is None:
                continue
            total += 1
            bad += t.count(marker) != 1
    if total == 0:
        return CheckResult("answer_marker_single", "skip", "no jsonl rows with a recognisable target field")
    status = "pass" if bad / total <= 0.02 else "fail"
    return CheckResult("answer_marker_single", status,
                       f"{bad}/{total} sampled targets do not contain {marker!r} exactly once")


@check("max_seq_len_headroom", "rows fit in max_seq_len (chars/4 estimate)")
def max_seq_len_headroom(ctx: Context) -> CheckResult:
    msl = get(ctx.card, "setup.method.hyperparams.max_seq_len")
    if not isinstance(msl, int) or msl <= 0:
        return CheckResult("max_seq_len_headroom", "warn",
                           "hyperparams.max_seq_len not declared; truncation cannot be estimated")
    total = over = 0
    longest = 0
    for path in _data_paths(ctx):
        for row in ctx.rows(path):
            chars = sum(len(v) for v in row.values() if isinstance(v, str))
            if not chars and isinstance(row.get("messages"), list):
                chars = sum(len(m.get("content", "")) for m in row["messages"] if isinstance(m, dict))
            est = chars // CHARS_PER_TOKEN
            total += 1
            longest = max(longest, est)
            over += est > msl
    if total == 0:
        return CheckResult("max_seq_len_headroom", "skip", "no jsonl rows to measure")
    frac = over / total
    detail = (f"{over}/{total} sampled rows estimated over {msl} tokens "
              f"(longest ~{longest}; estimate is chars/{CHARS_PER_TOKEN})")
    if frac > 0.02:
        return CheckResult("max_seq_len_headroom", "fail", detail)
    if over:
        return CheckResult("max_seq_len_headroom", "warn", detail)
    return CheckResult("max_seq_len_headroom", "pass", detail)


@check("comparator_same_protocol", "the comparator's eval file exists and used the same n")
def comparator_same_protocol(ctx: Context) -> CheckResult:
    comp = get(ctx.card, "evaluation.comparator") or {}
    path = comp.get("path") if isinstance(comp, dict) else None
    if not path:
        return CheckResult("comparator_same_protocol", "skip", "no comparator path")
    p = Path(path)
    if not p.is_file():
        return CheckResult("comparator_same_protocol", "fail", f"{p} does not exist")
    n = get(ctx.card, "evaluation.protocol.n")
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return CheckResult("comparator_same_protocol", "pass", f"{p.name} exists; not JSON, n not verifiable")
    for key in ("n", "limit", "num_samples", "samples"):
        if isinstance(payload, dict) and key in payload:
            if payload[key] != n:
                return CheckResult("comparator_same_protocol", "fail",
                                   f"{p.name} records {key}={payload[key]}, the protocol says n={n}")
            return CheckResult("comparator_same_protocol", "pass", f"{p.name} records {key}={n}")
    return CheckResult("comparator_same_protocol", "pass", f"{p.name} exists; it does not record n")


@check("parent_checkpoint_loadable", "a local parent checkpoint has a config.json")
def parent_checkpoint_loadable(ctx: Context) -> CheckResult:
    path = get(ctx.card, "setup.parent_checkpoint.path")
    if not path:
        return CheckResult("parent_checkpoint_loadable", "skip", "no parent path")
    p = Path(str(path))
    if not str(path).startswith("/"):
        return CheckResult("parent_checkpoint_loadable", "pass", f"{path} is a hub id; not checked offline")
    if not p.is_dir():
        return CheckResult("parent_checkpoint_loadable", "fail", f"{p} is not a directory")
    if not (p / "config.json").is_file():
        return CheckResult("parent_checkpoint_loadable", "fail",
                           f"{p} has no config.json; the grader's loader will not accept it")
    return CheckResult("parent_checkpoint_loadable", "pass", f"{p} has config.json")


# --------------------------------------------------------------- catalogue

def skill_dir() -> Path:
    """Where the scientist skill lives: $AWM_EXP_PROTOCOL_DIR, else <repo>/skills/exp_protocol."""
    env = os.environ.get("AWM_EXP_PROTOCOL_DIR")
    return Path(env) if env else paths.REPO_ROOT / "skills" / "exp_protocol"


def load_pitfalls(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else skill_dir() / "pitfalls.yaml"
    data = yaml.safe_load(p.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"{p}: pitfalls.yaml must be a list")
    return data


def run_preflight(card: dict[str, Any], session_dir: Path | None = None,
                  pitfalls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ctx = Context(card, Path(session_dir) if session_dir else None, {})
    results = [fn(ctx) for _, fn in CHECKS.values()]
    summary = {s: sum(r.status == s for r in results) for s in ("pass", "warn", "fail", "skip")}
    catalogue = load_pitfalls() if pitfalls is None else pitfalls
    reminders = [{"id": p["id"], "symptom": p["symptom"], "guidance": p["guidance"]}
                 for p in catalogue if p.get("check") is None]
    return {"ran_at": now(), "results": [asdict(r) for r in results],
            "summary": summary, "reminders": reminders}


def render(report: dict[str, Any]) -> str:
    lines = [f"{r['status'].upper():5} {r['check']}: {r['detail']}" for r in report["results"]]
    s = report["summary"]
    lines.append(f"-- {s['pass']} pass, {s['warn']} warn, {s['fail']} fail, {s['skip']} skip")
    if report["reminders"]:
        lines.append("-- not checkable by machine; check yourself:")
        for rem in report["reminders"]:
            lines.append(f"   * {rem['id']}: {rem['guidance']}")
    return "\n".join(lines)
