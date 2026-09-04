"""Pre-flight: the checks a training launch must pass, each one a pitfall someone already fell into.

A check reads the card and the files it names and returns pass / warn /
fail / skip with one line of detail. Nothing here starts a process. The
catalogue in ``skills/exp_protocol/pitfalls.yaml`` names which check covers
which pitfall; pitfalls without a check are printed as reminders.

Legacy raw-text checks read the **first** ``SAMPLE_ROWS`` rows of each jsonl file,
not a random sample, and says so in its detail: a file concatenated from a
good source and a bad one passes on its head. The token estimate is
``chars / CHARS_PER_TOKEN``, an English-prose constant that under-counts CJK
and dense code and ignores whatever the trainer's template adds at run time;
both push toward a false pass, which the detail also states.
Opted-in rendered-training evidence instead verifies every prepared token row;
only complete valid evidence supersedes those raw heuristics.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from awm import paths

from .schema import get, now

#: Keys that hold the full training target in a jsonl row, most specific first.
#: ``answer`` comes last because it is often the short gold label, not the target.
TARGET_KEYS = ("completion", "target", "output", "response")
FALLBACK_KEYS = ("text", "answer")
SAMPLE_ROWS = 500
CHARS_PER_TOKEN = 4
#: A target that does not end with the stop token is the eos pitfall; a few
#: malformed rows are tolerated, a systematic mismatch is not.
STOP_TOKEN_MIN_FRAC = 0.95
#: The share of rows allowed to carry the answer marker other than exactly once.
BAD_MARKER_MAX_FRAC = 0.02
#: The share of rows allowed to exceed max_seq_len (origin/main's runtime guard uses 2 %).
OVER_LEN_MAX_FRAC = 0.02
HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


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
    _rendered_training: dict[str, Any] | None = None

    def rendered_training(self) -> dict[str, Any]:
        if self._rendered_training is None:
            from .rendered_training import check_card

            self._rendered_training = check_card(self.card, self.session_dir)
        return self._rendered_training

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


def target_text(row: dict[str, Any]) -> tuple[str, str] | None:
    """The training target of a row and the field it came from, or None."""
    for key in TARGET_KEYS:
        if isinstance(row.get(key), str):
            return row[key], key
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get("content"), str):
        return msgs[-1]["content"], "messages[-1].content"
    for key in FALLBACK_KEYS:
        if isinstance(row.get(key), str):
            return row[key], key
    return None


def row_chars(row: dict[str, Any]) -> int:
    """Every character the trainer could see: top-level strings plus all message contents."""
    chars = sum(len(v) for v in row.values() if isinstance(v, str))
    msgs = row.get("messages")
    if isinstance(msgs, list):
        chars += sum(len(m.get("content", "")) for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), str))
    return chars


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _sample_note() -> str:
    return f"first {SAMPLE_ROWS} rows of each file"


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
    return CheckResult("data_n_examples_match", "pass", f"{seen} file(s) match (whole-file line count)")


@check("command_resolves", "cwd is a directory, the script is a file, and argv names it")
def command_resolves(ctx: Context) -> CheckResult:
    cwd = get(ctx.card, "setup.command.cwd")
    script = get(ctx.card, "setup.command.script")
    argv = get(ctx.card, "setup.command.argv") or []
    problems = []
    if cwd and not Path(cwd).is_dir():
        problems.append(f"cwd {cwd} is not a directory")
    if script and not Path(script).is_file():
        problems.append(f"script {script} is not a file")
    if problems:
        return CheckResult("command_resolves", "fail", "; ".join(problems))
    if not script:
        return CheckResult("command_resolves", "warn", "no setup.command.script named; the lock cannot hash what will run")
    name = Path(str(script)).name
    if not any(name in str(a) for a in argv):
        return CheckResult("command_resolves", "warn",
                           f"script {name} does not appear in argv {argv}; the lock would hash a file the command does not run")
    return CheckResult("command_resolves", "pass", "cwd and script resolve; argv names the script")


@check("output_dir_creatable", "output_dir exists or its parent does")
def output_dir_creatable(ctx: Context) -> CheckResult:
    out = get(ctx.card, "setup.output_dir")
    if not out:
        return CheckResult("output_dir_creatable", "skip", "no output_dir")
    p = Path(out)
    if p.is_dir() or p.parent.is_dir():
        return CheckResult("output_dir_creatable", "pass", str(p))
    return CheckResult("output_dir_creatable", "warn", f"neither {p} nor its parent exists yet")


def _targets(ctx: Context) -> tuple[list[str], set[str]]:
    texts: list[str] = []
    fields: set[str] = set()
    for path in _data_paths(ctx):
        for row in ctx.rows(path):
            hit = target_text(row)
            if hit is None:
                continue
            texts.append(hit[0])
            fields.add(hit[1])
    return texts, fields


def _rendered_supersedes(ctx: Context, check_id: str) -> CheckResult | None:
    if (get(ctx.card, "setup.rendered_training") is not None
            and ctx.rendered_training().get("verified_preparation") is True):
        return CheckResult(check_id, "skip", "raw heuristic superseded by complete all-row rendered token evidence; not a raw PASS")
    return None


@check("rendered_training_evidence", "verify actual prepared token arrays and bound sources/settings")
def rendered_training_evidence(ctx: Context) -> CheckResult:
    report = ctx.rendered_training()
    return CheckResult("rendered_training_evidence", report["status"], report["detail"])


@check("stop_token_consistent", "training targets end with the declared stop token")
def stop_token_consistent(ctx: Context) -> CheckResult:
    superseded = _rendered_supersedes(ctx, "stop_token_consistent")
    if superseded is not None:
        return superseded
    tok = get(ctx.card, "setup.method.stop_token")
    if not tok:
        return CheckResult("stop_token_consistent", "warn",
                           "setup.method.stop_token not declared; the eos-mismatch pitfall cannot be checked")
    texts, fields = _targets(ctx)
    if not texts:
        return CheckResult("stop_token_consistent", "skip",
                           f"no recognisable target field in the {_sample_note()} (looked for {TARGET_KEYS + ('messages',) + FALLBACK_KEYS})")
    ok = sum(t.rstrip().endswith(tok) for t in texts)
    frac = ok / len(texts)
    status = "pass" if frac >= STOP_TOKEN_MIN_FRAC else "fail"
    return CheckResult("stop_token_consistent", status,
                       f"{ok}/{len(texts)} targets end with {tok!r} (pass needs >={STOP_TOKEN_MIN_FRAC:.0%}; "
                       f"{_sample_note()}; field={'/'.join(sorted(fields))})")


@check("answer_marker_single", "each target contains the answer marker exactly once")
def answer_marker_single(ctx: Context) -> CheckResult:
    superseded = _rendered_supersedes(ctx, "answer_marker_single")
    if superseded is not None:
        return superseded
    marker = get(ctx.card, "setup.method.answer_marker")
    if not marker:
        return CheckResult("answer_marker_single", "warn",
                           "setup.method.answer_marker not declared; the double-format pitfall cannot be checked")
    texts, fields = _targets(ctx)
    if not texts:
        return CheckResult("answer_marker_single", "skip", f"no recognisable target field in the {_sample_note()}")
    bad = sum(t.count(marker) != 1 for t in texts)
    status = "pass" if bad / len(texts) <= BAD_MARKER_MAX_FRAC else "fail"
    return CheckResult("answer_marker_single", status,
                       f"{bad}/{len(texts)} targets do not contain {marker!r} exactly once "
                       f"(fail above {BAD_MARKER_MAX_FRAC:.0%}; {_sample_note()}; field={'/'.join(sorted(fields))})")


@check("max_seq_len_headroom", "rows fit in max_seq_len (chars/4 estimate)")
def max_seq_len_headroom(ctx: Context) -> CheckResult:
    superseded = _rendered_supersedes(ctx, "max_seq_len_headroom")
    if superseded is not None:
        return superseded
    msl = get(ctx.card, "setup.method.hyperparams.max_seq_len")
    if not isinstance(msl, int) or msl <= 0:
        return CheckResult("max_seq_len_headroom", "warn",
                           "hyperparams.max_seq_len not declared; truncation cannot be estimated")
    total = over = longest = measured = 0
    for path in _data_paths(ctx):
        for row in ctx.rows(path):
            chars = row_chars(row)
            measured += chars
            est = chars // CHARS_PER_TOKEN
            total += 1
            longest = max(longest, est)
            over += est > msl
    if total == 0 or measured == 0:
        return CheckResult("max_seq_len_headroom", "skip", f"no measurable text in the {_sample_note()}")
    frac = over / total
    detail = (f"{over}/{total} rows estimated over {msl} tokens, longest ~{longest} "
              f"(fail above {OVER_LEN_MAX_FRAC:.0%}; {_sample_note()}; estimate is chars/{CHARS_PER_TOKEN} of the row text "
              f"only — template and system-prompt tokens added at training time are not counted)")
    if frac > OVER_LEN_MAX_FRAC:
        return CheckResult("max_seq_len_headroom", "fail", detail)
    if over:
        return CheckResult("max_seq_len_headroom", "warn", detail)
    return CheckResult("max_seq_len_headroom", "pass", detail)


@check("comparator_same_protocol", "verify recorded comparator counts; mark absent evidence unverified")
def comparator_same_protocol(ctx: Context) -> CheckResult:
    comp = get(ctx.card, "evaluation.comparator") or {}
    if isinstance(comp, dict) and comp.get("defer_validation") is True:
        from .comparator import check_output

        report = check_output(ctx.card, allow_missing=True)
        return CheckResult("comparator_same_protocol", report["status"], report["detail"])
    path = comp.get("path") if isinstance(comp, dict) else None
    if not path:
        return CheckResult("comparator_same_protocol", "skip", "no comparator path")
    p = Path(path)
    if not p.is_file():
        return CheckResult("comparator_same_protocol", "fail", f"{p} does not exist")
    n = get(ctx.card, "evaluation.protocol.n")
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError, UnicodeError):
        return CheckResult("comparator_same_protocol", "warn",
                           f"{p.name}: unverified comparator; no readable JSON count evidence")
    if not isinstance(payload, dict):
        return CheckResult("comparator_same_protocol", "warn",
                           f"{p.name}: unverified comparator; unsupported report shape")
    if "status" in payload and payload["status"] not in ("success", "completed"):
        return CheckResult("comparator_same_protocol", "fail",
                           f"{p.name}: evaluator did not report successful completion")
    # A requested limit can disprove the declared comparison, but matching it
    # cannot prove how many samples actually completed. Never infer n from SE.
    if "limit" in payload and (type(payload["limit"]) is not int or payload["limit"] != n):
        return CheckResult("comparator_same_protocol", "fail",
                           f"{p.name}: requested limit does not match protocol n={n}")
    results = payload.get("results")
    has_counts = any(key in payload for key in ("n", "num_samples", "samples"))
    if isinstance(results, dict):
        has_counts = has_counts or any(key in results for key in ("total_samples", "completed_samples"))
        scores = results.get("scores")
        if isinstance(scores, list):
            has_counts = has_counts or any(isinstance(score, dict) and "scored_samples" in score
                                           for score in scores)
    if not has_counts:
        return CheckResult("comparator_same_protocol", "warn",
                           f"{p.name}: actual count unverified; limit, population and stderr are not count evidence")
    direct_counts = [(key, payload[key]) for key in ("n", "num_samples") if key in payload]
    if "samples" in payload:
        value = payload["samples"]
        direct_counts.append(("samples", len(value) if isinstance(value, list) else value))
    if isinstance(results, dict):
        direct_counts.extend((key, results[key]) for key in ("total_samples", "completed_samples")
                             if key in results)
    for source, actual in direct_counts:
        if type(actual) is not int or actual != n:
            return CheckResult("comparator_same_protocol", "fail",
                               f"{p.name}: {source}={actual!r}, expected actual n={n}")
    metric = get(ctx.card, "hypothesis.expected_effect.metric")
    if not isinstance(metric, str) or not metric.strip():
        return CheckResult("comparator_same_protocol", "warn",
                           f"{p.name}: count fields match; metric and remaining comparison identity unverified")
    from .comparator import helper

    try:
        checked = helper().inspect_output(str(p.resolve()), n, metric)
    except (OSError, ValueError, TypeError, ImportError, SyntaxError):
        return CheckResult("comparator_same_protocol", "warn",
                           f"{p.name}: comparator helper unavailable; evidence remains unverified")
    return CheckResult("comparator_same_protocol", checked["status"], checked["detail"])


@check("parent_checkpoint_loadable", "a local parent checkpoint has a config.json")
def parent_checkpoint_loadable(ctx: Context) -> CheckResult:
    path = get(ctx.card, "setup.parent_checkpoint.path")
    if not path:
        return CheckResult("parent_checkpoint_loadable", "skip", "no parent path")
    s = str(path)
    p = Path(s)
    if not s.startswith("/"):
        if HUB_ID_RE.match(s) and not p.exists():
            return CheckResult("parent_checkpoint_loadable", "pass", f"{s} is a hub id; not checked offline")
        return CheckResult("parent_checkpoint_loadable", "fail",
                           f"{s} is a relative path; write the absolute path of the checkpoint directory")
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


def pitfalls_path(session_dir: Path | None = None) -> Path | None:
    """The copy ``install`` put in the session dir wins; else the skill dir's; else None."""
    candidates = []
    if session_dir is not None:
        candidates.append(Path(session_dir) / "skills" / "exp_protocol" / "pitfalls.yaml")
    candidates.append(skill_dir() / "pitfalls.yaml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_pitfalls(path: Path | None = None) -> list[dict[str, Any]]:
    """The catalogue at ``path`` (default: the skill dir's). Missing file → empty list, never an exception."""
    p = Path(path) if path else skill_dir() / "pitfalls.yaml"
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text()) or []
    if not isinstance(data, list):
        raise ValueError(f"{p}: pitfalls.yaml must be a list")
    return data


def run_preflight(card: dict[str, Any], session_dir: Path | None = None,
                  pitfalls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ctx = Context(card, Path(session_dir) if session_dir else None, {})
    results = [fn(ctx) for _, fn in CHECKS.values()]
    summary = {s: sum(r.status == s for r in results) for s in ("pass", "warn", "fail", "skip")}
    catalogue_path: str | None = None
    if pitfalls is None:
        found = pitfalls_path(ctx.session_dir)
        catalogue = load_pitfalls(found) if found else []
        catalogue_path = str(found) if found else None
    else:
        catalogue = pitfalls
    reminders = [{"id": p["id"], "symptom": p["symptom"], "guidance": p["guidance"]}
                 for p in catalogue if p.get("check") is None]
    report = {"ran_at": now(), "results": [asdict(r) for r in results],
              "summary": summary, "reminders": reminders, "catalogue": catalogue_path}
    if get(card, "setup.rendered_training") is not None:
        report["rendered_training"] = ctx.rendered_training()
    return report


def render(report: dict[str, Any]) -> str:
    lines = []
    for r in report["results"]:
        desc = CHECKS.get(r["check"], ("", None))[0]
        lines.append(f"{r['status'].upper():5} {r['check']} — {desc}: {r['detail']}")
    s = report["summary"]
    lines.append(f"-- {s['pass']} pass, {s['warn']} warn, {s['fail']} fail, {s['skip']} skip")
    if report.get("catalogue") is None and "catalogue" in report:
        lines.append("-- no pitfalls.yaml found (looked in <session>/skills/exp_protocol and the skill dir); "
                     "reminders unavailable")
    if report["reminders"]:
        lines.append("-- not checkable by machine; check yourself:")
        for rem in report["reminders"]:
            lines.append(f"   * {rem['id']}: {rem['guidance']}")
    return "\n".join(lines)
