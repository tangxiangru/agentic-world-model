"""Every evaluation an agent launched, and whether a number ever came back.

The second question is the one that matters. Counting evaluation calls
overstates what an agent knew, in both directions: a backgrounded evaluation
polled five times counts five, and an evaluation whose output was truncated
before the accuracy line counts one despite telling the agent nothing. One run
in the corpus piped both of its evaluations through ``| head -100``, which cut
them off above the accuracy print — it finished the task having never seen a
single score, and never knew. A verifier that returns no signal is not a
verifier, so ``got_signal`` is a first-class column rather than something a
consumer infers from the call.

Pairing has the same shape as ``train_spans``: launches are mostly backgrounded
(``nohup … &``, or the harness moving a long command off the foreground), so the
launching event's own result carries a PID or a task id, never a score. The
accuracy arrives later, through one of four channels the corpus actually uses:

*   the agent cats the ``--json-output-file`` it asked for
*   the agent greps ``Accuracy:`` out of the run's log
*   the launch ran in the foreground and its own result holds the output
*   the harness returns a backgrounded task, keyed by the task id it printed

So a launch is joined to the first later result that both contains a score and
refers to that launch — by its output file, its log, or its task id. A result
that merely contains a score is not evidence for *this* evaluation.

``python evaluate.py`` must be an invocation, not a mention: agents read
``evaluate.py`` with ``sed``, grep for it in ``ps aux``, and search
``inspect_ai`` source for its behaviour, none of which run anything. As in
``train_spans``, requiring the script to sit in python's argument position
separates the two. Codex wraps everything in ``/bin/bash -lc '…'``, which is
unwrapped before matching.

**``got_signal`` is a lower bound, and how tight a bound differs by family.**
False means no score could be *traced* to this launch, which is not the same as
the agent having learned nothing. Measured over the 180 in-scope runs, 70% of
Claude Code evaluations link to a score against 53% of codex ones, and the gap
is a retrieval-pattern difference rather than a difference in what the agents
knew: codex habitually reads its scores through analysis scripts it wrote
itself, pointed at inspect_ai's timestamped log directory
(``analyze_eval.py logs/2026-07-17T01-32-28+02-00_gpqa-main_<hash>.json``),
and that filename cannot be predicted from the launching command. Checked
against the alternative explanation: **none** of the unlinked results are
truncated by the trace, and they average 26 kB of vLLM server log, so the score
genuinely is not in the launch's own output.

Three channels are followed — the requested output file, a log named on the
command line, and the evaluated checkpoint's own name — and a fourth, the
harness's background task id, when the launch was moved off the foreground.
Beyond those, whether an evaluation informed the agent is a judgment-layer
question, and for codex it usually is.

``run_eval.sh`` is an agent-written wrapper whose positional signature differs
per run (``run_eval.sh <model> <tag> <limit> <temp>`` in one, ``run_eval.sh
<model> <limit> <out>`` in another). It is recorded as an evaluation, but its
``limit`` is left unknown rather than guessed from position — 180 calls against
2,536 direct ones, so guessing would buy little and cost the column's meaning.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from awm import paths
from awm.traj import scripts

DTYPES: dict[str, str] = {
    "run_id": "string",
    "i": "Int64",
    "ts": "string",
    "form": "string",
    "model_path": "string",
    "limit": "Int64",
    "max_connections": "Int64",
    "json_output_file": "string",
    "mode": "string",
    "tier": "Int64",
    "n_calls": "Int64",
    "got_signal": "boolean",
    "signal_i": "Int64",
    "signal_ts": "string",
    "signal_via": "string",
    "accuracy": "Float64",
    "wait_s": "Float64",
    "command": "string",
}

COLUMNS: tuple[str, ...] = tuple(DTYPES)

#: Codex issues every command as ``/bin/bash -lc '<real command>'``.
_BASH_LC = re.compile(r"^\s*/bin/bash\s+-lc\s+(['\"])(?P<inner>.*)\1\s*$", re.S)

#: The script must be python's argument, not merely named. ``sed -n '1,260p'
#: evaluate.py``, ``ps aux | grep evaluate.py`` and greps through inspect_ai's
#: source all name it without running it.
_LAUNCH_PY = re.compile(r"\bpython3?\s+(?:-[\w-]+\s+)*[\w./-]*evaluate\.py\b")

#: ``evaluate_aime2024.py``, ``evaluate_validation.py``, ``evaluate_historical.py``
#: — the official scorer copied and pointed at a *different* test set. 43 calls
#: across the corpus. They are not the official verifier and must not enter the
#: four-tier table, but they are not invisible either: in one run the three calls
#: to ``evaluate_2024.py`` carried that run's only cross-branch checkpoint choice.
_LAUNCH_VARIANT = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*[\w./-]*evaluate[_-][\w-]+\.py\b"
)

#: ``python -c "import evaluate; evaluate.DEFAULT_EPOCHS=3; sys.argv=[…];
#: evaluate.main()"`` runs the official scorer without ever naming its file on a
#: command line. One run drove nine full evaluations this way -- including the
#: ones that chose what it submitted -- and none were recorded.
#: Judged on the whole command, never per segment: the import and the call sit
#: either side of a ``;`` that segmentation would split apart.
_LAUNCH_IMPORT = re.compile(r"\bimport\s+(?:[\w,\s]*\b)?evaluate\b.*?\bevaluate\.main\s*\(", re.S)
_LAUNCH_SH = re.compile(r"(?:^|[;&|]\s*|\bbash\s+|\bsh\s+)[\w./-]*run_eval[\w.-]*\.sh\b")

#: A command that *writes* a script naming a trainer or an evaluator is not
#: running one. Agents build wrappers with ``cat > run_eval.sh <<'EOF' … EOF``,
#: and the body holds a full command line; matching inside it invents launches
#: that never happened, inflating both the count and the share that returned a
#: score.
_HEREDOC_BODY = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?^\1", re.S | re.M)


def strip_heredocs(command: str) -> str:
    """The command without any here-document payload."""
    return _HEREDOC_BODY.sub(" ", command)

#: One shell string often chains a training and an evaluation. Scanning the whole
#: string for ``--limit`` took a trainer's ``--limit 25000`` as the evaluation's
#: sample size and scored a 60-question run as a full test set, so flags are read
#: from the launching segment alone.
#: ``do`` / ``done`` / ``then`` split too: a ``for m in a b; do evaluate.py $m;
#: done`` is one tool call evaluating several models, and treating it as one
#: event kept only the first score. Six commands per run in one trace.
_SEGMENT = re.compile(r"&&|\|\||[;&|]|\n|\bdo\b|\bdone\b|\bthen\b|\bfi\b")

_MODEL_PATH = re.compile(r"--model[-_]path['\"]?[=, ]\s*['\"]?([^\s'\",]+)")
#: ``--limit 448`` on a command line, or ``'--limit','448'`` inside a sys.argv
#: list built in a ``python -c`` one-liner.
_LIMIT = re.compile(r"--limit['\"]?[=, ]\s*['\"]?(-?\d+)")
_MAXCONN = re.compile(r"--max[-_]connections[= ]\s*(\d+)")
_JSON_OUT = re.compile(r"--json[-_]output[-_]file[= ]\s*['\"]?([^\s'\"]+)")

_BACKGROUND = re.compile(r"\bnohup\b|[^&>]&\s*(?:$|\n|;|echo\b)")

#: Three spellings the corpus produces: ``"accuracy": 0.7266`` from the json the
#: harness writes, ``Accuracy: 0.2339 (±0.0402)`` from the log line, and
#: ``accuracy  0.820`` — whitespace-aligned columns, no separator at all — from
#: the wrappers agents write for themselves. Demanding a colon missed every
#: score in a run whose evaluations all went through such a wrapper.
#: The quote may also be a single one: ``print(json.load(open(f)))`` prints a
#: Python dict repr, and that is how the codex runs habitually read a score back.
#: ``pass@1`` is the same quantity under the name a self-built scorer gives it;
#: 6 runs (3%) spell it that way.
_ACCURACY = re.compile(
    r'''["']?(?:accuracy|pass@?1)["']?(?:\s*[:=]\s*|\s+)([0-9.]+)''', re.I
)

#: The harness prints these when it moves a command off the foreground, and
#: echoes the same id back on the retrieval that finally carries the output.
_BG_HANDLE = re.compile(r"background \(ID: (\w+)\)|background with ID: (\w+)")
_TASK_ID = re.compile(r"<task_id>(\w+)</task_id>")

#: Full test-set sizes, so ``--limit 1319`` on GSM8K reads as a full evaluation
#: rather than a subsample. Absent benchmarks fall back to "any positive limit
#: is a subsample", which is the safe direction: it never calls a partial
#: evaluation complete.
_FULL_SIZE = {
    "gsm8k": 1319,
    "humaneval": 164,
    "aime2025": 30,
    "gpqamain": 448,
    "arenahardwriting": 250,
}

#: ``--limit``'s default, read from each task's own ``evaluate.py``. It is *not*
#: the full set for most benchmarks, and it differs per benchmark: an omitted
#: flag means 150 questions on GSM8K and HumanEval, 50 on GPQA, 32 on the two
#: LLM-judged ones, and the whole set only on AIME and BFCL. Treating the absent
#: flag as "full" everywhere filed subsample runs as fourth-tier evaluations.
_LIMIT_DEFAULT = {
    "aime2025": None,
    "aime2026": None,
    "bfcl": None,
    "gpqamain": 50,
    "gsm8k": 150,
    "humaneval": 150,
    "arenahardwriting": 32,
    "healthbench": 32,
}


def _parse_ts(ts: Any) -> datetime | None:
    if ts is None or ts is pd.NA or not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds(a: Any, b: Any) -> float | None:
    ta, tb = _parse_ts(a), _parse_ts(b)
    return None if ta is None or tb is None else (tb - ta).total_seconds()


def unwrap(command: str) -> str:
    """Strip codex's ``/bin/bash -lc '…'`` so one matcher serves both harnesses."""
    m = _BASH_LC.match(command)
    return m.group("inner") if m else command


def _command(event: dict[str, Any]) -> str:
    if event.get("type") != "tool_use":
        return ""
    return unwrap((event.get("args") or {}).get("command") or "")


#: ``python evaluate.py --help`` reads the interface. That is a first-tier static
#: check, not an evaluation, and counting it as one inflates the tally with calls
#: that never touched a model.
_HELP_ONLY = re.compile(r"evaluate\.py\s+(?:[\w./-]+\s+)*--help\b")

#: ``pgrep -f 'python evaluate.py --model-path X'`` quotes a whole command line
#: to *watch* a running evaluation. The quoted text satisfies every launch
#: pattern, so one evaluation was counted twice and the phantom row was handed
#: a score belonging to a different model 96 events later.
_INSPECTS_PROCESS = re.compile(r"\b(?:pgrep|pkill|ps|rg|grep|awk)\b")


def segments(command: str) -> list[str]:
    # Join backslash continuations first: a flag on the wrapped line belongs to
    # the same invocation, and splitting on the newline dropped a `--limit 40`
    # onto its own segment, filing a 40-question run as a full evaluation.
    command = re.sub(r"\\\s*\n\s*", " ", command)
    return [s for s in _SEGMENT.split(command) if s.strip()]


def _form(command: str, known: dict[str, set[str]] | None = None) -> str | None:
    outside = strip_heredocs(command)
    # A watcher names the launch it is watching; drop those segments first.
    kept = [seg for seg in segments(outside) if not _INSPECTS_PROCESS.search(seg)]
    if not kept:
        return None
    outside = " ; ".join(kept)
    if _HELP_ONLY.search(outside):
        return None
    if _LAUNCH_PY.search(outside) or _LAUNCH_IMPORT.search(outside):
        return "evaluate.py"
    if _LAUNCH_VARIANT.search(outside):
        return "evaluator_variant"
    if _LAUNCH_SH.search(outside):
        return "run_eval.sh"
    # An agent that wrapped evaluation in a script of its own naming — one run
    # called it ``work/ev.sh`` and by name-matching never evaluated at all.
    #
    # A script that also trains is not one of these. Training scripts commonly
    # score what they just trained, so they read as evaluators too; counting
    # their launch as an evaluation turns every training into a phantom eval.
    if known and scripts.invoked_purely(outside, known, "evaluator"):
        return "own_wrapper"
    return None


def _int(pattern: re.Pattern[str], command: str) -> int | None:
    m = pattern.search(command)
    return int(m.group(1)) if m else None


def _str(pattern: re.Pattern[str], command: str) -> str | None:
    m = pattern.search(command)
    return m.group(1) if m else None


def tier_for(limit: int | None, benchmark: str | None, form: str | None = None) -> int | None:
    """Third tier for a subsample, fourth for the whole test set.

    ``evaluate.py`` with no ``--limit`` scores the whole set: that is its
    default, and it is how one agent ran all seventeen of its full evaluations.
    Reading the absent flag as "unknown" left every one of them untiered.

    A variant of the scorer pointed at another test set gets no tier at any
    sample size. The four tiers are rungs on the *official* set; a copy aimed at
    AIME 2024 is a self-built proxy, however faithfully it reuses the code.
    """
    if form == "evaluator_variant":
        return None
    if limit is None:
        if form != "evaluate.py" or benchmark not in _LIMIT_DEFAULT:
            return None
        default = _LIMIT_DEFAULT[benchmark]
        if default is None:
            return 4
        limit = default
    if limit < 0:
        return 4
    full = _FULL_SIZE.get(benchmark or "")
    if full is not None and limit >= full:
        return 4
    return 3


#: ``for step in 100 200 300; do python evaluate.py …; done`` — one event, three
#: evaluations. Two annotators reported the undercount independently. Across the
#: corpus it is 10%, but it is not spread evenly: 16% for codex, 1% for
#: claude-code, because packing evaluations into a shell loop is a codex habit.
#: **So evaluation counts must not be compared across the two families.**
_LOOP = re.compile(r"\bfor\s+\w+\s+in\b|\bwhile\s+read\b")


def call_count(command: str, own_text: str | None) -> int:
    """How many evaluations this one event ran.

    A loop body runs once per score it printed; when the loop printed none, one
    is the floor rather than the truth, and the row is a lower bound like every
    other row.
    """
    if not _LOOP.search(command):
        return 1
    return max(1, len(_ACCURACY.findall(own_text or "")))


def _own_slice(text: str | None, command: str) -> str | None:
    """The part of a result that belongs to this launch, when it can be told.

    Agents chain ``summarize_eval.py <previous log> && evaluate.py <next>``; the
    result then opens with the previous run's score. When the evaluated model's
    name appears in the text, everything before its first mention belongs to the
    neighbour and is dropped.
    """
    if not text:
        return text
    model = _MODEL_PATH.search(command)
    if not model:
        return text
    leaf = model.group(1).rstrip("/").split("/")[-1]
    at = text.find(leaf)
    return text[at:] if at > 0 else text


def _accuracy_in(text: str | None) -> float | None:
    if not text:
        return None
    for raw in _ACCURACY.findall(text):
        try:
            v = float(raw)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


def _handles(text: str | None) -> set[str]:
    if not text:
        return set()
    return {g for m in _BG_HANDLE.findall(text) for g in m if g}


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _result_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("type") == "tool_result" and e.get("parent_tool_use"):
            out.setdefault(e["parent_tool_use"], e)
    return out


def events_for_run(
    run_id: str, events: list[dict[str, Any]], benchmark: str | None = None
) -> list[dict[str, Any]]:
    """Every evaluation this run launched, joined to the score it produced."""
    events = sorted(events, key=lambda e: (e.get("agent_id") or "", e.get("i") or 0))
    known = scripts.learn(events)
    results = _result_index(events)
    uses = [e for e in events if e.get("type") == "tool_use"]
    ordered = [e for e in events if e.get("type") in ("tool_use", "tool_result")]

    rows: list[dict[str, Any]] = []
    for pos, event in enumerate(uses):
        whole = re.sub(r"\\\s*\n\s*", " ", _command(event))
        # A one-liner that imports the scorer and calls main() is one launch
        # spanning several segments; treat the command as a single unit.
        if _LAUNCH_IMPORT.search(whole) and not _INSPECTS_PROCESS.search(whole):
            rows.extend(_one(
                run_id, event, pos, whole, "evaluate.py", uses, ordered, results,
                benchmark, known,
            ))
            continue
        # A command may launch several evaluations (``run_eval.sh a … ;
        # run_eval.sh b …``); each is its own verification event, and reporting
        # one dropped the highest subsample reading of an entire run.
        launching = [
            seg for seg in segments(strip_heredocs(whole)) if _form(seg, known) is not None
        ]
        for command in (launching or []):
            form = _form(command, known)
            rows.extend(_one(
                run_id, event, pos, command, form, uses, ordered, results, benchmark, known
            ))
    return rows


def _one(
    run_id: str,
    event: dict[str, Any],
    pos: int,
    command: str,
    form: str,
    uses: list[dict[str, Any]],
    ordered: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    benchmark: str | None,
    known: dict[str, set[str]],
) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        own = results.get(event.get("tool_use_id") or "")
        own_text = (own or {}).get("text") or ""
        # ``usage: evaluate.py [-h] …`` — argparse rejected the arguments and no
        # evaluation happened. Three of one run's launches failed this way and
        # each was handed the score of a later, successful rerun.
        refused = bool(own is not None and own.get("is_error"))
        limit = _int(_LIMIT, command)
        if limit is None and form in ("run_eval.sh", "own_wrapper"):
            # The wrapper takes its sample size positionally. ``-1`` is the full
            # test set and is the only form the fourth tier ever appears in for
            # these runs; leaving it unparsed left every full evaluation untiered.
            m = _LAUNCH_SH.search(command) or re.search(r"\b(?:bash|sh)\s+\S+", command)
            tail = command[m.end():] if m else ""
            # Strip redirections first: ``2>&1`` offered a literal 2 and three
            # full evaluations were filed as 2-sample subsamples.
            tail = re.sub(r"\d*>[&]?\d*", " ", tail)
            nums = [int(x) for x in re.findall(r"(?<![\w.-])(-?\d{1,6})(?![\w.-])", tail)]
            limit = next((n for n in nums if n == -1 or 1 <= n <= 100000), None)
        json_out = _str(_JSON_OUT, command)
        background = bool(_BACKGROUND.search(command)) or bool(_handles(own_text))

        artifacts = {a for a in (json_out,) if a}
        artifacts |= set(re.findall(r"[\w./-]+\.log\b", command))
        # The evaluated checkpoint's own name: agents that read a score out of an
        # inspect_ai log rather than the file they asked for still tend to name
        # the candidate they were judging.
        model_path = _str(_MODEL_PATH, command)
        if model_path:
            leaf = model_path.rstrip("/").split("/")[-1]
            if len(leaf) >= 4:
                artifacts.add(leaf)
        if form in ("run_eval.sh", "own_wrapper"):
            # The wrapper builds its own output name from a positional tag
            # (``res_${T}.json``), so the tag is the only handle the launch
            # carries. Take every word-like positional as a candidate.
            outside = strip_heredocs(command)
            m = _LAUNCH_SH.search(outside)
            tail = outside[m.end():] if m else outside
            for arg in re.findall(r"[\w./-]{4,}", tail.split("|")[0].split(";")[0][:200]):
                artifacts.add(arg)
                artifacts.add(arg.rstrip("/").split("/")[-1])
        handles = _handles(own_text)

        got, s_i, s_ts, via, acc = False, None, None, None, None
        # A concatenated block can carry the previous evaluation's summary before
        # this one's. Reading the first number attributed a neighbour's score.
        direct = _accuracy_in(_own_slice(own_text, command))
        if refused:
            direct = None
        if direct is not None and not background:
            got, s_i, s_ts, via, acc = True, own.get("i"), own.get("ts"), "returned", direct
        else:
            start_i = event.get("i") or 0
            if refused:
                ordered = []
            # A launch that crashed and was relaunched writing the *same* output
            # file must not receive the relaunch's score. Three runs had a score
            # back-filled onto a launch that evaluated nothing, and one had six
            # of twelve rows carry a neighbour's number.
            superseded_at = None
            for later_use in uses[pos + 1 :]:
                lc = _command(later_use)
                if _form(lc, known) is None:
                    continue
                if json_out and _str(_JSON_OUT, lc) == json_out:
                    superseded_at = later_use.get("i")
                    break
            for later in ordered:
                if (later.get("i") or 0) <= start_i or later.get("type") != "tool_result":
                    continue
                if superseded_at is not None and (later.get("i") or 0) > superseded_at:
                    break
                text = later.get("text") or ""
                value = _accuracy_in(text)
                if value is None:
                    continue
                parent_use = next(
                    (u for u in uses if u.get("tool_use_id") == later.get("parent_tool_use")),
                    None,
                )
                parent_cmd = _command(parent_use) if parent_use else ""
                task_ids = set(_TASK_ID.findall(text))
                if handles and task_ids & handles:
                    channel = "task_id"
                elif task_ids and any(
                    a and a in text for a in artifacts
                ):
                    # A TaskOutput retrieval that names this launch's artifact.
                    # The harness returns background work this way 3,222 times
                    # across the corpus, and the launch does not always print a
                    # handle to match on, so the artifact is the link.
                    channel = "task_id"
                elif any(a in parent_cmd or a in text for a in artifacts):
                    channel = "artifact"
                else:
                    continue
                got, s_i, s_ts, via, acc = True, later.get("i"), later.get("ts"), channel, value
                break

        rows.append(
            {
                "run_id": run_id,
                "i": event.get("i"),
                "ts": event.get("ts"),
                "form": form,
                "model_path": model_path,
                "limit": limit,
                "max_connections": _int(_MAXCONN, command),
                "json_output_file": json_out,
                "mode": "background" if background else "foreground",
                "tier": tier_for(limit, benchmark, form),
                "n_calls": call_count(command, own_text),
                "got_signal": got,
                "signal_i": s_i,
                "signal_ts": s_ts,
                "signal_via": via,
                "accuracy": acc,
                "wait_s": _seconds(event.get("ts"), s_ts) if got else None,
                "command": command[:400],
            }
        )
        return rows


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLUMNS)).astype(DTYPES)


def empty() -> pd.DataFrame:
    return frame([])


def build(events_dir: Path | None = None, benchmarks: dict[str, str] | None = None) -> pd.DataFrame:
    """Evaluation events for every converted run under ``events_dir``."""
    root = Path(events_dir) if events_dir is not None else paths.events_dir("posttrainbench")
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for stream in sorted(root.glob("*.jsonl.gz")):
            run_id = stream.name[: -len(".jsonl.gz")]
            rows.extend(
                events_for_run(
                    run_id, list(_iter_events(stream)), (benchmarks or {}).get(run_id)
                )
            )
    df = frame(rows)
    return df.sort_values(["run_id", "i"], kind="stable").reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    return p


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).astype(DTYPES)


__all__ = [
    "COLUMNS",
    "DTYPES",
    "build",
    "empty",
    "events_for_run",
    "frame",
    "load",
    "save",
    "tier_for",
    "unwrap",
]
