"""How long each training actually occupied the box.

The question this answers is "where did the ten hours go", and the honest unit
for that is **wall clock the agent spent waiting**, not the number a trainer
prints about itself. So a span here runs from the command that launched a
training to the moment that training's output was first *used*, and it counts
aborted attempts too: an OOM relaunch three minutes later still cost three
minutes of the budget.

Why this exists at all. The obvious extractor reads ``train_runtime`` — the
summary line HuggingFace's Trainer prints when it finishes — out of the
trajectory. That silently measures the wrong population. A long training is
launched in the background (``nohup … &``) with its output redirected to a log
file, so its summary line lands in ``logs/sft1.log`` and reaches the trajectory
only if the agent later cats the whole file, which agents mostly do not. A
short smoke test is run in the foreground and its summary comes straight back.
The result is an extractor that reliably captures smoke tests and misses real
training: on the champion GSM8K run it reported 138 seconds, being the smoke
test's genuine ``train_runtime``, for a run that trained for over four hours.

So pairing is by artifact, not by summary line:

*   **Foreground** launches need no heuristic. The harness timestamps both the
    ``tool_use`` and its ``tool_result``, and the agent was blocked in between,
    so the span is exact.
*   **Background** launches return immediately. Their span ends at the first
    later command that *consumes* the artifact — hands it to ``finalize.py``,
    an evaluation, a soup, or the next training's ``--model`` — or at the next
    launch writing the same ``--out``, which means this attempt was abandoned.

That last distinction is the one that has to be got right. Agents poll a
running training constantly (``ls ckpt/grpo1``, ``tail logs/grpo1.log``,
``until grep -q "^saved ckpt" …``), and treating the first of those as the end
cuts a 1.8-hour GRPO run off at 1.0 hours. Inspecting an artifact is not
finishing with it; only consuming it is.

``kind`` is decided by what the command asks for (``--max-samples``,
``--max-steps``, an output directory named ``smoke``/``sanity``/…), never by how
long the span turned out to be. Classifying by duration is what let a 138-second
smoke test be read as a run's entire training history.

``train_runtime_s`` is kept when the trajectory happened to show it. It is not
the measurement — ``sec`` is — but where both exist their ratio is the loading
and tokenisation overhead the agent also waited through, and it is what lets the
foreground spans calibrate the background ones.
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

#: Column -> pandas dtype, in table order. The whole contract of the table.
DTYPES: dict[str, str] = {
    "run_id": "string",
    "i": "Int64",
    "out_dir": "string",
    "kind": "string",
    "mode": "string",
    "ts_start": "string",
    "ts_end": "string",
    "sec": "Float64",
    "end_reason": "string",
    "train_runtime_s": "Float64",
    "command": "string",
}

COLUMNS: tuple[str, ...] = tuple(DTYPES)

#: A trainer is *invoked* only when the script is python's argument. This is
#: what separates a launch from the many commands that merely name one —
#: ``sed -i 's/…/' train_grpo.py``, ``ps aux | grep -c train_sft``,
#: ``tail logs/sft1.log`` — none of which put the script after ``python``.
#: The basename must *begin* with ``train``: ``prepare_training_data.py`` merely
#: contains the word and builds a corpus on CPU.
_LAUNCH = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*(?:[\w./-]*/)?train[\w-]*\.py"
    r"|\btorchrun\b"
    r"|\baccelerate\s+launch\b"
)

#: ``pgrep -f "python train_sft.py --model x"`` and ``ps aux | grep`` quote a
#: whole command line to *watch* it. The quoted text satisfies every launch
#: pattern, so process inspection has to be excluded before matching.
#: ``ps -eo cmd | rg 'python train_sft.py.*ckpt'`` watches a training; the
#: quoted pattern satisfies every launch form. The pipe puts ``ps`` on the left
#: and the search on the right, so the whole segment counts as inspection —
#: and ripgrep is as common as grep in these traces.
_INSPECTS_PROCESS = re.compile(r"\b(?:pgrep|pkill|ps|rg|grep|awk)\b")

#: A command that *writes* a script naming a trainer or an evaluator is not
#: running one. Agents build wrappers with ``cat > run_eval.sh <<'EOF' … EOF``,
#: and the body holds a full command line; matching inside it invents launches
#: that never happened, inflating both the count and the share that returned a
#: score.
_HEREDOC_BODY = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?^\1", re.S | re.M)


def strip_heredocs(command: str) -> str:
    """The command without any here-document payload."""
    return _HEREDOC_BODY.sub(" ", command)

_OUT_DIR = re.compile(r"--out(?:put)?(?:[-_]dir)?[= ]\s*['\"]?([^\s'\"]+)")

#: ``nohup`` or a bare trailing ``&``. ``2>&1`` and ``&&`` must not match, hence
#: the exclusion of ``>`` and ``&`` immediately before the ampersand.
_BACKGROUND = re.compile(r"\bnohup\b|[^&>]&\s*(?:$|\n|;|echo\b)")

#: The harness moves a foreground command off the foreground when it outruns the
#: bash timeout, and says so in the result. Reading only the command text scored
#: a 44-minute SFT as ``foreground / 0.00h / returned``.
_MOVED_TO_BACKGROUND = re.compile(r"background \(ID: \w+\)|background with ID: \w+")

#: Deliberately reduced runs, by what the command asks for.
#:
#: Not ``--max-steps``, in any form. It is how a real GRPO run is configured,
#: and a step cap says nothing about intent: one run capped a full-parameter
#: fine-tune and an 8.5-minute GRPO stage whose checkpoints were merged and
#: evaluated. Both readings of the flag have now been tried and both misfile
#: real trainings, so intent comes from the artifact's name or a slice of data
#: too small to train on, and from nothing else.
_SMOKE_SAMPLES = re.compile(r"--max[-_]samples[= ]\s*(\d+)")
#: The marker may sit anywhere in the basename: one run named its throwaways
#: ``work/sft_smoke`` and ``sft_smoke2``, and anchoring the pattern to the start
#: scored every one of them as a real training.
_SMOKE_DIR = re.compile(
    r"(?:^|/)[\w-]*(?:smoke|sanity|debug|dryrun|tiny|scratch)[\w-]*/?$"
    r"|(?:^|/)(?:test|tst)[\w-]*/?$"
    r"|^/tmp/",
    re.I,
)

#: Rows this small are a pipeline check, not a recipe. It bounds the *requested
#: workload*, never the observed duration — classifying by how long a span turned
#: out to run is what let a 138-second smoke test stand in for a run's whole
#: training history.
_SMOKE_MAX_SAMPLES = 5000

#: Using an artifact, as opposed to peeking at it. Each verb must take *this*
#: artifact as its argument: co-occurrence is not enough, or
#: ``rm -rf ckpt/grpo2 && cp -r eval_grpo150 ckpt/grpo150_bf16`` reads as a
#: consumption of ``ckpt/grpo2`` when it is that artifact's deletion.
#: ``--model X`` counts — the next stage training from a checkpoint is proof the
#: previous stage finished.
_CONSUME_VERBS = (
    r"finalize\.py\s+",
    r"run_eval\.sh\s+",
    r"--model[= ]\s*",
    r"--model-path[= ]\s*",
    r"cp\s+-r\s+",
    r"soup\.py\b[^;&|]*?",
)

#: Deleting the artifact ends the span too, but it is an abandonment, not a use.
#: Newlines end a shell statement too: without excluding them a first-line
#: ``rm -rf …/checkpoint-*`` reached a third-line ``--model …`` and read a
#: consumption as a discard.
_DISCARD_VERB = r"\brm\s+-[rf]{1,2}\s+[^;&|\n]*?"

#: ``pkill -f "train_sft.py"`` ends every training then running, and names the
#: script rather than the artifact, so an out_dir-keyed rule never sees it. One
#: run launched a training and killed it nine seconds later to free the GPU; the
#: run-end fallback had scored that as 6.29 hours.
#: Where the artifact goes when the command line does not say. A training script
#: with its ``--output-dir`` defaulted in code leaves no destination on the
#: command line, and the out_dir-keyed rules then have nothing to pair against,
#: so the span runs to the end of the run. But the trainer announces the
#: destination when it finishes, and that announcement reaches the trajectory:
#: 1,090 such lines across 154 of the 234 in-scope runs. One run's 1.85h SFT was
#: being scored as 6.31h for exactly this reason.
_SAVED_TO = re.compile(
    r"(?:Saved|Saving|saved|Wrote|wrote)\s+(?:final\s+)?(?:model\s+)?(?:to|->)\s+['\"]?([\w./-]+)"
)

#: When the script announces nothing either, the destination still surfaces as
#: the constant the agent wrote into the trainer. One run kept every output path
#: in ``OUTPUT_DIR = "/home/ben/task/sft_run3"``; all five of its trainings fell
#: through to the run-end bound and its occupancy read 98% against a measured
#: 88%.
_OUT_CONST = re.compile(
    r"(?:OUTPUT_DIR|OUT_DIR|SAVE_DIR|output_dir)\s*=\s*['\"]([\w./-]+)['\"]"
)

#: The log the launch redirects into. When the destination is hidden in the
#: script, this is still on the command line and the agent polls it while the
#: training runs — so it identifies the training even when the artifact cannot,
#: and it is the more reliable of the two fallbacks because it comes from this
#: launch rather than from anywhere in the run.
_LOG_TARGET = re.compile(r">>?\s*([\w./-]+\.log)\b")

_KILL = re.compile(r"\b(?:pkill|killall)\b[^;&|]*?(train[\w./-]*\.py|torchrun|accelerate)")

_TRAIN_RUNTIME = re.compile(r"train_runtime['\"]?\s*[:=]\s*([0-9.]+)")


def _parse_ts(ts: Any) -> datetime | None:
    # Reads both raw events (``None``) and the frame's nullable strings
    # (``pd.NA``), whose truthiness raises rather than being falsy.
    if ts is None or ts is pd.NA or not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds(a: str | None, b: str | None) -> float | None:
    ta, tb = _parse_ts(a), _parse_ts(b)
    if ta is None or tb is None:
        return None
    return (tb - ta).total_seconds()


def _command(event: dict[str, Any]) -> str:
    """The shell text of a ``tool_use``, or empty for every other event."""
    if event.get("type") != "tool_use":
        return ""
    args = event.get("args") or {}
    return args.get("command") or ""


def _launch_segments(command: str) -> list[str]:
    """Shell segments, so a watcher in one does not vouch for a launch in another."""
    return [seg for seg in re.split(r"&&|\|\||[;&|]|\n", command) if seg.strip()]


#: A trainer invoked with ``--skip-train`` builds or checks data and never takes
#: an optimiser step. One run had four of its eight recorded trainings be these.
#: ``--help`` prints the interface and exits. Reading a trainer's flags is a
#: first-tier static check, not a training.
#: Flags that mean this invocation takes no optimiser step, or keeps nothing if
#: it does. ``--prepare-only`` builds a dataset; ``--help`` prints flags;
#: ``--save-strategy no --no-final-save`` runs a throughput probe that cannot
#: produce a candidate. One run had seven of its fourteen recorded trainings be
#: these.
_NO_TRAIN = re.compile(
    r"--skip[-_]train\b|--dry[-_]run\b|--no[-_]train\b|--prepare[-_]only\b"
    r"|--data[-_]only\b|(?:^|\s)--help\b"
)

#: Keeps no weights: a probe, not a candidate.
_KEEPS_NOTHING = re.compile(r"--no[-_]final[-_]save\b")

#: A background launch is normally answered the instant the shell backgrounds
#: it, so an error on that result means the shell died *before* the launch. The
#: form that produces it: ``pkill -f "train.py --data v2"; nohup python train.py
#: --data v2 &`` — the pattern matches the shell's own command line, the shell
#: is signalled, and the launch after the semicolon never runs. Two of one run's
#: eight recorded launches were these, and because their spans ended within
#: seconds they were then read as crashes. The rule is deliberately confined to
#: background launches: a foreground trainer that ran and raised also answers
#: with an error, and that one *is* a training.
_SHELL_DIED = re.compile(r"\bExit code (?!0\b)\d+|\bkilled by signal\b", re.I)


def _is_launch(command: str, known: dict[str, set[str]] | None = None) -> bool:
    outside = strip_heredocs(command)
    if _NO_TRAIN.search(outside):
        return False
    if _KEEPS_NOTHING.search(outside) and re.search(
        r"--save[-_]strategy[= ]\s*(?:no|none)\b", outside
    ):
        return False
    segments = [s for s in _launch_segments(outside) if not _INSPECTS_PROCESS.search(s)]
    outside = " ; ".join(segments)
    if _LAUNCH.search(outside):
        return True
    # Naming is a convention agents never agreed to: one run put its whole GRPO
    # stage in ``work/grpo.py``. What the run's own writes say the script is for
    # is the reliable signal.
    return bool(known) and scripts.invoked(outside, known, "trainer")


def _out_dir(command: str) -> str | None:
    m = _OUT_DIR.search(command)
    return m.group(1).rstrip("/") if m else None


def _is_background(command: str) -> bool:
    return bool(_BACKGROUND.search(command))


def _kind(command: str, out_dir: str | None) -> str:
    if out_dir and _SMOKE_DIR.search(out_dir):
        return "smoke"
    m = _SMOKE_SAMPLES.search(command)
    if m and int(m.group(1)) <= _SMOKE_MAX_SAMPLES:
        return "smoke"
    return "real"


def _mentions(command: str, out_dir: str) -> bool:
    """Does this command name that artifact (or a checkpoint inside it)?"""
    return bool(re.search(_target(out_dir), command))


def _target(out_dir: str) -> str:
    """Match the artifact itself or a checkpoint path inside it.

    Matched on the basename with an optional directory prefix, because the two
    spellings rarely agree: a script fixes its destination as
    ``/home/ben/task/work/sft_run1`` and the evaluation that consumes it writes
    ``work/sft_run1``. Demanding the recorded spelling left five runs' spans
    unpaired and charged them to the end of the run.

    Bounded on both sides. Without a left boundary ``sft_full`` matched inside
    ``runs/grader_sft_full.json``, so deleting a stale result file read as
    discarding the checkpoint that run went on to submit.
    """
    leaf = re.escape(out_dir.rstrip("/").split("/")[-1])
    return r"(?<![\w./-])(?:[\w./-]*/)?" + leaf + r"(?:/\S*)?"


def _consumes(command: str, out_dir: str) -> bool:
    tgt = _target(out_dir)
    return any(re.search(verb + tgt, command) for verb in _CONSUME_VERBS)


def _discards(command: str, out_dir: str) -> bool:
    """Only removal of the *whole* artifact abandons the training.

    ``rm -rf ckpt/grpo2/checkpoint-160`` is housekeeping — agents delete spent
    checkpoints to free disk while the run they belong to is still the one being
    submitted — so the artifact path must not continue into a subpath.
    """
    leaf = re.escape(out_dir.rstrip("/").split("/")[-1])
    whole = r"(?<![\w./-])(?:[\w./-]*/)?" + leaf + r"(?![\w./-])"
    return bool(re.search(_DISCARD_VERB + whole, command))


def ordered_after(events: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    """Results that arrive after this event, in stream order."""
    start = event.get("i") or 0
    return [
        e for e in events
        if e.get("type") == "tool_result" and (e.get("i") or 0) > start
    ]


def _announced_dir(
    later_uses: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    later_results: list[dict[str, Any]],
) -> str | None:
    """The destination the trainer printed, when the command line omitted it."""
    for r in later_results[:400]:
        m = _SAVED_TO.search(r.get("text") or "")
        if m:
            return m.group(1).rstrip("/")
    return None


def _constant_dir(events: list[dict[str, Any]], before_i: int) -> str | None:
    """The output path the agent hard-coded into the trainer it wrote."""
    best: str | None = None
    for e in events:
        if e.get("type") != "tool_use" or (e.get("i") or 0) >= before_i:
            continue
        args = e.get("args") or {}
        for text in (args.get("content"), args.get("new_string"), args.get("command")):
            if not isinstance(text, str):
                continue
            for m in _OUT_CONST.finditer(text):
                best = m.group(1).rstrip("/")
    return best


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _result_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``tool_use_id`` -> the ``tool_result`` answering it."""
    out: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("type") == "tool_result" and e.get("parent_tool_use"):
            out.setdefault(e["parent_tool_use"], e)
    return out


def _train_runtime_in(text: str | None) -> float | None:
    if not text:
        return None
    vals = [float(v) for v in _TRAIN_RUNTIME.findall(text)]
    return max(vals) if vals else None


def spans_for_run(run_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every training this run launched, with the wall clock it occupied."""
    events = sorted(events, key=lambda e: (e.get("agent_id") or "", e.get("i") or 0))
    known = scripts.learn(events)
    dests = scripts.destinations(events)
    results = _result_index(events)
    uses = [e for e in events if e.get("type") == "tool_use"]
    last_ts = next((e.get("ts") for e in reversed(events) if e.get("ts")), None)

    launches: list[tuple[int, dict[str, Any], str, str | None]] = []
    for pos, e in enumerate(uses):
        cmd = _command(e)
        if not _is_launch(cmd, known):
            continue
        out = _out_dir(cmd)
        if out is None:
            # Ordered by how specific the evidence is to *this* launch. The
            # script's own fixed destination is best: it comes from the file
            # being run. Then the log this command redirects into, which the
            # agent polls while it waits. Only then anything found elsewhere in
            # the run, which could belong to a different training.
            out = scripts.destination_for(cmd, dests)
        if out is None:
            m = _LOG_TARGET.search(strip_heredocs(cmd))
            out = m.group(1) if m else None
        if out is None:
            out = _announced_dir(uses[pos + 1 :], results, ordered_after(events, e))
        if out is None:
            out = _constant_dir(events, e.get("i") or 0)
        launches.append((pos, e, cmd, out))

    rows: list[dict[str, Any]] = []
    for pos, event, cmd, out_dir in launches:
        ts_start = event.get("ts")
        own_result = results.get(event.get("tool_use_id") or "")
        background = _is_background(cmd) or bool(
            _MOVED_TO_BACKGROUND.search((own_result or {}).get("text") or "")
        )
        if (
            background
            and own_result is not None
            and own_result.get("is_error")
            and _SHELL_DIED.search(own_result.get("text") or "")
        ):
            continue

        if not background:
            result = results.get(event.get("tool_use_id") or "")
            ts_end = result.get("ts") if result else None
            if ts_end:
                end_reason = "returned"
            elif result is not None:
                # The result is there, the release just carries no clock — the
                # PostTrainBench codex runs and the old_container claude batch
                # publish no ``ts`` at all. Never synthesise one; these runs are
                # simply outside every duration statistic.
                end_reason = "untimed"
            else:
                end_reason = "no_result"
            runtime = _train_runtime_in(result.get("text") if result else None)
        else:
            ts_end, end_reason, runtime = last_ts, "run_end", None
            window_texts: list[str] = []
            last_seen: str | None = None
            for later in uses[pos + 1 :]:
                later_cmd = _command(later)
                if out_dir and _is_launch(later_cmd, known) and _out_dir(later_cmd) == out_dir:
                    ts_end, end_reason = later.get("ts"), "superseded"
                    break
                if _KILL.search(later_cmd):
                    ts_end, end_reason = later.get("ts"), "killed"
                    break
                # Use beats deletion when one command does both: agents clear
                # spent checkpoints and then hand the directory to the next
                # stage in the same breath.
                if out_dir and _consumes(later_cmd, out_dir):
                    ts_end, end_reason = later.get("ts"), "consumed"
                    break
                if out_dir and _discards(later_cmd, out_dir):
                    ts_end, end_reason = later.get("ts"), "discarded"
                    break
                if out_dir and _mentions(later_cmd, out_dir):
                    last_seen = later.get("ts") or last_seen
                    r = results.get(later.get("tool_use_id") or "")
                    if r is not None:
                        window_texts.append(r.get("text") or "")
            else:
                # Nothing ever used it. Running to the end of the run is an
                # upper bound that, summed over several such launches, exceeds
                # the budget many times over; the last time the agent looked at
                # it is the longest occupancy the trajectory actually shows.
                if last_seen is not None:
                    ts_end, end_reason = last_seen, "last_seen"
            runtime = max(
                (v for v in (_train_runtime_in(t) for t in window_texts) if v is not None),
                default=None,
            )

        rows.append(
            {
                "run_id": run_id,
                "i": event.get("i"),
                "out_dir": out_dir,
                "kind": _kind(cmd, out_dir),
                "mode": "background" if background else "foreground",
                "ts_start": ts_start,
                "ts_end": ts_end,
                "sec": _seconds(ts_start, ts_end),
                "end_reason": end_reason,
                "train_runtime_s": runtime,
                "command": cmd[:400],
            }
        )
    return rows


def occupied_seconds(spans: pd.DataFrame) -> float:
    """Wall clock these spans occupied, counting overlap once.

    Summing ``sec`` answers "how much training was launched", which is not the
    same question and is not bounded by the budget: agents run a training while
    an earlier one is still going, and on 28% of runs the naive sum exceeded the
    run's own wall clock, once by 7.5x. The box was busy once, so the budget
    share has to be the union of the intervals, not their total.
    """
    iv = [
        (_parse_ts(a), _parse_ts(b))
        for a, b in zip(spans["ts_start"], spans["ts_end"])
        if _parse_ts(a) is not None and _parse_ts(b) is not None
    ]
    iv = sorted((a, b) for a, b in iv if b > a)
    total = 0.0
    cur_start, cur_end = None, None
    for start, end in iv:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += (cur_end - cur_start).total_seconds()
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    if cur_end is not None:
        total += (cur_end - cur_start).total_seconds()
    return total


def occupancy_by_run(spans: pd.DataFrame, kind: str | None = "real") -> pd.DataFrame:
    """Per run: occupied training seconds, launches, the naive sum, and the span.

    ``span_s`` — first event to last — is the honest denominator for a budget
    share. The index's ``duration_s`` is what the harness recorded, and for a
    run whose session died early that is the *budget*, not the elapsed time: 5%
    of the corpus has the two differing by more than 1.5x and three runs by more
    than 3x, the worst claiming 10.08 hours against 1.79 of events. Dividing by
    the budget on those runs understates training occupancy several-fold.
    """
    df = spans if kind is None else spans[spans["kind"] == kind]
    rows = []
    for run_id, g in df.groupby("run_id", sort=True):
        starts = [t for t in (_parse_ts(x) for x in g["ts_start"]) if t is not None]
        ends = [t for t in (_parse_ts(x) for x in g["ts_end"]) if t is not None]
        rows.append({
            "run_id": run_id,
            "n_launches": len(g),
            "occupied_s": occupied_seconds(g),
            "sum_s": float(g["sec"].sum(skipna=True)),
            "span_s": (max(ends) - min(starts)).total_seconds() if starts and ends else None,
        })
    return pd.DataFrame(rows, columns=["run_id", "n_launches", "occupied_s", "sum_s", "span_s"])


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return df.astype(DTYPES)


def empty() -> pd.DataFrame:
    return frame([])


def build(
    events_dir: Path | None = None, run_ids: set[str] | None = None
) -> pd.DataFrame:
    """Spans for every converted run under ``events_dir``, sorted by run then ``i``."""
    root = Path(events_dir) if events_dir is not None else paths.events_dir("posttrainbench")
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for stream in sorted(root.glob("*.jsonl.gz")):
            run_id = stream.name[: -len(".jsonl.gz")]
            if run_ids is not None and run_id not in run_ids:
                continue
            rows.extend(spans_for_run(run_id, list(_iter_events(stream))))
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
    "frame",
    "load",
    "save",
    "occupancy_by_run",
    "occupied_seconds",
    "spans_for_run",
]
