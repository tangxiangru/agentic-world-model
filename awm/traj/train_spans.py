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
_LAUNCH = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*[\w./-]*train[\w./-]*\.py"
    r"|\btorchrun\b"
    r"|\baccelerate\s+launch\b"
)

#: A command that *writes* a script naming a trainer or an evaluator is not
#: running one. Agents build wrappers with ``cat > run_eval.sh <<'EOF' … EOF``,
#: and the body holds a full command line; matching inside it invents launches
#: that never happened, inflating both the count and the share that returned a
#: score.
_HEREDOC_BODY = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?^\1", re.S | re.M)


def strip_heredocs(command: str) -> str:
    """The command without any here-document payload."""
    return _HEREDOC_BODY.sub(" ", command)

_OUT_DIR = re.compile(r"--out(?:put[-_]dir)?[= ]\s*['\"]?([^\s'\"]+)")

#: ``nohup`` or a bare trailing ``&``. ``2>&1`` and ``&&`` must not match, hence
#: the exclusion of ``>`` and ``&`` immediately before the ampersand.
_BACKGROUND = re.compile(r"\bnohup\b|[^&>]&\s*(?:$|\n|;|echo\b)")

#: Deliberately reduced runs, by what the command asks for.
#:
#: Not ``--max-steps``: that is how a real GRPO run is configured, and treating
#: it as a smoke marker labelled 51 multi-hour trainings as smoke tests. What
#: does mark intent is naming the artifact as a throwaway, or asking for a slice
#: of the data too small to train on.
_SMOKE_SAMPLES = re.compile(r"--max[-_]samples[= ]\s*(\d+)")
_SMOKE_DIR = re.compile(r"(?:^|/)(?:smoke|sanity|bench|debug|dry|tiny|test)\w*/?$", re.I)

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
_DISCARD_VERB = r"\brm\s+-[rf]{1,2}\s+[^;&|]*?"

#: ``pkill -f "train_sft.py"`` ends every training then running, and names the
#: script rather than the artifact, so an out_dir-keyed rule never sees it. One
#: run launched a training and killed it nine seconds later to free the GPU; the
#: run-end fallback had scored that as 6.29 hours.
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


def _is_launch(command: str, known: dict[str, set[str]] | None = None) -> bool:
    outside = strip_heredocs(command)
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
    return out_dir in command


def _target(out_dir: str) -> str:
    """Match the artifact itself or a checkpoint path inside it."""
    return re.escape(out_dir) + r"(?:/\S*)?"


def _consumes(command: str, out_dir: str) -> bool:
    tgt = _target(out_dir)
    return any(re.search(verb + tgt, command) for verb in _CONSUME_VERBS)


def _discards(command: str, out_dir: str) -> bool:
    """Only removal of the *whole* artifact abandons the training.

    ``rm -rf ckpt/grpo2/checkpoint-160`` is housekeeping — agents delete spent
    checkpoints to free disk while the run they belong to is still the one being
    submitted — so the artifact path must not continue into a subpath.
    """
    whole = re.escape(out_dir) + r"(?![\w/-])"
    return bool(re.search(_DISCARD_VERB + whole, command))


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
    results = _result_index(events)
    uses = [e for e in events if e.get("type") == "tool_use"]
    last_ts = next((e.get("ts") for e in reversed(events) if e.get("ts")), None)

    launches: list[tuple[int, dict[str, Any], str, str | None]] = []
    for pos, e in enumerate(uses):
        cmd = _command(e)
        if _is_launch(cmd, known):
            launches.append((pos, e, cmd, _out_dir(cmd)))

    rows: list[dict[str, Any]] = []
    for pos, event, cmd, out_dir in launches:
        ts_start = event.get("ts")
        background = _is_background(cmd)

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
                if out_dir and _discards(later_cmd, out_dir):
                    ts_end, end_reason = later.get("ts"), "discarded"
                    break
                if out_dir and _consumes(later_cmd, out_dir):
                    ts_end, end_reason = later.get("ts"), "consumed"
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
    """Per run: occupied training seconds, launches, and the naive sum."""
    df = spans if kind is None else spans[spans["kind"] == kind]
    rows = [
        {
            "run_id": run_id,
            "n_launches": len(g),
            "occupied_s": occupied_seconds(g),
            "sum_s": float(g["sec"].sum(skipna=True)),
        }
        for run_id, g in df.groupby("run_id", sort=True)
    ]
    return pd.DataFrame(rows, columns=["run_id", "n_launches", "occupied_s", "sum_s"])


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
