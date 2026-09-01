"""Reject annotations whose evidence does not exist.

An agent asked to categorise a change can be wrong, and no script can tell.
An agent asked to *point at* the event it read can be checked exactly, and that
is the whole design: every judgement carries ``(i, fragment)``, and this module
confirms that event ``i`` exists in the run's committed stream and that the
fragment appears in it verbatim. Fabricated locations — the failure mode that
makes bulk agent summarisation unusable — do not survive that.

What it does not do is equally important. A surviving pointer means the agent
read a real event, not that it read it correctly; that is what the double
annotation in the spec measures. This is the cheap mechanical floor under the
expensive statistical check, and the two answer different questions.

Matching normalises whitespace and unescapes the pipe that table rendering adds,
because an agent quoting from the brief quotes the rendered form. It does not
normalise anything else: case, punctuation and identifiers must match, or the
check would pass on paraphrase, which is the thing being excluded.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from awm import paths

#: Fragments shorter than this match too much to be evidence of anything.
MIN_FRAGMENT = 8


@dataclass(frozen=True)
class Problem:
    run_id: str
    table: str
    row: int
    reason: str
    detail: str


@dataclass
class Report:
    checked: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        return f"{self.checked} judgement(s) checked, {len(self.problems)} rejected"


def _norm(text: str) -> str:
    """Collapse whitespace and undo the pipe escaping the brief's tables add."""
    return re.sub(r"\s+", " ", text.replace("\\|", "|").replace("⏎", " ")).strip()


def _flatten(value: Any, into: list[str]) -> None:
    """Every string an argument structure holds, at any depth."""
    if isinstance(value, str):
        into.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _flatten(v, into)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _flatten(v, into)


def event_text(event: dict[str, Any]) -> str:
    """Everything of an event a fragment could legitimately have been quoted from.

    Argument *values*, never their JSON rendering. Serialising the args escapes
    every inner quote, so a fragment quoted verbatim from the command —
    ``pkill -f "train_sft.py"`` — would be searched against
    ``pkill -f \\"train_sft.py\\"`` and rejected as fabricated. That failure
    discarded valid judgements in three of four anchor runs.
    """
    parts = [event.get("text") or "", event.get("summary") or ""]
    _flatten(event.get("args"), parts)
    return _norm(" ".join(parts))


def stream_index(run_id: str, events_root: Path | None = None) -> dict[int, str]:
    """``i`` -> searchable text, for one committed run.

    ``i`` numbers each ``(run_id, agent_id)`` stream separately, so one run can
    hold several events at the same index — 180 of them in the champion run, one
    index carrying 33. Keeping only the last would reject valid pointers whose
    event happened to be overwritten by a sub-agent's placeholder, so every
    event at an index is concatenated and the fragment may match any of them.
    """
    root = Path(events_root) if events_root is not None else paths.events_dir("posttrainbench")
    path = root / f"{run_id}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[int, str] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            i = e.get("i")
            if i is not None:
                text = event_text(e)
                key = int(i)
                out[key] = f"{out[key]} {text}" if key in out else text
    return out


def _check_evidence(
    run_id: str, table: str, row: int, evidence: Any, index: dict[int, str]
) -> list[Problem]:
    if not isinstance(evidence, list) or not evidence:
        return [Problem(run_id, table, row, "no_evidence", "evidence is empty")]
    problems: list[Problem] = []
    for item in evidence:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            problems.append(Problem(run_id, table, row, "malformed", repr(item)[:120]))
            continue
        i, fragment = item
        try:
            i = int(i)
        except (TypeError, ValueError):
            problems.append(Problem(run_id, table, row, "bad_index", repr(i)[:60]))
            continue
        if i not in index:
            problems.append(Problem(run_id, table, row, "no_such_event", f"i={i}"))
            continue
        if not isinstance(fragment, str) or len(fragment.strip()) < MIN_FRAGMENT:
            problems.append(
                Problem(run_id, table, row, "fragment_too_short", repr(fragment)[:60])
            )
            continue
        if _norm(fragment) not in index[i]:
            problems.append(
                Problem(run_id, table, row, "fragment_not_found", f"i={i}: {fragment[:80]}")
            )
    return problems


#: Tables whose rows must each carry evidence, and the key holding it.
_EVIDENCED = ("changes", "trainings", "proposed_category", "definition_defect", "boundary_case")


def check_annotation(
    annotation: dict[str, Any], events_root: Path | None = None
) -> Report:
    """Every pointer in one run's annotation, against that run's stream."""
    run_id = annotation.get("run_id") or ""
    report = Report()
    index = stream_index(run_id, events_root)
    for table in _EVIDENCED:
        for row, entry in enumerate(annotation.get(table) or []):
            if not isinstance(entry, dict):
                report.problems.append(Problem(run_id, table, row, "malformed", repr(entry)[:120]))
                continue
            report.checked += 1
            report.problems.extend(
                _check_evidence(run_id, table, row, entry.get("evidence"), index)
            )
    # verifications point at events by index alone; the index must still exist.
    for row, entry in enumerate(annotation.get("verifications") or []):
        if not isinstance(entry, dict):
            continue
        report.checked += 1
        i = entry.get("i")
        if i is None or int(i) not in index:
            report.problems.append(
                Problem(run_id, "verifications", row, "no_such_event", f"i={i}")
            )
    return report


def check_files(paths_: Iterable[Path], events_root: Path | None = None) -> dict[str, Report]:
    """Check a batch of annotation files, keyed by run id."""
    out: dict[str, Report] = {}
    for p in paths_:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        data.setdefault("run_id", Path(p).stem)
        out[data["run_id"]] = check_annotation(data, events_root)
    return out


__all__ = [
    "MIN_FRAGMENT",
    "Problem",
    "Report",
    "check_annotation",
    "check_files",
    "event_text",
    "stream_index",
]
