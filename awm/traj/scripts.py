"""What each run's own scripts are for, learned from the trace that wrote them.

Both extractors originally found launches by filename: a trainer was
``…train….py``, an evaluation was ``evaluate.py`` or ``run_eval*.sh``. Agents do
not honour that. One run put its whole GRPO stage in ``work/grpo.py`` and wrapped
every evaluation in ``work/ev.sh``; by name-matching, that run trained 15% of the
time and never once evaluated. By its trace it trained about 61% of the time and
evaluated fourteen times, thirteen of which returned a score.

Naming is a convention, and a convention an agent never agreed to. What is
actually reliable is that a script an agent runs is a script the agent first
wrote, in the same trace, with the content visible — so the trace states what
each of its own scripts is for, and this reads that statement.

A definition is a ``Write`` tool call or a ``cat > path <<TAG`` heredoc. Its body
is classified by what it *does*: importing or instantiating a trainer makes the
file a trainer; invoking ``evaluate.py`` or ``inspect_eval`` makes it an
evaluation wrapper. A file can be both, and a few are — a script that trains and
then scores what it trained.

The classification is deliberately narrow. It fires on the trainer classes and
the evaluation entry point this benchmark actually uses, not on the word "train"
appearing somewhere in a comment, because a false positive here manufactures
launches that never happened and those are worse than the misses this replaces.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Instantiating or importing one of these makes a file a training script. The
#: bare verb ``.train()`` is included because a hand-rolled loop still trains,
#: but only alongside a transformers/trl import, so a comment cannot trip it.
_TRAINER_BODY = (
    re.compile(r"\b(?:SFTTrainer|GRPOTrainer|DPOTrainer|KTOTrainer|ORPOTrainer|PPOTrainer)\b"),
    re.compile(r"\bTrainer\s*\("),
    re.compile(r"\bfrom\s+trl\b|\bimport\s+trl\b"),
    re.compile(r"\.train\s*\(\s*\)", ),
)

#: Calling the benchmark's scorer, at any remove, makes a file an evaluator.
_EVAL_BODY = (
    re.compile(r"\bevaluate\.py\b"),
    re.compile(r"\binspect_eval\b|\bfrom\s+inspect_ai\b"),
)

#: Where a script gets defined: the Write tool, or a heredoc redirect.
_HEREDOC_DEF = re.compile(
    r"cat\s*>\s*['\"]?(?P<path>[\w./-]+\.(?:py|sh))['\"]?\s*<<-?\s*['\"]?(?P<tag>\w+)['\"]?"
    r"(?P<body>.*?)^(?P=tag)",
    re.S | re.M,
)

_BASH_LC = re.compile(r"^\s*/bin/bash\s+-lc\s+(['\"])(?P<inner>.*)\1\s*$", re.S)


def _unwrap(command: str) -> str:
    m = _BASH_LC.match(command)
    return m.group("inner") if m else command


def _roles(body: str) -> set[str]:
    roles: set[str] = set()
    if any(p.search(body) for p in _TRAINER_BODY):
        # ``.train()`` alone is too weak; require a training library nearby.
        if re.search(r"\b(?:trl|transformers|torch)\b", body) or re.search(
            r"\b(?:SFT|GRPO|DPO|KTO|ORPO|PPO)Trainer\b", body
        ):
            roles.add("trainer")
    if any(p.search(body) for p in _EVAL_BODY):
        roles.add("evaluator")
    return roles


def learn(events: list[dict[str, Any]]) -> dict[str, set[str]]:
    """``script path -> {"trainer", "evaluator"}``, from this run's own writes.

    Both the full path and the bare basename are registered, because an agent
    writes ``work/ev.sh`` and later runs ``bash work/ev.sh`` or ``./ev.sh``.
    """
    out: dict[str, set[str]] = {}

    def record(path: str, body: str) -> None:
        roles = _roles(body)
        if not roles or not path:
            return
        for key in (path, path.rstrip("/").split("/")[-1]):
            out.setdefault(key, set()).update(roles)

    for e in events:
        if e.get("type") != "tool_use":
            continue
        args = e.get("args") or {}
        if e.get("tool") == "Write":
            path = args.get("file_path") or ""
            if path.endswith((".py", ".sh")):
                record(path, args.get("content") or "")
            continue
        command = _unwrap(args.get("command") or "")
        for m in _HEREDOC_DEF.finditer(command):
            record(m.group("path"), m.group("body"))
    return out


def _invocation(command: str, path: str) -> bool:
    """Is ``path`` being *run* here, rather than edited or read?

    Matched on the basename with an optional directory prefix. A script written
    to ``/home/ben/task/work/ev.sh`` is invoked as ``bash work/ev.sh`` or
    ``./ev.sh``, and demanding the recorded spelling matches neither.
    Interpreter flags and leading environment assignments sit between the verb
    and the path (``LIGER=0 nohup python work/grpo.py``), so both are skipped.
    """
    leaf = re.escape(path.rstrip("/").split("/")[-1])
    ref = rf"(?:[\w./-]*/)?{leaf}"
    return bool(
        re.search(rf"\b(?:python3?|bash|sh|source)\s+(?:-[\w-]+\s+)*{ref}\b", command)
        or re.search(rf"(?:^|[;&|]\s*|\bnohup\s+)\./{ref}\b", command)
    )


def invoked(command: str, known: dict[str, set[str]], role: str) -> bool:
    """Does this command run one of the run's own scripts in that role?"""
    command = _unwrap(command)
    return any(
        role in roles and _invocation(command, path) for path, roles in known.items()
    )


def invoked_purely(command: str, known: dict[str, set[str]], role: str) -> bool:
    """As :func:`invoked`, but only for scripts whose *only* role is this one.

    A trainer that scores its own output afterwards carries both roles. Its
    launch is a training launch; reading it as an evaluation as well doubles
    the count and leaves every phantom evaluation with no score attached.
    """
    command = _unwrap(command)
    return any(
        roles == {role} and _invocation(command, path) for path, roles in known.items()
    )


def summary(known: dict[str, set[str]]) -> str:
    """One line naming what this run called its trainers and evaluators."""
    by_role: dict[str, list[str]] = {}
    for path, roles in sorted(known.items()):
        if "/" not in path:
            continue  # the basename alias; the full path already reads better
        for r in roles:
            by_role.setdefault(r, []).append(path)
    return json.dumps(by_role, ensure_ascii=False)


__all__ = ["invoked", "invoked_purely", "learn", "summary"]
