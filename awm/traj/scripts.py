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
#: Running the scorer, not borrowing a type from it. ``from inspect_ai.model
#: ._openai import openai_chat_tool_param`` is how a *data builder* reproduces
#: the inference-time tool schema; matching the bare import made seven of one
#: run's sixteen evaluation rows be `build_sft.py`, each handed a later
#: evaluation's score.
_EVAL_BODY = (
    re.compile(r"\bevaluate\.py\b"),
    re.compile(r"\binspect_eval\s*\(|\binspect_ai\s*\.\s*eval\s*\("
               r"|\bfrom\s+inspect_ai\s+import\s+[^\n]*\beval\b"),
)

#: Writing a decoding config makes a file a config writer. Agents package this
#: as a helper — ``set_gen_config.py sft_run1 0.0`` — and then the filename
#: never appears on any command line again. Six real writes in one run were
#: invisible for that reason, while a memory note that merely mentioned the
#: filename was recorded as a write.
_CONFIG_BODY = (
    re.compile(r"generation_config\.json"),
    re.compile(r"\bGenerationConfig\b.*?\bsave_pretrained\b", re.S),
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


#: Docstrings and comments describe a file; they do not make it one. A data
#: builder opening with ``"""Build SFT data in the exact format used by
#: evaluate.py."""`` was classified an evaluator, and its launch then collected
#: a neighbouring evaluation's score.
_DOCSTRING = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
_COMMENT = re.compile(r"(?m)(?<!['\"])#[^\n]*")


def _code_only(body: str) -> str:
    """The body with its prose removed, so a mention cannot pose as a call."""
    return _COMMENT.sub(" ", _DOCSTRING.sub(" ", body))


def _roles(body: str) -> set[str]:
    body = _code_only(body)
    roles: set[str] = set()
    if any(p.search(body) for p in _TRAINER_BODY):
        # ``.train()`` alone is too weak; require a training library nearby.
        if re.search(r"\b(?:trl|transformers|torch)\b", body) or re.search(
            r"\b(?:SFT|GRPO|DPO|KTO|ORPO|PPO)Trainer\b", body
        ):
            roles.add("trainer")
    if any(p.search(body) for p in _EVAL_BODY):
        roles.add("evaluator")
    if any(p.search(body) for p in _CONFIG_BODY):
        roles.add("config_writer")
    return roles


#: A trainer named the conventional way. A shell wrapper calling one is a
#: trainer too, transitively, even though its own body holds no trl import.
_CALLS_TRAINER = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*(?:[\w./-]*/)?train[\w-]*\.py"
    r"|\btorchrun\b|\baccelerate\s+launch\b"
)


#: Where a script writes, when its caller does not say. An agent that wrote the
#: trainer usually fixed the destination inside it — as an argparse default, a
#: module constant, or a literal passed to ``save_pretrained`` — and then invoked
#: it bare. Five annotators reported the same consequence independently: with no
#: artifact to pair against, the span ran to the end of the run, in one case
#: charging 19.3h of training to a 7.5h budget.
_DEST = (
    re.compile(r"""--out(?:put)?(?:[-_]dir)?['\"]?\s*,[^)]*?default\s*=\s*['\"]([\w./-]+)['\"]"""),
    re.compile(r"""(?:OUTPUT_DIR|OUT_DIR|SAVE_DIR|FINAL_DIR|output_dir|final_dir|out_dir)\s*=\s*['\"]([\w./-]+)['\"]"""),
    re.compile(r"""save_pretrained\(\s*['\"]([\w./-]+)['\"]"""),
)


def destinations(events: list[dict[str, Any]]) -> dict[str, str]:
    """``script path -> the output directory its own source fixes``."""
    out: dict[str, str] = {}

    def record(path: str, body: str) -> None:
        if not path or "trainer" not in _roles(body) and not _CALLS_TRAINER.search(body):
            return
        for pat in _DEST:
            m = pat.search(body)
            if m:
                dest = m.group(1).rstrip("/")
                for key in (path, path.rstrip("/").split("/")[-1]):
                    out.setdefault(key, dest)
                return

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


def destination_for(command: str, dests: dict[str, str]) -> str | None:
    """The destination of whichever known script this command runs."""
    command = _unwrap(command)
    for path, dest in dests.items():
        if _invocation(command, path):
            return dest
    return None


def learn(events: list[dict[str, Any]], rounds: int = 3) -> dict[str, set[str]]:
    """``script path -> {"trainer", "evaluator"}``, from this run's own writes.

    Both the full path and the bare basename are registered, because an agent
    writes ``work/ev.sh`` and later runs ``bash work/ev.sh`` or ``./ev.sh``.

    Roles propagate through wrappers. One run drove every training through
    ``nohup bash pipeline.sh`` — a shell file that trains, finalises and
    evaluates in sequence — and because a shell wrapper's own body holds no
    ``trl`` import, seven of that run's eight trainings went unrecorded and its
    training occupancy read as 2.9% instead of over 80%. So a script whose body
    *invokes* a trainer is itself a trainer, and the resolution repeats so a
    wrapper around a wrapper still resolves.
    """
    out: dict[str, set[str]] = {}
    bodies: dict[str, str] = {}

    def record(path: str, body: str) -> None:
        if not path:
            return
        for key in (path, path.rstrip("/").split("/")[-1]):
            bodies[key] = bodies.get(key, "") + "\n" + body
            roles = _roles(body)
            if roles:
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

    # A body that calls a conventionally named trainer, or any script already
    # known to be one, makes this file a trainer as well.
    for _ in range(rounds):
        grew = False
        for path, body in bodies.items():
            if "trainer" in out.get(path, set()):
                continue
            calls = _CALLS_TRAINER.search(body) or any(
                "trainer" in roles and _invocation(body, other)
                for other, roles in out.items()
            )
            if calls:
                out.setdefault(path, set()).add("trainer")
                grew = True
        if not grew:
            break
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
    """As :func:`invoked`, but only for scripts whose *only* role is this one."""
    command = _unwrap(command)
    return any(
        roles == {role} and _invocation(command, path) for path, roles in known.items()
    )


def invoked_without_training(command: str, known: dict[str, set[str]], role: str) -> bool:
    """As :func:`invoked`, excluding scripts that also launch a training.

    A trainer that scores its own output afterwards carries both roles, and its
    launch is a training launch; counting it as an evaluation too doubles the
    count and leaves a phantom evaluation with no score.

    Demanding the role be the *only* one was too strong. An agent's thin
    evaluation wrapper commonly stamps the decoding config before scoring, so
    it carries ``config_writer`` as well — and one run lost 12 of its 15
    evaluations that way, its fourth-tier count reading four times low. Only
    ``trainer`` disqualifies.
    """
    command = _unwrap(command)
    return any(
        role in roles and "trainer" not in roles and _invocation(command, path)
        for path, roles in known.items()
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


__all__ = [
    "destination_for",
    "destinations",
    "invoked",
    "invoked_purely",
    "invoked_without_training",
    "learn",
    "summary",
]
