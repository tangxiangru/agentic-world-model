"""Where each run touched a ``generation_config.json``, and what it wrote if the trace says.

C1 — the decoding configuration — turns on field-level detail: the `base_greedy`
comparison in the reference document looked like a one-field edit and was three,
because rewriting the file whole also dropped ``max_new_tokens: 2048``, which
vLLM reads. So the natural mechanical artifact would be a field-level diff.

**It cannot be built.** Measured over the 180 in-scope runs, of roughly 1,500
events that touch a ``generation_config.json`` only 41 carry parseable content —
29 Claude Code ``Write`` calls and 12 shell heredocs. The dominant form is a
``json.dump`` inside a Python heredoc (163 + 52), which is arbitrary code, and
codex publishes its file edits as ``file_change`` events holding a path and a
``kind`` and nothing else: **zero** codex config writes expose their content.
Executing the heredocs to find out is neither safe nor reproducible.

So this module does what the trace supports and stops there. It locates every
event that reads or writes one of these files, classifies the access, and
attaches the content when it is literally present. The field-level diff moves to
the judgment layer, where an agent reads the heredoc — which is right there in
the trace, legible to a reader and not to a parser.

That split is not a concession, it is the boundary working as intended: this is
precisely a question a script cannot answer and an agent can. What the module
still buys is the skeleton — an agent gets told which events to read rather than
searching a 900-event stream for them, and ``content_available`` records, per
family, how much of C1 rests on agent reading rather than on parsing.

The asymmetry is itself a result to carry forward: any C1 claim compared across
the two families is comparing a partly-parsed record against a wholly-read one.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from awm import paths

DTYPES: dict[str, str] = {
    "run_id": "string",
    "i": "Int64",
    "ts": "string",
    "tool": "string",
    "path": "string",
    "access": "string",
    "form": "string",
    "content_available": "boolean",
    "content": "string",
    "command": "string",
}

COLUMNS: tuple[str, ...] = tuple(DTYPES)

_GEN_CONFIG = re.compile(r"[\w./-]*generation_config\.json")

_BASH_LC = re.compile(r"^\s*/bin/bash\s+-lc\s+(['\"])(?P<inner>.*)\1\s*$", re.S)

#: ``cat > path/generation_config.json <<'EOF' … EOF`` — the one shell form whose
#: payload is literal JSON rather than code that computes JSON.
_HEREDOC = re.compile(
    r"cat\s*>\s*['\"]?(?P<path>[\w./-]*generation_config\.json)['\"]?\s*"
    r"<<\s*'?(?P<tag>\w+)'?(?P<body>.*?)^(?P=tag)",
    re.S | re.M,
)

_WRITES_VIA_CODE = re.compile(r"json\.dump|\.write\(|>\s*[\w./-]*generation_config\.json")
_READ_ONLY = re.compile(
    r"\b(?:cat|bat|sed\s+-n|grep|rg|head|tail|ls|stat|diff|md5sum)\b|json\.load|\.read\(\)"
)


def unwrap(command: str) -> str:
    m = _BASH_LC.match(command)
    return m.group("inner") if m else command


def _paths_in(text: str) -> list[str]:
    seen: list[str] = []
    for p in _GEN_CONFIG.findall(text):
        if p not in seen:
            seen.append(p)
    return seen


#: The channel that actually writes most generation configs. Agents package a
#: checkpoint for submission with a ``finalize``/``package``/``prep`` script that
#: writes the file as a side effect, so the path never appears on the command
#: line. Missing it left the decisive C1 change of two runs — setting
#: ``temperature: 0.0`` — absent from the table entirely.
_FINALIZER = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*(?:[\w./-]*/)?"
    r"(?:finalize|finalise|package|prep|prepare|export|publish)[\w-]*\.py\b"
)


def finalizer_call(command: str) -> str | None:
    """The finalizer invocation in this command, if any."""
    m = _FINALIZER.search(command)
    return m.group(0) if m else None


def _classify_shell(command: str) -> tuple[str, str]:
    """``(access, form)`` for a shell command that names a generation config."""
    if _HEREDOC.search(command):
        return "write", "heredoc"
    if re.search(r"json\.dump", command):
        return "write", "python_code"
    if re.search(r"\brm\b|\bmv\b|\bcp\b", command):
        return "write", "file_op"
    if _WRITES_VIA_CODE.search(command):
        return "write", "shell_redirect"
    if _READ_ONLY.search(command):
        return "read", "shell_read"
    return "unknown", "shell_other"


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def writes_for_run(run_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every access to a ``generation_config.json`` this run made."""
    rows: list[dict[str, Any]] = []
    for e in sorted(events, key=lambda x: (x.get("agent_id") or "", x.get("i") or 0)):
        if e.get("type") != "tool_use":
            continue
        tool = e.get("tool") or ""
        args = e.get("args") or {}
        blob = json.dumps(args, ensure_ascii=False)
        names_file = "generation_config.json" in blob
        runs_finalizer = tool in ("Bash", "command_execution") and bool(
            finalizer_call(unwrap(args.get("command") or ""))
        )
        if not names_file and not runs_finalizer:
            continue

        base = {"run_id": run_id, "i": e.get("i"), "ts": e.get("ts"), "tool": tool}

        if tool == "Write":
            path = args.get("file_path") or ""
            if not _GEN_CONFIG.search(path):
                continue
            content = args.get("content")
            rows.append({**base, "path": path, "access": "write", "form": "write_tool",
                         "content_available": bool(content), "content": content,
                         "command": None})
            continue

        if tool == "Read":
            path = args.get("file_path") or ""
            if _GEN_CONFIG.search(path):
                rows.append({**base, "path": path, "access": "read", "form": "read_tool",
                             "content_available": False, "content": None, "command": None})
            continue

        if tool == "Edit":
            path = args.get("file_path") or ""
            if _GEN_CONFIG.search(path):
                rows.append({**base, "path": path, "access": "write", "form": "edit_tool",
                             "content_available": bool(args.get("new_string")),
                             "content": args.get("new_string"), "command": None})
            continue

        if tool == "file_change":
            # Codex publishes a path and a kind. No content, ever.
            for change in args.get("changes") or []:
                p = change.get("path") or ""
                if _GEN_CONFIG.search(p):
                    rows.append({**base, "path": p, "access": "write", "form": "file_change",
                                 "content_available": False, "content": None, "command": None})
            continue

        command = unwrap(args.get("command") or "")
        if not command:
            continue
        if "generation_config.json" not in command:
            if finalizer_call(command):
                rows.append({**base, "path": None, "access": "write",
                             "form": "finalizer", "content_available": False,
                             "content": None, "command": command[:400]})
            continue
        access, form = _classify_shell(command)
        heredoc = _HEREDOC.search(command)
        for p in _paths_in(command):
            rows.append({
                **base,
                "path": p,
                "access": access,
                "form": form,
                "content_available": bool(heredoc) and heredoc.group("path") == p,
                "content": heredoc.group("body").strip() if heredoc and heredoc.group("path") == p
                else None,
                "command": command[:400],
            })
    return rows


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLUMNS)).astype(DTYPES)


def empty() -> pd.DataFrame:
    return frame([])


def build(events_dir: Path | None = None) -> pd.DataFrame:
    root = Path(events_dir) if events_dir is not None else paths.events_dir("posttrainbench")
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for stream in sorted(root.glob("*.jsonl.gz")):
            run_id = stream.name[: -len(".jsonl.gz")]
            rows.extend(writes_for_run(run_id, list(_iter_events(stream))))
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
    "finalizer_call",
    "DTYPES",
    "build",
    "empty",
    "frame",
    "load",
    "save",
    "unwrap",
    "writes_for_run",
]
