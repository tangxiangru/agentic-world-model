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
from awm.traj import scripts

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
    """Every generation-config path named, degenerate ones dropped.

    A path built at runtime — ``sys.argv[1] + '/generation_config.json'`` —
    leaves only the suffix in the source, which is not a path and must not be
    recorded as one. The access is still real, so the caller keeps the row with
    no path rather than dropping it.
    """
    seen: list[str] = []
    for p in _GEN_CONFIG.findall(text):
        if p.lstrip("/") == "generation_config.json" and p.startswith("/"):
            continue
        if p not in seen:
            seen.append(p)
    return seen


#: The channel that actually writes most generation configs. Agents package a
#: checkpoint for submission with a ``finalize``/``package``/``prep`` script that
#: writes the file as a side effect, so the path never appears on the command
#: line. Missing it left the decisive C1 change of two runs — setting
#: ``temperature: 0.0`` — absent from the table entirely.
#: ``prepare`` is deliberately absent. ``prepare_data.py`` /
#: ``prepare_metamath.py`` / ``prepare_fewshot_data.py`` build training corpora
#: and write only ``*.jsonl``; including the verb scored four such calls per run
#: as configuration writes, and in one run that false positive disguised the
#: fact that the run never touched a generation config at all.
_FINALIZER = re.compile(
    r"\bpython3?\s+(?:-[\w-]+\s+)*(?:[\w./-]*/)?"
    r"(?:finalize|finalise|package|export|publish|patch_model|make_final)[\w-]*\.py\b"
)


def finalizer_call(command: str) -> str | None:
    """The finalizer invocation in this command, if any."""
    m = _FINALIZER.search(command)
    return m.group(0) if m else None


#: The filename inside prose an agent is writing elsewhere. ``echo "… force
#: temperature=0.0 in generation_config.json for +17pt" >> MEMORY.md`` touches no
#: model directory; matching the mention invented a config access and invited the
#: annotator to describe a change that never happened.
_PROSE_WRITE = re.compile(
    r"\b(?:echo|printf)\b(?P<body>[^|;&]*?)>>?\s*(?P<target>\S+)"
)


def mentions_only_in_prose(command: str) -> bool:
    """True when the filename appears only in text being written somewhere else.

    Every occurrence has to sit inside such a body for this to hold: a command
    that both notes the filename and touches a real config is still an access.
    """
    spans = [
        m.span("body") for m in _PROSE_WRITE.finditer(command)
        if "generation_config.json" in m.group("body")
        and "generation_config.json" not in m.group("target")
    ]
    if not spans:
        return False
    return all(
        any(lo <= m.start() < hi for lo, hi in spans)
        for m in re.finditer(r"generation_config\.json", command)
    )


#: Where one command's verbs stop applying. ``mkdir … && mv tokenizer.json … &&
#: cp … && sed -n '/eos/,+4p' exact_v1/generation_config.json`` writes four
#: tokenizer files and *reads* the generation config; classifying on the whole
#: command called it a write, and one run's config writes read as 11 instead of 3.
_SEGMENT = re.compile(r"&&|\|\||[;|]|\n")


def _governing_segment(command: str, path: str | None) -> str:
    """The part of the command whose verbs act on this path.

    A heredoc is never segmented — of any kind, ``cat > f <<TAG`` or
    ``python - <<PY``: its body carries newlines, and splitting on them would
    cut the path away from the verb acting on it two lines below.
    """
    if not path or "<<" in command:
        return command
    for seg in _SEGMENT.split(command):
        if path in seg:
            return seg
    return command


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


#: Setting the decoding config on the *object*, then letting the trainer write
#: it out. ``tok.eos_token = "<|im_end|>"`` → ``model.generation_config
#: .eos_token_id = …`` → ``trainer.save_model()`` never names the file, so a
#: filename-matching extractor records no config access at all. 40 of 234 runs
#: (17%) do this, in 49 events that mention the filename nowhere.
_OBJECT_WRITE = re.compile(
    r"\b(?:model|m|trainer\.model)\.generation_config\s*\.\s*\w+\s*="
    r"|\bgeneration_config\s*\.\s*(?:eos_token_id|temperature|top_p|top_k"
    r"|do_sample|repetition_penalty|max_new_tokens)\s*="
)

#: Constructing one and writing it out names no file either. Matched as two
#: conditions rather than one adjacency, because the argument list carries its
#: own parentheses — ``GenerationConfig(…, pad_token_id=tok.pad_token_id,
#: eos_token_id=[tok.convert_tokens_to_ids("<eos>")]).save_pretrained(out)``.
_CONSTRUCTS = re.compile(r"\bGenerationConfig\s*\(")
_SERIALISES = re.compile(r"\.\s*(?:save_pretrained|to_json_file)\s*\(")


def _flatten(value: Any) -> str:
    """Every argument value, joined — not the JSON rendering of them.

    ``json.dumps`` turns a newline into a literal backslash-n, so the text
    reads ``…save_pretrained(out)nGenerationConfig(…`` and a ``\\b`` anchor
    fails where a real line break would have satisfied it. The pointer verifier
    hit the same thing with escaped quotes; matching the values avoids both.
    """
    if isinstance(value, dict):
        return "\n".join(_flatten(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten(v) for v in value)
    return str(value)


def _refused(events: list[dict[str, Any]]) -> set[str]:
    """``tool_use_id`` of every call the harness declined.

    A ``Write`` answered with ``File has not been read yet`` did not land. Two
    annotators reported this independently, on different runs: counting the
    refused call puts a config state in the record seconds before the write that
    actually happened, and overstates how many writes carry parseable content.
    """
    return {
        e["parent_tool_use"] for e in events
        if e.get("type") == "tool_result" and e.get("is_error") and e.get("parent_tool_use")
    }


def writes_for_run(run_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every access to a ``generation_config.json`` this run made."""
    rows: list[dict[str, Any]] = []
    known = scripts.learn(events)
    refused = _refused(events)
    for e in sorted(events, key=lambda x: (x.get("agent_id") or "", x.get("i") or 0)):
        if e.get("type") != "tool_use" or e.get("tool_use_id") in refused:
            continue
        tool = e.get("tool") or ""
        args = e.get("args") or {}
        blob = _flatten(args)
        names_file = "generation_config.json" in blob
        runs_finalizer = tool in ("Bash", "command_execution") and bool(
            finalizer_call(unwrap(args.get("command") or ""))
        )
        writes_config = tool in ("Bash", "command_execution") and scripts.invoked(
            unwrap(args.get("command") or ""), known, "config_writer"
        )
        edits_object = bool(_OBJECT_WRITE.search(blob)) or bool(
            _CONSTRUCTS.search(blob) and _SERIALISES.search(blob)
        )
        if not names_file and not runs_finalizer and not writes_config and not edits_object:
            continue

        base = {"run_id": run_id, "i": e.get("i"), "ts": e.get("ts"), "tool": tool}

        if edits_object and not names_file:
            rows.append({**base, "path": None, "access": "write",
                         "form": "object_attr", "content_available": False,
                         "content": None,
                         "command": unwrap(args.get("command") or "")[:400]})
            continue

        if writes_config and not names_file:
            rows.append({**base, "path": None, "access": "write",
                         "form": "own_writer", "content_available": False,
                         "content": None,
                         "command": unwrap(args.get("command") or "")[:400]})
            continue

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
        if not command or mentions_only_in_prose(command):
            continue
        if "generation_config.json" not in command:
            if finalizer_call(command):
                rows.append({**base, "path": None, "access": "write",
                             "form": "finalizer", "content_available": False,
                             "content": None, "command": command[:400]})
            continue
        heredoc = _HEREDOC.search(command)
        named = _paths_in(command) or [None]
        for p in named:
            access, form = _classify_shell(_governing_segment(command, p))
            rows.append({
                **base,
                "path": p,
                "access": access,
                "form": form,
                "content_available": bool(heredoc) and p is not None
                and heredoc.group("path") == p,
                "content": heredoc.group("body").strip()
                if heredoc and p is not None and heredoc.group("path") == p
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
