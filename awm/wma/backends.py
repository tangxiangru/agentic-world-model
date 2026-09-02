"""What actually produces a verdict.

A backend receives a ``Brief`` — which card, which session directory, where
the verdict must be written, the mode, the budget, the prompt — and must
leave a valid verdict file behind. Two kinds:

* ``HeuristicBackend`` — deterministic rules. The baseline every model-backed
  skill has to beat, and the test double: no network, no model, no cost.
* ``CommandBackend`` — an agentic CLI (Claude Code, Codex) run as a
  subprocess with the session directory as cwd and the prompt on stdin,
  guided by ``skills/wma/SKILL.md``. The agent writes the file itself; the
  backend only checks that it did and that it validates.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awm.exp_protocol.schema import get, load_card, now

from . import schema


class BackendError(RuntimeError):
    """The backend did not leave a valid verdict behind."""


@dataclass
class Budget:
    cpu_min: float = 10
    gpu_min: float = 0
    wall_min: float = 15
    max_turns: int = 40


@dataclass
class Brief:
    card_id: str
    session_dir: Path
    card_path: Path
    verdict_path: Path
    skill_dir: Path
    mode: str
    budget: Budget
    model: str | None
    prompt: str
    history_dir: Path | None = None
    effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Backend:
    name = "base"
    #: whether the backend reads skills/wma — if so the verdict's wma_skill is the skill's hash, whatever
    #: the agent wrote there; the heuristic backend keeps its own stamp.
    reads_skill = True

    def run(self, brief: Brief) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------- heuristic

TRAINING = ("sft", "rft", "dpo", "grpo", "distill")


class HeuristicBackend(Backend):
    """Fixed priors, stated as such. Beat this before believing a skill."""

    name = "heuristic"
    reads_skill = False

    def run(self, brief: Brief) -> None:
        card = load_card(brief.card_path)
        family = get(card, "setup.method.family") or "other"
        elapsed = get(card, "situation.elapsed_h") or 0.0
        n_data = sum(int(d.get("n_examples") or 0) for d in (get(card, "setup.data") or []) if isinstance(d, dict))
        v = schema.empty_verdict(brief.card_id)
        # The priors do not read the skill; the ledger must not attribute them to a skill version.
        v.update({"backend": self.name, "mode": brief.mode, "issued_at": now(), "wma_skill": "heuristic-priors"})
        v["evidence"] = [{"id": "e1", "path": str(brief.card_path), "locator": "setup.method / situation",
                          "note": f"family={family}, elapsed_h={elapsed}, n_examples={n_data}"}]
        if family in TRAINING:
            l0, l1 = 0.75, 0.6
            interval = [-0.02, 0.03] if n_data < 2000 else [-0.02, 0.06]
            l2_conf = 0.4
        elif family in ("merge", "decode-config"):
            l0, l1 = 0.85, 0.7
            interval, l2_conf = [-0.05, 0.05], 0.3
        else:
            l0, l1 = 0.6, 0.5
            interval, l2_conf = [-0.05, 0.05], 0.25
        v["levels"] = {
            "L0_runs": {"answer": "yes", "confidence": l0, "basis": ["e1"]},
            "L1_valid": {"answer": "yes", "confidence": l1, "basis": ["e1"]},
            "L2_effect": {"metric": "accuracy", "direction": "higher", "interval": interval,
                          "confidence": l2_conf, "basis": ["e1"]},
            "L3_worth_now": {"answer": "yes" if elapsed < 8 else "defer", "confidence": 0.5,
                             "expected_cost_h": get(card, "setup.budget.planned_h") or 1.0, "basis": ["e1"]},
        }
        v["suggestions"] = {"preconditions": [], "cheaper_variants": []}
        schema.dump_verdict(brief.verdict_path, v)


# --------------------------------------------------------------- command

#: Tool inputs that name a file the agent read or touched; Bash is scanned separately.
PATH_INPUTS = ("file_path", "path", "pattern", "notebook_path")
FILE_TOOLS = ("Read", "Glob", "Grep", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead", "LS")
#: Tokens in a shell command that look like paths: absolute, or climbing out of cwd.
SHELL_PATH_RE = re.compile(r"(?:^|[\s=\"'(])((?:/|\.\./)[^\s\"'();|&<>]+)")


def _read_outside(tok: str, roots: list[Path], cwd: Path) -> bool:
    """Whether a path-looking token names something real outside the fence.

    A sed/awk regex literal ('/^result:/,/^conclusion:/p'), a mistyped path or a glob with no match
    cannot have been read, so only what exists counts; a glob is expanded and any match outside counts.
    """
    p = os.path.abspath(os.path.join(cwd, tok))
    if not p.strip("/"):
        return False        # '//' from sed 's/x//' is the root directory (abspath keeps '//'); nothing is read from it
    if any(ch in tok for ch in "*?["):
        return any(not _inside(m, roots, cwd) for m in glob.glob(p))
    return os.path.exists(p) and not _inside(tok, roots, cwd)


def _inside(path: str, roots: list[Path], cwd: Path) -> bool:
    """Lexical containment: no symlink resolution, so ``history/<run>/x`` counts as inside ``history``."""
    p = Path(os.path.abspath(os.path.join(cwd, path)))
    return any(p == r or r in p.parents for r in roots)


def cli_project_dir(session_dir: Path) -> Path:
    """Where Claude Code keeps this cwd's own state (~/.claude/projects/<cwd with / as ->): an oversized
    tool result is spilled to tool-results/ there and the agent reads it back — its own output, not a leak."""
    name = re.sub(r"[^A-Za-z0-9-]", "-", os.path.abspath(session_dir))
    return Path.home() / ".claude" / "projects" / name


def _fence(brief: Brief) -> list[Path]:
    """Where the agent may read: the session, the skill, the history link and what it points at,
    and the CLI's own spill directory for this session."""
    roots = [Path(os.path.abspath(brief.session_dir)), Path(os.path.abspath(brief.skill_dir)),
             cli_project_dir(brief.session_dir)]
    # The agent's own scratch (grep output parked in /tmp and read back) is not a leak — unless the
    # session itself lives under the temp dir (tests, odd layouts), where its truth would be too.
    scratch = Path(os.path.abspath(tempfile.gettempdir()))
    if not Path(os.path.abspath(brief.session_dir)).is_relative_to(scratch):
        roots.append(scratch)
    try:
        roots.append(Path(brief.skill_dir).resolve())
    except OSError:
        pass
    if brief.history_dir:
        h = Path(brief.history_dir)
        roots.append(Path(os.path.abspath(h)))
        try:
            roots.append(h.resolve())
            for entry in h.iterdir():
                roots.append(entry.resolve())
        except OSError:
            pass
    for root in brief.extra.get("allowed_roots") or []:
        roots.append(Path(os.path.abspath(root)))
    return roots


def history_dirs(history: Path) -> list[Path]:
    """The history link's target, then each distinct parent of what its entries point at (the corpus side)."""
    out: list[Path] = []
    try:
        out.append(history.resolve())
        for entry in sorted(history.iterdir()):
            parent = entry.resolve().parent
            if parent not in out:
                out.append(parent)
    except OSError:
        pass
    return out


def transcript_path(verdict_path: Path, output_dir: Path | None = None) -> Path:
    """``exp-NN.transcript[.tag].jsonl`` beside the verdict: what the agent did, turn by turn."""
    verdict_path = Path(verdict_path)
    m = schema.VERDICT_FILE_RE.match(verdict_path.name)
    if not m:
        raise ValueError(f"{verdict_path.name} is not a verdict file name")
    tag = f".{m.group(2)}" if m.group(2) else ""
    filename = f"{m.group(1)}.transcript{tag}.jsonl"
    return Path(output_dir) / filename if output_dir else verdict_path.with_name(filename)


def scan_transcript(stdout: str, brief: Brief) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a Claude Code stream-json transcript: measured cost, and every path the agent touched."""
    cost: dict[str, Any] = {}
    files = 0
    outside: list[str] = []
    roots = _fence(brief)
    cwd = Path(brief.session_dir)
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            if isinstance(ev.get("total_cost_usd"), (int, float)):
                cost["usd"] = round(float(ev["total_cost_usd"]), 4)
            if isinstance(ev.get("num_turns"), int):
                cost["turns"] = ev["num_turns"]
            continue
        content = (ev.get("message") or {}).get("content") if ev.get("type") == "assistant" else None
        for block in content or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name, inp = block.get("name"), block.get("input") or {}
            if name in FILE_TOOLS:
                files += 1
                for key in PATH_INPUTS:
                    val = inp.get(key)
                    if isinstance(val, str) and val and _read_outside(val, roots, cwd):
                        outside.append(val)
            elif name == "Bash":
                cmd = str(inp.get("command") or "")
                if any(_read_outside(tok, roots, cwd) for tok in SHELL_PATH_RE.findall(cmd)):
                    outside.append(f"bash: {cmd}")
    return cost, {"files": files, "outside": outside}


def rescan(dirs: list[Path], *, skill_dir: Path | None = None) -> dict[str, int]:
    """Re-derive ``access`` / ``leak_suspected`` for every verdict from its kept transcript.

    The fence is a harness rule, not the agent's judgment: when the rule changes, the flags are recomputed
    from the transcript instead of buying the verdict again. Nothing else in the verdict is touched.
    """
    from .review import default_skill_dir

    skill_dir = Path(skill_dir) if skill_dir else default_skill_dir()
    counts = {"scanned": 0, "changed": 0}
    for d in dirs:
        for t in sorted(Path(d).rglob("exp-*.transcript*.jsonl")):
            m = re.match(r"^(exp-\d+)\.transcript(?:\.([A-Za-z0-9_-]+))?\.jsonl$", t.name)
            if not m:
                continue
            vp = schema.verdict_path(t.with_name(m.group(1) + ".yaml"), tag=m.group(2))
            if not vp.is_file():
                continue
            session = vp.parents[2]
            history = session / "history"
            brief = Brief(card_id=m.group(1), session_dir=session, card_path=t.with_name(m.group(1) + ".yaml"),
                          verdict_path=vp, skill_dir=skill_dir, mode="offline", budget=Budget(), model=None,
                          prompt="", history_dir=history if history.exists() else None)
            _, access = scan_transcript(t.read_text(), brief)
            v = schema.load_verdict(vp)
            counts["scanned"] += 1
            before = (v.get("access"), bool(v.get("leak_suspected")))
            v["access"] = access
            if access["outside"]:
                v["leak_suspected"] = True
            else:
                v.pop("leak_suspected", None)
            if before != (access, bool(access["outside"])):
                counts["changed"] += 1
                schema.dump_verdict(vp, v)
    return counts


class CommandBackend(Backend):
    """Run an agent CLI in the session directory; it must write the verdict file."""

    def __init__(self, name: str, argv_template: list[str], model: str | None = None, *,
                 effort: str | None = None, effort_flag: list[str] | None = None,
                 transcript: str | None = None, history_flag: str | None = None,
                 max_turns_flag: str | None = None) -> None:
        self.name = name
        self.argv_template = list(argv_template)
        self.model = model
        # Effort is part of the measurement: passed on the command line, never inherited from the
        # user's CLI settings, and stamped on the verdict with the model so the ledger can group by it.
        self.effort = effort
        self.effort_flag = list(effort_flag or [])   # e.g. ["--effort", "{effort}"]
        self.transcript = transcript          # "stream-json" → stdout is parsed for cost and file access
        self.history_flag = history_flag      # e.g. --add-dir: lets the agent read the history link's target
        self.max_turns_flag = max_turns_flag  # e.g. --max-turns: a hard stop on top of the wall budget

    def argv(self, brief: Brief | None = None) -> list[str]:
        out = []
        for a in self.argv_template:
            if "{model}" in a:
                if not self.model:
                    continue
                a = a.replace("{model}", self.model)
            out.append(a)
        if not self.model:
            # drop a dangling "--model" flag whose value was skipped
            out = [a for i, a in enumerate(out) if not (a == "--model" and (i + 1 >= len(out) or out[i + 1].startswith("-")))]
        if self.effort and self.effort_flag:
            out += [a.replace("{effort}", self.effort) for a in self.effort_flag]
        if brief is not None:
            if self.history_flag:
                out += [self.history_flag, str(Path(brief.skill_dir).resolve())]
            if self.history_flag and brief.history_dir:
                for d in history_dirs(Path(brief.history_dir)):
                    out += [self.history_flag, str(d)]
            if self.max_turns_flag and brief.budget.max_turns:
                out += [self.max_turns_flag, str(int(brief.budget.max_turns))]
        return out

    def run(self, brief: Brief) -> None:
        argv = self.argv(brief)
        if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
            raise BackendError(f"{self.name}: executable not found: {argv[0]}")
        if brief.verdict_path.exists():
            brief.verdict_path.unlink()
        started = time.monotonic()
        try:
            proc = subprocess.run(argv, input=brief.prompt, text=True, cwd=str(brief.session_dir),
                                  capture_output=True, timeout=max(1.0, brief.budget.wall_min * 60),
                                  check=False)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.name}: timed out after {brief.budget.wall_min} min") from exc
        wall_min = round((time.monotonic() - started) / 60, 6)
        if not brief.verdict_path.is_file():
            tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
            raise BackendError(f"{self.name}: no verdict written to {brief.verdict_path} (exit {proc.returncode}); {tail!r}")
        # What the run measurably was — kept even when the file the agent wrote is unusable.
        measured: dict[str, Any] = {"backend": self.name, "wall_min": wall_min}
        if self.model:
            measured["model"] = self.model
        if self.effort:
            measured["effort"] = self.effort
        if self.transcript == "stream-json":
            # Kept whole: the iteration agent reads it by hand, and a fence fix can rescan it.
            transcript_path(
                brief.verdict_path, brief.extra.get("transcript_dir")
            ).write_text(proc.stdout or "")
            cost, access = scan_transcript(proc.stdout or "", brief)
            measured.update({"cost": cost, "access": access, "leak_suspected": bool(access["outside"])})
        try:
            v = schema.load_verdict(brief.verdict_path)
        except ValueError as exc:
            schema.reject_verdict(brief.verdict_path, f"invalid verdict JSON: {exc}", **measured)
            raise BackendError(f"{self.name}: invalid verdict JSON: {exc}") from exc
        schema.normalize_verdict(v)
        report = schema.validate_verdict(v)
        if not report.ok:
            schema.reject_verdict(brief.verdict_path, report.render(), **measured)
            raise BackendError(f"{self.name}: invalid verdict:\n{report.render()}")
        v["backend"] = self.name
        if self.model:
            v["model"] = self.model
        if self.effort:
            v["effort"] = self.effort
        if self.transcript == "stream-json":
            v.setdefault("cost", {})
            v["cost"].update(measured["cost"])
            v["access"] = measured["access"]
            if measured["leak_suspected"]:
                v["leak_suspected"] = True
        v.setdefault("cost", {})
        v["cost"]["wall_min"] = wall_min          # measured beats self-reported
        schema.dump_verdict(brief.verdict_path, v)


BACKENDS: dict[str, Any] = {
    "heuristic": lambda model, effort: HeuristicBackend(),
    # stream-json (which needs --verbose in print mode) gives the measured cost and every tool call,
    # so the ledger's cost is real and a read outside the fence is caught, not trusted away.
    "claude": lambda model, effort: CommandBackend(
        "claude", ["claude", "--print", "--verbose", "--output-format", "stream-json",
                   "--model", "{model}", "--setting-sources", "", "--no-session-persistence",
                   "--dangerously-skip-permissions"], model,
        effort=effort, effort_flag=["--effort", "{effort}"],
        transcript="stream-json", history_flag="--add-dir", max_turns_flag="--max-turns"),
    "codex": lambda model, effort: CommandBackend(
        "codex", ["codex", "exec", "--skip-git-repo-check", "--yolo", "--model", "{model}"], model,
        effort=effort, effort_flag=["-c", "model_reasoning_effort={effort}"]),
}


def get_backend(name: str, model: str | None = None, effort: str | None = None) -> Backend:
    if name not in BACKENDS:
        raise BackendError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name](model, effort)
