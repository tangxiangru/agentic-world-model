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

import shutil
import subprocess
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
    extra: dict[str, Any] = field(default_factory=dict)


class Backend:
    name = "base"

    def run(self, brief: Brief) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------- heuristic

TRAINING = ("sft", "rft", "dpo", "grpo", "distill")


class HeuristicBackend(Backend):
    """Fixed priors, stated as such. Beat this before believing a skill."""

    name = "heuristic"

    def run(self, brief: Brief) -> None:
        card = load_card(brief.card_path)
        family = get(card, "setup.method.family") or "other"
        elapsed = get(card, "situation.elapsed_h") or 0.0
        n_data = sum(int(d.get("n_examples") or 0) for d in (get(card, "setup.data") or []) if isinstance(d, dict))
        v = schema.empty_verdict(brief.card_id)
        v.update({"backend": self.name, "mode": brief.mode, "issued_at": now()})
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

class CommandBackend(Backend):
    """Run an agent CLI in the session directory; it must write the verdict file."""

    def __init__(self, name: str, argv_template: list[str], model: str | None = None) -> None:
        self.name = name
        self.argv_template = list(argv_template)
        self.model = model

    def argv(self) -> list[str]:
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
        return out

    def run(self, brief: Brief) -> None:
        argv = self.argv()
        if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
            raise BackendError(f"{self.name}: executable not found: {argv[0]}")
        if brief.verdict_path.exists():
            brief.verdict_path.unlink()
        try:
            proc = subprocess.run(argv, input=brief.prompt, text=True, cwd=str(brief.session_dir),
                                  capture_output=True, timeout=max(1.0, brief.budget.wall_min * 60))
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.name}: timed out after {brief.budget.wall_min} min") from exc
        if not brief.verdict_path.is_file():
            tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
            raise BackendError(f"{self.name}: no verdict written to {brief.verdict_path} (exit {proc.returncode}); {tail!r}")
        try:
            v = schema.load_verdict(brief.verdict_path)
        except ValueError as exc:
            raise BackendError(f"{self.name}: invalid verdict JSON: {exc}") from exc
        report = schema.validate_verdict(v)
        if not report.ok:
            raise BackendError(f"{self.name}: invalid verdict:\n{report.render()}")
        if not v.get("backend"):
            v["backend"] = self.name
        schema.dump_verdict(brief.verdict_path, v)


BACKENDS: dict[str, Any] = {
    "heuristic": lambda model: HeuristicBackend(),
    "claude": lambda model: CommandBackend(
        "claude", ["claude", "--print", "--model", "{model}", "--dangerously-skip-permissions"], model),
    "codex": lambda model: CommandBackend(
        "codex", ["codex", "exec", "--skip-git-repo-check", "--yolo", "--model", "{model}"], model),
}


def get_backend(name: str, model: str | None = None) -> Backend:
    if name not in BACKENDS:
        raise BackendError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name](model)
