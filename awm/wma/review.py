"""Ask a backend for a verdict on one card.

``review`` is the pull entry point: load the card, refuse a post-hoc verdict
(a card that already has a result), build the brief, run the backend, check
the file it left, stamp what it left blank. It never modifies the card. The
For local/offline use the session gets a ``skills/wma`` link. An online
sidecar instead reads a private absolute skill path and never exposes that
link to the scientist.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from awm import paths
from awm.exp_protocol.schema import get, load_card, migrate_v1, now

from . import schema
from .backends import Backend, BackendError, Brief, Budget


class ReviewError(ValueError):
    pass


def default_skill_dir() -> Path:
    env = os.environ.get("AWM_WMA_SKILL_DIR")
    return Path(env) if env else paths.REPO_ROOT / "skills" / "wma"


_PREPARE_LOCK = threading.Lock()


def prepare_session(session_dir: Path, skill_dir: Path) -> Path:
    """``<session>/skills/wma`` → the skill, so the agent can read SKILL.md relative to cwd.

    Safe to call concurrently: parallel reviews of several cards share one session.
    """
    link = Path(session_dir) / "skills" / "wma"
    target = Path(skill_dir).resolve()
    with _PREPARE_LOCK:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if Path(os.readlink(link)).resolve() == target:
                return link
            link.unlink()
        elif link.exists():
            raise ReviewError(f"{link} exists and is not a symlink; refusing to replace it")
        try:
            os.symlink(target, link)
        except FileExistsError:
            # another process got there first; accept it if it points where we want
            if not (link.is_symlink() and Path(os.readlink(link)).resolve() == target):
                raise
    return link


def build_prompt(brief: Brief) -> str:
    mode_rule = (
        "Mode: offline replay. Only static probes are allowed (static_check, data_probe): read code, data and "
        "configs; do not run training, evaluation, or anything that needs a GPU."
        if brief.mode == "offline" else
        "Mode: online. Static and dynamic probes are allowed within the budget; run them in a scratch copy, "
        "never in the scientist's working files."
    )
    history = (f"Historical experience (other runs, read-only): {brief.history_dir}\n"
               if brief.history_dir else "")
    skill = Path(brief.skill_dir) / "SKILL.md"
    b = brief.budget
    return f"""You are the world-model agent (WMA). Read {skill} first and follow it.

Task: produce a verdict for card {brief.card_id}.
Card (sections 0-4 are the proposal): {brief.card_path}
Session directory (the scientist's context, read-only): {brief.session_dir}
{history}{mode_rule}
Budget: cpu_min={b.cpu_min}, gpu_min={b.gpu_min}, wall_min={b.wall_min}. Stop and write what you have before it runs out.

Write the verdict as JSON to exactly this path and nothing else:
{brief.verdict_path}
Its shape is skills/wma/verdict.example.json (schema awm-wma-verdict-v1). Set mode to "{brief.mode}".
Do not modify any other file. Do not read outside the session directory except the WMA skill and historical experience paths above.
Do not look for the card's result anywhere; you are estimating it.
"""


def make_brief(session_dir: Path, card_id: str, *, mode: str, budget: Budget, model: str | None,
               skill_dir: Path, history_dir: Path | None = None, tag: str | None = None,
               effort: str | None = None) -> Brief:
    session_dir = Path(session_dir)
    card_path = session_dir / "memory" / "cards" / f"{card_id}.yaml"
    brief = Brief(card_id=card_id, session_dir=session_dir, card_path=card_path,
                  verdict_path=schema.verdict_path(card_path, tag=tag), skill_dir=Path(skill_dir), mode=mode,
                  budget=budget, model=model, prompt="", history_dir=history_dir, effort=effort)
    brief.prompt = build_prompt(brief)
    return brief


def review(session_dir: Path, card_id: str, backend: Backend, *, mode: str = "offline",
           budget: Budget | None = None, model: str | None = None, skill_dir: Path | None = None,
           history_dir: Path | None = None, force: bool = False, tag: str | None = None,
           effort: str | None = None, expose_skill: bool = True,
           transcript_dir: Path | None = None,
           allowed_roots: list[Path] | None = None) -> dict[str, Any]:
    if mode not in schema.MODES:
        raise ReviewError(f"mode must be one of {schema.MODES}")
    skill_dir = Path(skill_dir) if skill_dir else default_skill_dir()
    try:
        brief = make_brief(session_dir, card_id, mode=mode, budget=budget or Budget(), model=model,
                           skill_dir=skill_dir, history_dir=history_dir, tag=tag, effort=effort)
    except ValueError as exc:
        raise ReviewError(str(exc)) from exc
    if not brief.card_path.is_file():
        raise ReviewError(f"no such card: {brief.card_path}")
    if transcript_dir is not None:
        transcript_dir = Path(transcript_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        brief.extra["transcript_dir"] = transcript_dir
    if allowed_roots:
        brief.extra["allowed_roots"] = [Path(root) for root in allowed_roots]
    card = migrate_v1(load_card(brief.card_path))
    # ``not_run`` is the schema's legal pre-launch sentinel. Some scientists
    # fill it while writing sections 0-4, before the WMA request; treating any
    # non-empty execution string as an outcome rejected those cards as post-hoc.
    execution = get(card, "result.execution")
    if not force and (
        execution not in (None, "", "not_run") or get(card, "conclusion.decision")
    ):
        raise ReviewError(f"{card_id} already has a result; a verdict now would be post-hoc (use force to override)")
    if expose_skill:
        prepare_session(brief.session_dir, skill_dir)
    try:
        backend.run(brief)
    except BackendError as exc:
        if brief.verdict_path.exists():        # a backend that raised but left a file: not a verdict
            schema.reject_verdict(brief.verdict_path, str(exc), backend=backend.name)
        raise ReviewError(str(exc)) from exc
    v = schema.load_verdict(brief.verdict_path)
    report = schema.validate_verdict(v)
    if not report.ok:
        schema.reject_verdict(brief.verdict_path, report.render(), backend=backend.name)
        raise ReviewError("backend left an invalid verdict:\n" + report.render())
    # The harness knows these; whatever the agent wrote there (the first real verdict copied the example
    # file's placeholders) is replaced. The heuristic backend's own skill stamp is the one exception.
    if backend.reads_skill:
        v["wma_skill"] = schema.skill_sha(skill_dir)
    v["backend"] = backend.name
    v["mode"] = mode
    if model:
        v["model"] = model
    if effort:
        v["effort"] = effort
    v["issued_at"] = now()
    v["card_id"] = card_id
    schema.dump_verdict(brief.verdict_path, v)
    return v
