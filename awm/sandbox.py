"""``awm sandbox setup``: the one step a PostTrainBench scaffold runs before the prompt.

The scaffold (``agents/claude_vertex_max_awm/solve.sh`` in the PostTrainBench
fork) mounts a read-only checkout of this repository at ``/home/ben/awm``,
puts it on ``PYTHONPATH``, and runs this command in the task directory with
the arguments the launcher forwarded in ``AWM_SANDBOX_SETUP``. What gets
installed is decided here, by the mounted commit; the scaffold never changes
between studies. Everything written lands inside ``--target`` and is listed
in ``<target>/awm_sandbox.json`` with the checkout's commit, so a harvested
result says which protocol it ran under.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from awm.exp_protocol import install as install_mod

RECORD = "awm_sandbox.json"
STOP_HOOK = "skills/exp_protocol/hooks/stop_open_cards.py"


class SandboxError(ValueError):
    pass


def _install_stop_hook(target: Path) -> Path:
    """Register the protocol's Stop hook in ``<target>/.claude/settings.json``; keep what is there."""
    settings_path = target / ".claude" / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError as exc:
            raise SandboxError(f"{settings_path} is not valid JSON: {exc}") from exc
        if not isinstance(settings, dict):
            raise SandboxError(f"{settings_path} must hold a JSON object")
    command = f"python3 {target / STOP_HOOK}"
    hooks = settings.setdefault("hooks", {})
    stops = hooks.setdefault("Stop", [])
    already = any(
        any(h.get("command") == command for h in entry.get("hooks", []))
        for entry in stops
        if isinstance(entry, dict)
    )
    if not already:
        stops.append({"hooks": [{"type": "command", "command": command}]})
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return settings_path


def setup(
    target: Path,
    *,
    sha: str,
    tool: str = "claude",
    exp_protocol: bool = False,
    stop_hook: bool = False,
) -> dict[str, Any]:
    target = Path(target).resolve()
    if stop_hook and not exp_protocol:
        raise SandboxError("--stop-hook needs --exp-protocol: the hook is part of the protocol")
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if exp_protocol:
        try:
            written += install_mod.install(target, tool)
        except install_mod.InstallError as exc:
            raise SandboxError(str(exc)) from exc
        if stop_hook:
            written.append(_install_stop_hook(target))
    record = {
        "schema_version": 1,
        "sha": sha,
        "tool": tool,
        "exp_protocol": exp_protocol,
        "stop_hook": stop_hook,
        "written": sorted(str(p.relative_to(target)) for p in written),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (target / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    return record


def _setup(args: argparse.Namespace) -> int:
    try:
        record = setup(
            Path(args.target),
            sha=args.sha,
            tool=args.tool,
            exp_protocol=args.exp_protocol,
            stop_hook=args.stop_hook,
        )
    except SandboxError as exc:
        print(f"not set up: {exc}")
        return 2
    for rel in record["written"]:
        print(f"wrote {rel}")
    print(f"wrote {RECORD} (sha={record['sha']})")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    sb = sub.add_parser("sandbox", help="what a PostTrainBench scaffold runs before the prompt")
    cmds = sb.add_subparsers(dest="cmd", required=True)
    s = cmds.add_parser("setup", help="install the study's pieces into the task directory")
    s.add_argument("--target", default=".", help="the task directory (default: cwd)")
    s.add_argument("--sha", default="unknown", help="commit of the mounted checkout, for provenance")
    s.add_argument("--tool", choices=("claude", "codex", "both"), default="claude")
    s.add_argument("--exp-protocol", action="store_true", help="install skills/exp_protocol")
    s.add_argument("--stop-hook", action="store_true",
                   help="also register the protocol's Claude Code Stop hook (needs --exp-protocol)")
    s.set_defaults(func=_setup)
