#!/usr/bin/env python3
"""Inject a ready completion-monitor event into a resumed Codex session."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / "data/ptb/monitor/exp_protocol_goal.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if state.get("status") != "ready":
        return 0
    terminal = ",".join(str(item) for item in state.get("terminal_jobs") or [])
    context = (
        "EXP_PROTOCOL_COMPLETION_EVENT: the background monitor reached "
        f"{state.get('terminal_count')}/{state.get('threshold')} terminal jobs at "
        f"{state.get('checked_at')}; jobs={terminal}. First read "
        "skills/exp_protocol_meta/SKILL.md, then harvest with the operator, require validator-clean "
        "receipt-backed bundles, and only when the new-clean window reaches eight start the local "
        "Claude Code Opus 5[1m] max trace review. Do not wait for a GitHub/Fable response."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
