"""Queue a periodic continuation of the existing WMA operator task.

Cron calls this small dispatcher. The Codex CLI queues work on the existing
task; the dispatcher never launches experiments or changes model permissions.
The local Codex queue database is read-only and used only for deduplication.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

DEFAULT_CONFIG = Path(
    "/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/timer/config.json"
)
BRANCH = "gangda_wma_evolve"


def queued_count(database: Path, thread_id: str) -> int:
    # Missing databases or changed schemas fail closed; never create an app DB.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            "SELECT COUNT(*) FROM queued_items WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return int(row[0])


def dispatch(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not config.get("enabled", False):
        return {"state": "disabled"}
    thread_id = str(UUID(config["thread_id"]))
    repo = Path(config["repo"])
    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    if branch != BRANCH:
        return {"state": "blocked", "reason": "branch_mismatch", "branch": branch}
    pending = queued_count(Path(config["queue_database"]), thread_id)
    if pending:
        return {"state": "skipped", "reason": "task_already_has_queued_input", "count": pending}
    message = Path(config["prompt_file"]).read_text().strip()
    if not message:
        raise ValueError("empty operator prompt")
    command = [
        config["codex_binary"], "queue", "--remote", config["remote"],
        "--thread", thread_id, "--message", message,
    ]
    if dry_run:
        return {"state": "dry_run", "thread_id": thread_id, "prompt_chars": len(message)}
    result = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, timeout=60, check=True,
    )
    return {"state": "queued", "thread_id": thread_id, "receipt": result.stdout.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state_dir = args.config.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "dispatch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "skipped", "reason": "dispatcher_running"}))
            return 0
        try:
            outcome = dispatch(json.loads(args.config.read_text()), dry_run=args.dry_run)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as exc:
            outcome = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
        outcome["checked_at"] = datetime.now(timezone.utc).isoformat()
        rendered = json.dumps(outcome, indent=2, sort_keys=True) + "\n"
        if not args.dry_run:
            temporary = state_dir / f".last_tick.{os.getpid()}.tmp"
            temporary.write_text(rendered)
            temporary.replace(state_dir / "last_tick.json")
        print(rendered, end="")
        return 1 if outcome["state"] in {"error", "blocked"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
