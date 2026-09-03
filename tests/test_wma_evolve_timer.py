import sqlite3
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from awm import wma_evolve_timer as timer

THREAD = "01a06887-7c0a-7fa3-814a-8c734c0c6f8d"


def config(tmp_path):
    database = tmp_path / "queue.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE queued_items (thread_id TEXT)")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Inspect owned WMA experiments.")
    return {
        "enabled": True, "thread_id": THREAD, "repo": str(tmp_path),
        "queue_database": str(database), "prompt_file": str(prompt),
        "codex_binary": "/codex", "remote": "unix:///codex.sock",
    }


def test_existing_input_suppresses_duplicate_monitor(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    with sqlite3.connect(cfg["queue_database"]) as db:
        db.execute("INSERT INTO queued_items VALUES (?)", (THREAD,))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "git"
        return CompletedProcess(command, 0, stdout=timer.BRANCH + "\n")

    monkeypatch.setattr(timer.subprocess, "run", run)
    assert timer.dispatch(cfg)["reason"] == "task_already_has_queued_input"
    assert len(calls) == 1


def test_dispatch_targets_existing_task_without_permission_overrides(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    with sqlite3.connect(cfg["queue_database"]) as db:
        db.execute("INSERT INTO queued_items VALUES ('another-task')")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = timer.BRANCH if command[0] == "git" else "queued"
        return CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(timer.subprocess, "run", run)
    assert timer.dispatch(cfg)["state"] == "queued"
    assert calls[-1] == [
        "/codex", "queue", "--remote", "unix:///codex.sock", "--thread", THREAD,
        "--message", "Inspect owned WMA experiments.",
    ]


def test_branch_change_blocks_dispatch(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    monkeypatch.setattr(
        timer.subprocess, "run",
        lambda command, **kwargs: CompletedProcess(command, 0, stdout="another-branch\n"),
    )
    assert timer.dispatch(cfg)["reason"] == "branch_mismatch"


def test_missing_queue_database_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        timer.queued_count(missing, THREAD)
    assert not Path(missing).exists()
