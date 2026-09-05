"""``awm sandbox setup``: what a PostTrainBench scaffold runs before the prompt."""

from __future__ import annotations

import json

import pytest

from awm import sandbox
from awm.cli import main


def test_setup_installs_the_protocol_and_records_the_sha(tmp_path) -> None:
    task = tmp_path / "task"
    record = sandbox.setup(task, sha="0123abcd", tool="claude", exp_protocol=True)
    assert (task / "skills" / "exp_protocol" / "SKILL.md").is_file()
    assert (task / ".claude" / "skills" / "exp_protocol").is_symlink()
    assert "skills/exp_protocol/SKILL.md" in (task / "CLAUDE.md").read_text()
    assert not (task / "skills" / "exp_protocol_meta").exists()
    on_disk = json.loads((task / "awm_sandbox.json").read_text())
    assert on_disk == record
    assert record["sha"] == "0123abcd" and record["exp_protocol"] is True
    assert "skills/exp_protocol" in record["written"] and "CLAUDE.md" in record["written"]
    assert all(not p.startswith("/") for p in record["written"])


def test_setup_without_the_protocol_writes_only_the_record(tmp_path) -> None:
    task = tmp_path / "task"
    record = sandbox.setup(task, sha="abc")
    assert sorted(p.name for p in task.iterdir()) == ["awm_sandbox.json"]
    assert record["written"] == []


def test_stop_hook_is_merged_into_existing_settings_and_idempotent(tmp_path) -> None:
    task = tmp_path / "task"
    settings = task / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash"]}}))
    sandbox.setup(task, sha="abc", exp_protocol=True, stop_hook=True)
    sandbox.setup(task, sha="abc", exp_protocol=True, stop_hook=True)
    on_disk = json.loads(settings.read_text())
    assert on_disk["permissions"] == {"allow": ["Bash"]}
    stops = on_disk["hooks"]["Stop"]
    assert len(stops) == 1
    command = stops[0]["hooks"][0]["command"]
    assert command.endswith("skills/exp_protocol/hooks/stop_open_cards.py")
    assert (task / "skills" / "exp_protocol" / "hooks" / "stop_open_cards.py").is_file()


def test_stop_hook_needs_the_protocol(tmp_path) -> None:
    with pytest.raises(sandbox.SandboxError, match="exp-protocol"):
        sandbox.setup(tmp_path / "task", sha="abc", stop_hook=True)


def test_setup_refuses_a_bad_tool(tmp_path) -> None:
    with pytest.raises(sandbox.SandboxError, match="tool"):
        sandbox.setup(tmp_path / "task", sha="abc", tool="emacs", exp_protocol=True)


def test_decision_mode_is_frozen_and_survives_repeat_setup(tmp_path):
    from awm.exp_protocol import treatment
    sandbox.setup(tmp_path, sha="abc", exp_protocol=True, decision_mode="multi-self")
    expected = treatment.identity(tmp_path)
    sandbox.setup(tmp_path, sha="abc", exp_protocol=True)
    assert treatment.identity(tmp_path) == expected
    with pytest.raises(sandbox.SandboxError, match="cannot change"):
        sandbox.setup(tmp_path, sha="abc", exp_protocol=True, decision_mode="single")
    assert treatment.identity(tmp_path) == expected


def test_decision_mode_requires_protocol_and_valid_hash(tmp_path):
    from awm.exp_protocol import treatment
    with pytest.raises(sandbox.SandboxError, match="exp-protocol"):
        sandbox.setup(tmp_path, sha="abc", decision_mode="single")
    sandbox.setup(tmp_path, sha="abc", exp_protocol=True, decision_mode="single")
    record = tmp_path / "awm_sandbox.json"
    value = json.loads(record.read_text())
    value["decision_mode"] = "multi-joint"
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash differs"):
        treatment.identity(tmp_path)


def test_cli_runs_setup(tmp_path, capsys) -> None:
    task = tmp_path / "task"
    rc = main(["sandbox", "setup", "--target", str(task), "--sha", "abc",
               "--exp-protocol", "--tool", "codex"])
    assert rc == 0
    assert (task / "AGENTS.md").is_file() and not (task / ".claude").exists()
    out = capsys.readouterr().out
    assert "awm_sandbox.json" in out and "sha=abc" in out


def test_cli_reports_a_refusal(tmp_path, capsys) -> None:
    assert main(["sandbox", "setup", "--target", str(tmp_path / "task"), "--stop-hook"]) == 2
    assert "not set up" in capsys.readouterr().out
