"""Locating config accesses, and being honest about which ones carry content."""

from __future__ import annotations

import pytest

from awm.traj import config_writes

GC = "work/sft1/generation_config.json"


def use(i: int, tool: str, args: dict) -> dict:
    return {
        "run_id": "r", "agent_id": "main", "i": i, "type": "tool_use", "role": "assistant",
        "ts": "2026-01-01T00:00:00Z", "tool": tool, "args": args, "tool_use_id": f"t{i}",
    }


def rows(events: list[dict]) -> list[dict]:
    return config_writes.writes_for_run("r", events)


class TestContentIsCapturedWhenPresent:
    def test_write_tool_carries_the_whole_file(self) -> None:
        body = '{\n  "do_sample": false,\n  "eos_token_id": [1, 106]\n}'
        r = rows([use(0, "Write", {"file_path": f"/home/ben/task/{GC}", "content": body})])
        assert len(r) == 1
        assert r[0]["access"] == "write"
        assert r[0]["content_available"] is True
        assert '"do_sample": false' in r[0]["content"]

    def test_shell_heredoc_carries_the_whole_file(self) -> None:
        cmd = (
            f"mkdir -p work/sft1 && cat > {GC} <<'EOF'\n"
            '{\n  "bos_token_id": 151643,\n  "temperature": 0.0\n}\n'
            "EOF\nls -la work/sft1"
        )
        r = rows([use(0, "Bash", {"command": cmd})])
        assert r[0]["form"] == "heredoc"
        assert r[0]["content_available"] is True
        assert '"temperature": 0.0' in r[0]["content"]


class TestContentIsAbsentWhenTheTraceDoesNotCarryIt:
    def test_codex_file_change_has_a_path_and_nothing_else(self) -> None:
        r = rows([use(0, "file_change", {"changes": [
            {"path": f"/home/ben/task/{GC}", "kind": "update"},
            {"path": "/home/ben/task/other/config.json", "kind": "update"},
        ]})])
        assert len(r) == 1, "only the generation_config path is ours"
        assert r[0]["access"] == "write"
        assert r[0]["content_available"] is False
        assert r[0]["content"] is None

    def test_python_heredoc_writing_it_is_located_but_not_parsed(self) -> None:
        cmd = (
            "/bin/bash -lc \"python - <<'PY'\n"
            "import json\n"
            f"p='{GC}'\n"
            "d=json.load(open(p)); d['temperature']=0.0; json.dump(d, open(p,'w'))\n"
            'PY"'
        )
        r = rows([use(0, "command_execution", {"command": cmd})])
        assert r[0]["access"] == "write"
        assert r[0]["form"] == "python_code"
        assert r[0]["content_available"] is False


class TestReadsAreNotWrites:
    @pytest.mark.parametrize(
        "cmd",
        [
            f"cat {GC}",
            f"sed -n '1,120p' {GC}",
            f"python3 -c \"import json; print(json.load(open('{GC}')).get('do_sample'))\"",
            f"grep -n temperature {GC}",
        ],
        ids=["cat", "sed", "python-load", "grep"],
    )
    def test_reading_is_classified_as_read(self, cmd: str) -> None:
        assert rows([use(0, "Bash", {"command": cmd})])[0]["access"] == "read"

    def test_read_tool(self) -> None:
        r = rows([use(0, "Read", {"file_path": f"/home/ben/task/{GC}"})])
        assert r[0]["access"] == "read"
        assert r[0]["content_available"] is False


class TestIrrelevantEventsAreSkipped:
    def test_an_edit_to_an_unrelated_file_is_ignored(self) -> None:
        assert rows([use(0, "Edit", {
            "file_path": "/home/ben/.claude/memory/notes.md",
            "old_string": "a", "new_string": "b",
        })]) == []

    def test_a_todo_mentioning_the_file_is_ignored(self) -> None:
        assert rows([use(0, "TodoWrite", {"todos": [
            {"content": "write generation_config.json", "status": "pending"},
        ]})]) == []


class TestFrame:
    def test_empty_frame_keeps_the_contract(self) -> None:
        df = config_writes.empty()
        assert list(df.columns) == list(config_writes.COLUMNS)
        assert df.empty
