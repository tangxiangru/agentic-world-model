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


class TestRejectedWrite:
    def test_a_write_the_harness_refused_did_not_happen(self) -> None:
        """``File has not been read yet`` — the tool call was declined.

        Two annotators reported this independently on different runs. Counting
        it puts a config state in the record five seconds before the write that
        actually landed, and inflates the count of parseable writes.
        """
        events = [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Write", "tool_use_id": "t1",
             "args": {"file_path": "sft_v1/generation_config.json",
                      "content": '{"temperature": 0.0}'}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "is_error": True,
             "text": "<tool_use_error>File has not been read yet.</tool_use_error>"},
        ]
        assert config_writes.writes_for_run("r", events) == []

    def test_a_write_that_landed_is_kept(self) -> None:
        events = [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Write", "tool_use_id": "t1",
             "args": {"file_path": "sft_v1/generation_config.json",
                      "content": '{"temperature": 0.0}'}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "text": "File created successfully."},
        ]
        assert len(config_writes.writes_for_run("r", events)) == 1


class TestProseMention:
    def test_a_filename_inside_a_note_is_not_an_access(self) -> None:
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash", "args": {
            "command": 'echo "- force temperature=0.0 in generation_config.json for +17pt"'
                       " >> ~/.claude/memory/MEMORY.md"}}]
        assert config_writes.writes_for_run("r", events) == []

    def test_a_runtime_built_path_is_an_access_with_no_path(self) -> None:
        """``sys.argv[1] + '/generation_config.json'`` leaves only the suffix.

        That suffix is not a path and must not be recorded as one, but the
        access is real: recording nothing loses a config write.
        """
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash", "args": {
            "command": "python -c \"import json,sys\np=sys.argv[1]+'/generation_config.json'\n"
                       "d=json.load(open(p)); d['temperature']=0.0; json.dump(d,open(p,'w'))\" out"}}]
        rows = config_writes.writes_for_run("r", events)
        assert len(rows) == 1
        assert rows[0]["path"] is None, "a degenerate suffix must not pose as a path"


class TestProseInSource:
    def test_a_docstring_naming_the_scorer_does_not_make_a_scorer(self) -> None:
        # A data builder whose docstring says it writes evaluate.py's format was
        # classified an evaluator on that line alone, and its launch then
        # collected a neighbouring evaluation's score.
        from awm.traj import scripts
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Write", "args": {
            "file_path": "build_data.py",
            "content": '"""Build SFT data in the exact format used by evaluate.py."""\n'
                       "import json\nrows = [json.loads(x) for x in open('gsm8k.jsonl')]\n"}}]
        assert "evaluator" not in scripts.learn(events).get("build_data.py", set())

    def test_a_real_call_still_classifies(self) -> None:
        from awm.traj import scripts
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Write", "args": {
            "file_path": "ev.sh", "content": "python evaluate.py --model-path $1 --limit 150\n"}}]
        assert "evaluator" in scripts.learn(events)["ev.sh"]


class TestVerbGovernsThePath:
    def test_a_read_beside_writes_of_other_files_is_a_read(self) -> None:
        """``mkdir && mv tokenizer.json … && cp … && sed -n '…' gc.json``.

        The mv and cp act on the tokenizer four; the only verb touching the
        generation config is sed. Classifying on the whole command called it a
        write, and one run's config writes read as 11 instead of 3.
        """
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash", "args": {
            "command": "mkdir -p v1/backup && mv v1/tokenizer.json v1/tokenizer_config.json"
                       " v1/backup/ && cp ~/hf/tokenizer.json v1/"
                       " && sed -n '/eos_token_id/,+4p' v1/generation_config.json"}}]
        rows = config_writes.writes_for_run("r", events)
        assert [(r["access"], r["form"]) for r in rows] == [("read", "shell_read")]

    def test_a_copy_of_the_config_itself_is_still_a_write(self) -> None:
        events = [{"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash", "args": {
            "command": "mkdir step20 && cp v1/checkpoint-20/generation_config.json step20/"}}]
        assert config_writes.writes_for_run("r", events)[0]["access"] == "write"

