import json
from pathlib import Path

import pytest

from tools.outcome_prediction.prefix_code_recovery import (
    _python_transform,
    _sed_replace,
    recover_code_events,
)
from tools.outcome_prediction.rpm_code_provenance import TraceIndex


def trace(tmp_path, calls):
    lines = []
    for n, (tool, inputs, output, structured) in enumerate(calls):
        for k in range(2):
            stamp = f"2026-09-01T00:00:{n * 2 + k:02d}Z"
            block = (
                {"type": "tool_use", "id": str(n), "name": tool, "input": inputs}
                if k == 0
                else {"type": "tool_result", "tool_use_id": str(n), "content": output}
            )
            event = {"timestamp": stamp, "message": {"content": [block]}}
            if k and structured is not None:
                event["tool_use_result"] = structured
            lines.append(f"[{stamp}] " + json.dumps(event))
    path = tmp_path / "trace.txt"
    path.write_text("\n".join(lines))
    return TraceIndex(path)


def test_complete_read_recovered_but_partial_read_not(tmp_path):
    full = {
        "file": {
            "filePath": "/home/ben/task/a.py",
            "content": "x=1\n",
            "numLines": 1,
            "startLine": 1,
            "totalLines": 1,
        }
    }
    partial = {"file": {**full["file"], "totalLines": 2}}
    index = trace(
        tmp_path,
        [
            ("Read", {"file_path": "/home/ben/task/a.py"}, "", full),
            ("Read", {"file_path": "/home/ben/task/a.py"}, "", partial),
        ],
    )
    result = recover_code_events(index)
    assert [v["content"] for v in result["versions"]] == ["x=1\n"]
    assert len(result["unresolved"]) == 1


def test_text_write_and_edit_with_exact_path_confirmation(tmp_path):
    index = trace(
        tmp_path,
        [
            (
                "Write",
                {"file_path": "/home/ben/task/a.py", "content": "x=1\n"},
                "File created successfully at: /home/ben/task/a.py (file state is current in your context — no need to Read it back)",
                None,
            ),
            (
                "Edit",
                {"file_path": "/home/ben/task/a.py", "old_string": "1", "new_string": "2"},
                "The file /home/ben/task/a.py has been updated successfully.",
                None,
            ),
            (
                "Write",
                {"file_path": "/home/ben/task/b.py", "content": "wrong"},
                "File created successfully at: /home/ben/task/b.py.other",
                None,
            ),
        ],
    )
    result = recover_code_events(index)
    assert [v["content"] for v in result["versions"]] == ["x=1\n", "x=2\n"]


def test_intervening_unknown_mutation_blocks_text_edit(tmp_path):
    index = trace(
        tmp_path,
        [
            (
                "Write",
                {"file_path": "/home/ben/task/a.py", "content": "x=1\n"},
                "File created successfully at: /home/ben/task/a.py",
                None,
            ),
            ("Bash", {"command": "echo unknown > a.py"}, "", {}),
            (
                "Edit",
                {"file_path": "/home/ben/task/a.py", "old_string": "1", "new_string": "2"},
                "The file /home/ben/task/a.py has been updated successfully.",
                None,
            ),
        ],
    )
    result = recover_code_events(index)
    assert len(result["versions"]) == 1
    assert result["unresolved"]


def test_literal_python_transform_without_execution():
    source = (
        "s=open('a.py').read()\ns=s.replace('one','two')\nopen('b.py','w').write(s)\nprint('done')"
    )
    assert _python_transform(source, lambda p: "one\n", "/w") == {"/w/b.py": "two\n"}
    with pytest.raises(ValueError):
        _python_transform("import os\nos.system('bad')", lambda p: "one", "/w")
    with pytest.raises(ValueError):
        _python_transform(
            "s=open('a.py').read()\nprint(open('b.py','w').write(s))", lambda p: "one", "/w"
        )


def test_restricted_sed_literal_and_bre_escaping():
    assert _sed_replace("TARGET=3\nTARGET=3\n", "s/TARGET=3/TARGET=6/") == "TARGET=6\nTARGET=6\n"
    assert _sed_replace("a/a\n", "s|a/a|b/b|") == "b/b\n"
    for bad in ["s/a/b/e", "s/a/b/;e malicious", "s/[a]/b/", r"s/\(a\)/\1/"]:
        with pytest.raises(ValueError):
            _sed_replace("a", bad)


def test_cp_and_literal_sed(tmp_path):
    index = trace(
        tmp_path,
        [
            (
                "Write",
                {"file_path": "/home/ben/task/a.py", "content": "x=1\n"},
                "File created successfully at: /home/ben/task/a.py",
                None,
            ),
            ("Bash", {"command": "cp a.py b.py\nsed -i 's/x=1/x=2/' b.py"}, "", {}),
        ],
    )
    result = recover_code_events(index)
    assert [(v["path"], v["content"]) for v in result["versions"]][-1] == (
        "/home/ben/task/b.py",
        "x=2\n",
    )


@pytest.mark.parametrize(
    "command",
    [
        "false && cp a.py b.py",
        "cp a.py b.py &",
        "cp a.py b.py | cat",
        "cd /other; cp a.py b.py",
        "cp a.py b.py; rm b.py",
        "cp a.py b.py; echo unknown > b.py",
    ],
)
def test_unproven_shell_copy_or_later_mutation_not_recovered(tmp_path, command):
    index = trace(
        tmp_path,
        [
            (
                "Write",
                {"file_path": "/home/ben/task/a.py", "content": "x=1\n"},
                "File created successfully at: /home/ben/task/a.py",
                None,
            ),
            ("Bash", {"command": command}, "", {}),
        ],
    )
    result = recover_code_events(index)
    assert not any(v["path"].endswith("/b.py") for v in result["versions"])


@pytest.mark.parametrize(
    "session,path,sha",
    [
        (
            "aime-r0-21",
            "evaluate.py",
            "6bdb89e5c0494392fd07ce49b8b2262824eef9152dc52c4011f9348225ff5a08",
        ),
        (
            "aime-r0-22",
            "add_wrapup2.py",
            "9dfed4e9ad7ff26453588f22edcc6e1f73adb93af2f27aaaf3567ddf893c9890",
        ),
        (
            "aime-r0-32",
            "prepare_omr2.py",
            "8203ae1ace043106dcaa7b65765d1e36f742993e1954fe345a3691023de6229e",
        ),
        (
            "aime2-r0-14",
            "data/prep_v5.py",
            "55040a04afbeb623b8d138a523c731e727f8ff803c6ad3ad7328f23d4c29899f",
        ),
    ],
)
def test_real_pinned_recoveries(session, path, sha):
    mirror = (
        Path(__file__).resolve().parents[1]
        / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7/cells"
    )
    source = mirror / session / "solve_out_sanitized.txt"
    if not source.is_file():
        pytest.skip("pinned corpus unavailable")
    result = recover_code_events(TraceIndex(source))
    matches = [v for v in result["versions"] if v["path"] == "/home/ben/task/" + path]
    assert any(v["sha256"] == sha for v in matches), result["unresolved"]
