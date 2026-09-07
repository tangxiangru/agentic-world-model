"""Recorded-version reconstruction never falls back to final card snapshots."""

import json
from pathlib import Path

import pytest

from tools.outcome_prediction.rpm_code_provenance import (
    TraceIndex,
    explicit_shell_mutations,
    reconstruct_card,
)

SCRIPT = "/home/ben/task/train.py"


def event(second, content, result=None):
    at = f"2026-09-04T00:00:{second:02d}Z"
    value = {"timestamp": at, "message": {"content": content}}
    if result is not None:
        value["tool_use_result"] = result
    return f"[{at}] {json.dumps(value)}\n"


def call(second, tool, arguments, ident="one"):
    return event(second, [{"type": "tool_use", "id": ident, "name": tool, "input": arguments}])


def result(second, value, ident="one", error=False):
    return event(second, [{"type": "tool_result", "tool_use_id": ident, "is_error": error}], value)


def write_events():
    return call(1, "Write", {"file_path": SCRIPT, "content": "x = 1\n"}) + result(
        2, {"filePath": SCRIPT, "content": "x = 1\n", "userModified": False}
    )


def index(tmp_path, events):
    path = tmp_path / "trace.txt"
    path.write_text(events)
    return TraceIndex(path)


def test_successful_write_preserves_exact_bytes(tmp_path):
    recovered = index(tmp_path, write_events()).reconstruct(SCRIPT, "2026-09-04T00:00:03Z")
    assert recovered["status"] == "reconstructed"
    assert recovered["content"] == "x = 1\n"
    assert recovered["evidence"]["result_line"] == 2


def test_edit_full_original_can_reconstruct_without_prior_write(tmp_path):
    trace = call(1, "Edit", {"file_path": SCRIPT, "old_string": "1", "new_string": "2"})
    trace += result(
        2, {"filePath": SCRIPT, "originalFile": "x = 1\n", "oldString": "1", "newString": "2"}
    )
    recovered = index(tmp_path, trace).reconstruct(SCRIPT, "2026-09-04T00:00:03Z")
    assert recovered["content"] == "x = 2\n"
    assert recovered["evidence"]["tool"] == "Edit"
    assert recovered["evidence"]["original_file_sha256"]


def test_result_must_be_strictly_before_cutoff(tmp_path):
    recovered = index(tmp_path, write_events()).reconstruct(SCRIPT, "2026-09-04T00:00:02Z")
    assert recovered["status"] == "unavailable"
    assert recovered["content"] is None


def test_pending_mutation_blocks_earlier_complete_capture(tmp_path):
    trace = write_events() + call(
        3, "Edit", {"file_path": SCRIPT, "old_string": "1", "new_string": "2"}, "two"
    )
    recovered = index(tmp_path, trace).reconstruct(SCRIPT, "2026-09-04T00:00:04Z")
    assert recovered["status"] == "blocked"
    assert recovered["content"] is None


def test_failed_edit_does_not_change_prior_version(tmp_path):
    trace = write_events() + call(
        3, "Edit", {"file_path": SCRIPT, "old_string": "1", "new_string": "2"}, "two"
    )
    trace += result(4, "Error", "two", error=True)
    recovered = index(tmp_path, trace).reconstruct(SCRIPT, "2026-09-04T00:00:05Z")
    assert recovered["status"] == "reconstructed"
    assert recovered["content"] == "x = 1\n"


def test_ambiguous_edit_blocks_prior_version(tmp_path):
    trace = write_events() + call(
        3, "Edit", {"file_path": SCRIPT, "old_string": "1", "new_string": "2"}, "two"
    )
    trace += result(
        4,
        {"filePath": SCRIPT, "originalFile": "x = 11\n", "oldString": "1", "newString": "2"},
        "two",
    )
    recovered = index(tmp_path, trace).reconstruct(SCRIPT, "2026-09-04T00:00:05Z")
    assert recovered["status"] == "blocked"


@pytest.mark.parametrize(
    "command",
    [
        "echo changed > train.py",
        "rm train.py && echo done",
        "cp another.py train.py",
        "sed -i 's/1/2/' train.py",
        "python -c \"p='train.py'; open(p,'w').write('x=2')\"",
        "python - <<'PY'\np = 'train.py'\ns = open(p).read()\nopen(p, 'w').write(s.replace('1','2'))\nPY\n",
    ],
)
def test_explicit_shell_write_is_a_blocker(tmp_path, command):
    assert explicit_shell_mutations(command, SCRIPT)
    trace = write_events() + call(3, "Bash", {"command": command}, "two")
    recovered = index(tmp_path, trace).reconstruct(SCRIPT, "2026-09-04T00:00:04Z")
    assert recovered["status"] == "blocked"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf runs/bench1 && python train.py --output runs/bench2",
        "rm runs/bench1; python train.py",
        "rm runs/bench1 || python train.py",
        "cat train.py | tee run.log",
        "cp train.py elsewhere.py",
        "cat > config.yaml <<'YAML'\nscript: train.py\nYAML\n",
    ],
)
def test_unrelated_shell_mutation_does_not_block_script(command):
    assert explicit_shell_mutations(command, SCRIPT) == []


def test_no_snapshot_fallback(tmp_path):
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "snapshot/train.py").write_text("print('future code')\n")
    recovered = index(tmp_path, "").reconstruct(SCRIPT, "2026-09-04T00:00:04Z")
    assert recovered["status"] == "unavailable"
    assert recovered["content"] is None


@pytest.mark.needs_data
def test_real_r0_30_siblings_share_preplan_training_version():
    cell = Path(__file__).resolve().parents[1] / "data/traj/raw/awm-gsm8k-trajectories/cells/r0-30"
    if not (cell / "solve_out_sanitized.txt").exists():
        pytest.skip("Downloaded recorder trace is unavailable")
    trace = TraceIndex(cell / "solve_out_sanitized.txt")
    cards = [
        json.loads((cell / f"wm/cards/exp-{i:02d}/record-01.json").read_text()) for i in (5, 6)
    ]
    recovered = [reconstruct_card(trace, card["card"], card["at"]) for card in cards]
    assert all(all(s["status"] == "reconstructed" for s in scripts) for scripts in recovered)
    assert (
        recovered[0][0]["sha256"]
        == recovered[1][0]["sha256"]
        == "e5a258785d779ec6290803af946bdfb7f99fccdab1fa39650750a2589ed1658a"
    )
    assert recovered[0][1]["sha256"] != recovered[1][1]["sha256"]
    assert all(scripts[0]["evidence"]["result_line"] == 1165 for scripts in recovered)
