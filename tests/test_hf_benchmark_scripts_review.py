"""Independent adversarial checks for recipe/data/environment provenance."""

import json
from datetime import datetime, timezone

import pytest
from test_hf_benchmark_scripts import TRAINING, _at, _event, _fixture, _write

from tools.outcome_prediction.hf_benchmark_scripts import (
    _select_builder,
    extract_ptb_scripts,
    parse_launches,
)


def test_explicit_data_identity_disallows_fallback_to_wrong_builder_output():
    candidate = {
        **parse_launches("python build.py --out /data/wrong", "/task")[0],
        "time": datetime(2026, 9, 4, tzinfo=timezone.utc), "line": 1,
    }
    setup = {"data": [{"built_by": "build.py", "path": "/data/wanted",
                       "build_command": "python build.py --out /data/wanted"}]}
    assert _select_builder([candidate], "build.py", setup, "/task") is None


@pytest.mark.parametrize("prefix", [
    "export LR=$NEXT; ",
    "source ./environment.sh; ",
    ". ./environment.sh; ",
    "export LR=1e-5; unset LR; ",
    "export LR=1e-5; export LR=$NEXT; ",
    "source ./environment.sh || true; ",
    "LR=1e-5 source ./environment.sh; ",
    "builtin export LR=9e-4; ",
])
def test_unknown_environment_mutation_cannot_produce_eligible_recipe(tmp_path, prefix):
    source = "import os\n" + TRAINING.replace(
        "learning_rate=1e-5", "learning_rate=float(os.environ.get('LR', '1e-5'))")
    _fixture(tmp_path, command=prefix + "python train.py --epochs 2", source=source, after=False)
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate", result["launch"]


def test_failed_builder_cannot_certify_preexisting_training_data(tmp_path):
    session = _fixture(tmp_path, after=False)
    trace = session / "solve_out_sanitized.txt"
    builder = _write(3, "/home/ben/task/build.py", 'raise RuntimeError("build failed")\n', "builder_source")
    builder += _event(_at(5), [{"type": "tool_use", "id": "builder_launch", "name": "Bash",
                              "input": {"command": "python build.py --out /data/wanted"}}])
    builder += _event(_at(6), [{"type": "tool_result", "tool_use_id": "builder_launch", "is_error": True}])
    trace.write_text(trace.read_text().replace(f"[{_at(10)}]", builder + f"[{_at(10)}]", 1))
    for path in (session / "wm/cards/exp-01").glob("record-*.json"):
        record = json.loads(path.read_text())
        record["card"]["setup"]["data"] = [{
            "built_by": "build.py", "path": "/data/wanted",
            "build_command": "python build.py --out /data/wanted",
        }]
        path.write_text(json.dumps(record))
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate", result["launch"]


def test_named_builder_with_unbound_fixed_output_is_not_eligible(tmp_path):
    session = _fixture(tmp_path, after=False)
    trace = session / "solve_out_sanitized.txt"
    builder = _write(3, "/home/ben/task/build.py", 'open("/data/wrong", "w").write("new data")\n', "builder_source")
    builder += _event(_at(5), [{"type": "tool_use", "id": "builder_launch", "name": "Bash",
                              "input": {"command": "python build.py"}}])
    builder += _event(_at(6), [{"type": "tool_result", "tool_use_id": "builder_launch"}])
    trace.write_text(trace.read_text().replace(f"[{_at(10)}]", builder + f"[{_at(10)}]", 1))
    for path in (session / "wm/cards/exp-01").glob("record-*.json"):
        record = json.loads(path.read_text())
        record["card"]["setup"]["data"] = [{
            "built_by": "build.py", "path": "/data/wanted", "build_command": "python build.py",
        }]
        path.write_text(json.dumps(record))
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate", result["launch"]


def test_unarchived_writer_cannot_be_hidden_by_parent_recipe_composition(tmp_path):
    session = _fixture(tmp_path, after=False)
    trace = session / "solve_out_sanitized.txt"
    # The unarchived training launch writes the same fixed output directory as
    # the archived parent. It has no --out flag, so argv-only writer detection
    # must not certify that the old archived parent supplied the child's input.
    extra = _write(21, "/home/ben/task/unarchived.py", TRAINING, "unarchived_source")
    extra += _event(_at(23), [{"type": "tool_use", "id": "unarchived_launch", "name": "Bash",
                            "input": {"command": "python unarchived.py"}}])
    extra += _event(_at(24), [{"type": "tool_result", "tool_use_id": "unarchived_launch"}])
    child = TRAINING.replace('"google/gemma-3-4b-pt"', '"/home/ben/task/checkpoint"')
    child = child.replace('output_dir="/home/ben/task/checkpoint"', 'output_dir="/home/ben/task/child"')
    child = child.replace('save_model("/home/ben/task/checkpoint")', 'save_model("/home/ben/task/child")')
    extra += _write(25, "/home/ben/task/child.py", child, "child_source")
    extra += _event(_at(30), [{"type": "tool_use", "id": "child_launch", "name": "Bash",
                            "input": {"command": "python child.py"}}])
    extra += _event(_at(31), [{"type": "tool_result", "tool_use_id": "child_launch"}])
    trace.write_text(trace.read_text() + extra)
    directory = session / "wm/cards/exp-02"
    directory.mkdir()
    card = {"setup": {
        "base_model": "google/gemma-3-4b-pt",
        "command": {"argv": ["python", "child.py"], "cwd": "/home/ben/task", "script": "child.py"},
        "output_dir": "/home/ben/task/child", "data": [],
        "parent_checkpoint": {"origin": "previous_experiment", "path": "/home/ben/task/checkpoint"},
    }, "result": {"output_checkpoint": "/home/ben/task/child"}}
    for number, second in [(1, 28), (2, 40)]:
        (directory / f"record-{number:02d}.json").write_text(json.dumps({"at": _at(second), "card": card}))
    ledger = session / "wm/records.jsonl"
    entries = [
        {"event": "submit", "seq": 3, "card_id": "exp-02", "stage": "plan", "at": _at(28), "path": "record-01.json"},
        {"event": "submit", "seq": 4, "card_id": "exp-02", "stage": "closed", "at": _at(40), "path": "record-02.json", "archived": "/archive/exp-02"},
    ]
    ledger.write_text(ledger.read_text() + "".join(json.dumps(e) + "\n" for e in entries))
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-02"]
    assert result["status"] == "candidate", result["launch"]
