"""Recorded launches, temporal code selection, and outcome isolation."""

import json

from tools.outcome_prediction.hf_benchmark_scripts import (
    _attach_parents,
    extract_ptb_scripts,
    parse_launches,
)


def _at(second):
    return f"2026-09-04T00:00:{second:02d}Z"


def _event(at, message, result=None):
    event = {"timestamp": at, "message": {"content": message}}
    if result is not None:
        event["tool_use_result"] = result
    return f"[{at}] " + json.dumps(event) + "\n"


def _write(at, path, content, key):
    return _event(_at(at), [{"type": "tool_use", "id": key, "name": "Write", "input": {
        "file_path": path, "content": content,
    }}]) + _event(_at(at + 1), [{"type": "tool_result", "tool_use_id": key}], {
        "filePath": path, "content": content,
    })


TRAINING = '''"""Earlier accuracy was 0.92."""
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
# accuracy 92%
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt")
args = TrainingArguments(output_dir="/home/ben/task/checkpoint", learning_rate=1e-5)
trainer = Trainer(model=model, args=args)
trainer.train()
trainer.save_model("/home/ben/task/checkpoint")
'''


def _fixture(root, *, before=True, after=True, command=None, extra="", source=TRAINING):
    session = root / "cells/r0-01"
    directory = session / "wm/cards/exp-01"
    directory.mkdir(parents=True)
    command = command or "python train.py --epochs 2 > train.log 2>&1"
    trace = _write(1, "/home/ben/task/train.py", source, "before") if before else ""
    trace += _event(_at(10), [{"type": "tool_use", "id": "launch", "name": "Bash", "input": {
        "command": command,
    }}])
    trace += _event(_at(11), [{"type": "tool_result", "tool_use_id": "launch"}])
    if after:
        trace += _write(12, "/home/ben/task/train.py", "known_accuracy = 0.99\n", "after")
    trace += extra
    (session / "solve_out_sanitized.txt").write_text(trace)
    setup = {
        "base_model": "google/gemma-3-4b-pt",
        "command": {"argv": ["python", "train.py", "--epochs", "2"],
                    "cwd": "/home/ben/task", "script": "train.py"},
        "output_dir": "/home/ben/task/checkpoint", "data": [],
        "parent_checkpoint": {"origin": "base_model", "path": "google/gemma-3-4b-pt"},
    }
    card = {"setup": setup, "result": {"output_checkpoint": "/home/ben/task/checkpoint"},
            "conclusion": {"accuracy": 0.999}, "evaluation": {"accuracy": 0.999}}
    for number, second in [(1, 8), (2, 20)]:
        (directory / f"record-{number:02d}.json").write_text(json.dumps({"at": _at(second), "card": card}))
    entries = [{"event": "submit", "seq": 1, "card_id": "exp-01", "stage": "plan",
                "at": _at(8), "path": "record-01.json"},
               {"event": "submit", "seq": 2, "card_id": "exp-01", "stage": "closed",
                "at": _at(20), "path": "record-02.json", "archived": "/archive/exp-01"}]
    (session / "wm/records.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    return session


def test_parse_launch_preserves_numeric_argument_and_removes_log_redirection():
    values = parse_launches(
        "cd /work; export HF_HOME=/cache; nohup CUDA_VISIBLE_DEVICES=0 python train.py --epochs 2 > log 2>&1 & echo $!",
        "/old",
    )
    assert len(values) == 1
    assert values[0]["argv"] == ["python", "train.py", "--epochs", "2"]
    assert values[0]["cwd"] == "/work"
    assert values[0]["env"] == {"HF_HOME": "/cache", "CUDA_VISIBLE_DEVICES": "0"}


def test_heredocs_input_pipes_and_dynamic_launches_are_not_invented():
    assert parse_launches("cat > x.py <<'PY'\npython train.py\nPY", "/task") == []
    assert parse_launches("python train.py | arbitrary_consumer", "/task") == []
    assert parse_launches("echo inputs | python train.py", "/task") == []
    assert parse_launches("python train.py --model $MODEL", "/task") == []


def test_diagnostic_output_pipe_keeps_actual_producer_invocation():
    launches = parse_launches("python build.py --n 2 2>&1 | tail -5", "/task")
    assert len(launches) == 1
    assert launches[0]["argv"] == ["python", "build.py", "--n", "2"]


def test_extract_uses_prelaunch_version_not_later_archive_snapshot(tmp_path):
    session = _fixture(tmp_path)
    snapshots = session / "wm/cards/exp-01/snapshot"
    snapshots.mkdir()
    (snapshots / "train.py").write_text("known_accuracy=0.99\n")
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "eligible", result["exclusion_reasons"]
    source = result["scripts"][0]["content"]
    assert "learning_rate=1e-5" in source
    assert "accuracy" not in source and "0.99" not in source
    assert "conclusion" not in json.dumps(result["launch"])
    assert result["provenance"]["script_evidence"][0]["call_line"] == 1


def test_postlaunch_only_source_is_candidate(tmp_path):
    _fixture(tmp_path, before=False)
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate"
    assert not result["scripts"]
    assert any(r.startswith("no_recorded_script_version_before_launch") for r in result["exclusion_reasons"])


def test_wrong_recorded_argv_cannot_bind_to_training(tmp_path):
    _fixture(tmp_path, command="python train.py --epochs 8")
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate"
    assert "no_observed_launch_matching_recorded_argv" in result["exclusion_reasons"]


def test_executable_outcome_literals_are_candidates(tmp_path):
    _fixture(tmp_path, source=TRAINING + "known_accuracy=0.85\n")
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate"
    assert "possible_outcome_literal_requires_content_review" in result["review_flags"]


def test_intervening_shell_mutation_blocks_old_version(tmp_path):
    session = _fixture(tmp_path, after=False)
    trace = session / "solve_out_sanitized.txt"
    content = trace.read_text()
    inserted = _event(_at(5), [{"type": "tool_use", "id": "change", "name": "Bash", "input": {
        "command": "perl -pi -e 's/1e-5/3e-5/' train.py",
    }}])
    trace.write_text(content.replace(f"[{_at(10)}]", inserted + f"[{_at(10)}]", 1))
    result = extract_ptb_scripts(tmp_path)["r0-01-exp-01"]
    assert result["status"] == "candidate"
    assert any(r.startswith("intervening_shell_script_mutation_unresolved") for r in result["exclusion_reasons"])


def test_only_unique_eligible_direct_parent_recipe_is_attached():
    parent = {
        "session_id": "s", "scripts": [{"path": "train.py", "content": "x=1", "role": "training"}],
        "launch": {"argv": ["python", "train.py"]}, "status": "eligible", "exclusion_reasons": [],
        "parent_checkpoint_ids": [],
        "provenance": {"target_output_path": "/ckpt/one", "first_archive_at": _at(3)},
    }
    child = {
        "session_id": "s", "scripts": [], "launch": {"cwd": "/task"}, "status": "candidate",
        "exclusion_reasons": ["parent_training_recipe_not_yet_bound"], "parent_checkpoint_ids": [],
        "provenance": {"launch_evidence": {"at": _at(8)},
                       "checkpoint_inputs": {"inputs": [{"path": "/ckpt/one"}]}},
    }
    rows = {"s-exp-01": parent, "s-exp-02": child}
    _attach_parents(rows)
    assert child["status"] == "eligible"
    assert child["parent_checkpoint_ids"] == ["s-exp-01"]
    assert child["launch"]["parent_recipes"][0]["scripts"] == parent["scripts"]
