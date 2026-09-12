"""Prefix selection is chronological, label-free, and independent of the future."""

import hashlib
import json
from datetime import datetime

from tools.outcome_prediction.prefix_dataset import (
    administrative_card_source,
    build_code_events,
    checkpoint_paths,
    code_payload,
    declared_paths,
    digest,
    heredoc_availability,
    literal_shell_sources,
    load_cards,
    recipe_prefix,
    shell_is_recipe,
    strip_python_comments,
    trajectory_inventory,
)
from tools.outcome_prediction.rpm_code_provenance import TraceIndex


def at(second):
    return f"2026-09-04T00:00:{second:02d}Z"


def card(exp_id, first, archive):
    return {
        "card_id": exp_id,
        "first": {"at": at(first), "seq": first},
        "archive": {"at": at(archive), "archived": "/home/ben/task/wm/checkpoints/" + exp_id},
        "plan_card": {"setup": {}},
        "archive_card": {"setup": {}, "result": {}},
        "archive_submission_count": 1,
    }


def event(second, text):
    return {
        "time": datetime.fromisoformat(at(second).replace("Z", "+00:00")),
        "line": second,
        "path": "/home/ben/task/train.py",
        "payload": {"kind": "file_version", "path": "/home/ben/task/train.py", "content": text},
        "flags": [],
        "evidence": {"available_at": at(second)},
    }


def test_prefix_includes_before_plan_and_excludes_at_or_after_archive():
    cards = [card("exp-01", 3, 8), card("exp-02", 10, 20), card("exp-03", 25, 30)]
    stream = [event(s, str(s)) for s in [1, 3, 7, 11, 19, 20, 26]]
    index = type("Index", (), {"cwd": "/home/ben/task"})()
    prefix, _, _ = recipe_prefix(index, cards, cards[1], stream)
    assert [item["exp_id"] for item in prefix] == ["exp-01", "exp-02"]
    assert [c["content"] for item in prefix for c in item["codes"]] == ["1", "3", "7", "11", "19"]


def test_python_cleaning_preserves_code_not_score_comments():
    source = '"""previous accuracy 87%"""\nfrom __future__ import annotations\n# score 0.87\nx = 42\ndef f():\n    """score 0.9"""\n'
    cleaned, warning = strip_python_comments(source)
    assert warning is None
    assert "accuracy" not in cleaned and "score" not in cleaned
    assert "x = 42" in cleaned
    compile(cleaned, "cleaned.py", "exec")


def test_card_serializer_outside_card_directory_is_not_recipe_source(tmp_path):
    content = "import yaml\np='memory/cards/exp-01.yaml'\nd=yaml.safe_load(open(p))\nd['result']={'measurements':[{'value':0.72}]}\nyaml.safe_dump(d,open(p,'w'))\n"
    assert administrative_card_source(content)
    assert not administrative_card_source(content + "trainer.train()\n")
    assert not administrative_card_source("p='data/train.jsonl'\nopen(p,'w').write('data')\n")
    index = type(
        "Index",
        (),
        {
            "cwd": "/home/ben/task",
            "calls": [],
            "snapshots": {
                "/tmp/helper.py": [
                    {
                        "content": content,
                        "time": datetime.fromisoformat(at(2)),
                        "result_line": 3,
                        "call_line": 2,
                        "tool_use_id": "write-card-helper",
                    }
                ]
            },
        },
    )()
    assert not build_code_events(index, [card("exp-01", 1, 8)])[0]
    assert index.omitted_source_events[0]["path"] == "/tmp/helper.py"


def test_executable_literal_is_flagged_not_silently_modified():
    source = "known_accuracy = 0.87\n"
    payload, flags = code_payload(source, "train.py", "file_version")
    assert payload["content"] == source
    assert "possible_outcome_literal_requires_content_review" in flags


def test_percent_log_message_removed_but_training_values_preserved():
    payload, flags = code_payload(
        'cp ckpt final_model; echo "final_model ready (exp01, 23.3%)"', None, "shell_command"
    )
    assert "23.3" not in payload["content"]
    assert "cp ckpt final_model" in payload["content"]
    assert not flags
    payload, flags = code_payload('echo "23.3%" > training_data.txt', None, "shell_command")
    assert "23.3%" in payload["content"]
    assert "percent_literal_requires_content_review" in flags
    payload, _ = code_payload('echo "exp05 was 0.60"', None, "shell_command")
    assert "0.60" not in payload["content"]


def test_python_heredoc_removes_measured_log_not_generation_policy():
    source = "python - <<'PY'\n# exp01 scored 30%\nconfig={'temperature':0.6}\nprint('final_model exp01 (30%)')\nPY"
    payload, _ = code_payload(source, None, "shell_command")
    assert "30%" not in payload["content"]
    assert "'temperature':0.6" in payload["content"]


def test_target_source_directory_wins_over_logging_directory():
    value = card("exp-01", 1, 8)
    value["archive_card"] = {
        "setup": {"output_dir": "/home/ben/task/logs"},
        "result": {"output_checkpoint": "/home/ben/task/model/final"},
    }
    assert checkpoint_paths(value) == [
        "/home/ben/task/model/final",
        "/home/ben/task/wm/checkpoints/exp-01",
    ]


def test_declared_builders_split_explicit_paths_not_free_text():
    source = {
        "setup": {
            "command": {"script": "none"},
            "data": [{"built_by": "build_a.py + /task/filter.py and inline mixing"}],
        }
    }
    assert declared_paths(source, "/task") == {"/task/build_a.py", "/task/filter.py"}


def test_shorthand_alias_requires_unique_observed_invocation_before_cutoff():
    target = card("exp-01", 2, 8)
    target["plan_card"] = {"setup": {"data": [{"built_by": "helper.py"}]}}
    source = event(1, "x=1")
    source["path"] = "/home/ben/task/scripts/helper.py"
    source["payload"]["path"] = source["path"]
    call = {
        "name": "Bash",
        "id": "launch",
        "line": 5,
        "time": datetime.fromisoformat(at(5)),
        "result": {"time": datetime.fromisoformat(at(6)), "is_error": False},
        "input": {"command": "cd /home/ben/task; HF_HOME=/cache python scripts/helper.py --n 3"},
    }
    index = type("Index", (), {"cwd": "/home/ben/task", "calls": [call]})()
    prefix, audit, missing = recipe_prefix(index, [target], target, [source])
    assert not missing
    assert audit[0]["declared_path_aliases"][0]["resolved_path"] == source["path"]
    assert prefix[0]["codes"] == [source["payload"]]
    call["time"] = datetime.fromisoformat(at(9))
    assert recipe_prefix(index, [target], target, [source])[2]
    call["time"] = datetime.fromisoformat(at(5))
    command = call["input"]["command"]
    call["input"]["command"] = "false && python scripts/helper.py"
    assert recipe_prefix(index, [target], target, [source])[2]
    call["input"]["command"] = command
    call["result"]["is_error"] = True
    assert recipe_prefix(index, [target], target, [source])[2]
    call["result"]["is_error"] = False
    another = event(1, "x=2")
    another["path"] = "/home/ben/task/other/helper.py"
    assert recipe_prefix(index, [target], target, [source, another])[2]


def test_oneoff_config_and_literal_source_writes_are_code():
    assert shell_is_recipe(
        'python -c "import json; json.dump({}, open("out/generation_config.json","w"))"', set()
    )
    assert shell_is_recipe("cat > helper.py <<'PY'\nx=1\nPY", set())
    assert not shell_is_recipe("cat eval_results.json", set())
    assert not shell_is_recipe("python submit.py memory/cards/exp-01.yaml --score 0.9", set())
    assert not shell_is_recipe("awm wm outcome --card exp-01 --final 0.8; python train.py", set())
    assert not shell_is_recipe("cat > SKILL.md <<'MD'\npython train.py: score0.8\nMD", set())
    assert not shell_is_recipe("python3 -c \"# train notes\nprint('exp01:0%, exp02:0%')\"", set())


def test_literal_shell_source_only_quoted_non_expanding_heredocs():
    assert literal_shell_sources("cat > train.py <<'PY'\nx=1\nPY", "/task") == [
        ("/task/train.py", "x=1\n")
    ]
    assert literal_shell_sources("cat <<'PY' > helper.py\nx=2\nPY", "/task") == [
        ("/task/helper.py", "x=2\n")
    ]
    assert not literal_shell_sources("cat > train.py <<PY\nx=$SECRET\nPY", "/task")
    assert not literal_shell_sources("cat > $OUT/train.py <<'PY'\nx=1\nPY", "/task")


def test_same_call_ledger_snapshot_bounds_initial_write_without_time_slack():
    target = card("exp-01", 1, 8)
    entry = {"at": at(4), "seq": 2, "snapshotted": ["helper.py"], "source": "/task/card.yaml"}
    target["submissions"] = [entry]
    call = {
        "time": datetime.fromisoformat(at(4)),
        "result": {"time": datetime.fromisoformat(at(9))},
        "input": {"command": "cat > helper.py << 'PY'\nx=1\nPY\nawm wm submit card.yaml"},
    }
    available, evidence = heredoc_availability(call, "/task/helper.py", [target], "/task")
    assert available == datetime.fromisoformat(at(5))
    assert evidence["snapshot_attestation_seq"] == 2
    assert (
        heredoc_availability(call, "/task/other.py", [target], "/task")[0] == call["result"]["time"]
    )
    entry["at"] = at(9)
    assert (
        heredoc_availability(call, "/task/helper.py", [target], "/task")[0]
        == call["result"]["time"]
    )
    entry["at"] = at(4)
    call["input"]["command"] = call["input"]["command"].replace(
        "awm wm", "echo unknown > helper.py\nawm wm"
    )
    assert (
        heredoc_availability(call, "/task/helper.py", [target], "/task")[0]
        == call["result"]["time"]
    )


def test_future_card_cannot_change_retained_code(tmp_path):
    path = tmp_path / "trace.txt"
    stamp = at(2)
    call = {
        "timestamp": stamp,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "one",
                    "name": "Bash",
                    "input": {"command": "python evaluate.py --model-path out"},
                }
            ]
        },
    }
    path.write_text(f"[{stamp}] {json.dumps(call)}\n")
    index = TraceIndex(path)
    early = card("exp-01", 1, 8)
    future = card("exp-02", 20, 30)
    future["plan_card"] = {"setup": {"command": {"script": "evaluate.py"}}}
    assert build_code_events(index, [early]) == build_code_events(index, [early, future])


def test_repeated_archive_reference_does_not_move_original_cutoff(tmp_path):
    directory = tmp_path / "session"
    card_dir = directory / "wm/cards/exp-01"
    card_dir.mkdir(parents=True)
    entries = []
    for i, stage in [(1, "plan"), (2, "closed"), (3, "closed")]:
        name = f"record-{i:02d}.json"
        (card_dir / name).write_text(json.dumps({"card": {"setup": {}, "result": {}}}))
        entries.append(
            {
                "card_id": "exp-01",
                "seq": i,
                "event": "submit",
                "stage": stage,
                "path": "/wm/cards/exp-01/" + name,
                "at": at(i),
                "archived": "/wm/checkpoints/exp-01" if i > 1 else None,
            }
        )
    (directory / "wm/records.jsonl").write_text("\n".join(json.dumps(x) for x in entries))
    recovered = load_cards(directory)[0]
    assert recovered["archive"]["at"] == at(2)
    assert recovered["archive_submission_count"] == 2


def test_inventory_retains_trajectory_without_checkpoint(tmp_path):
    directory = tmp_path / "cells/unlabeled"
    directory.mkdir(parents=True)
    (directory / "solve_out_sanitized.txt").write_text("")
    inventory = trajectory_inventory(tmp_path)
    assert inventory[0]["trajectory_id"] == "unlabeled"
    assert inventory[0]["status"] == "no_archived_checkpoint_in_manifest"


def test_snapshot_recovery_requires_pre_archive_time_hash_and_attestation(tmp_path):
    target = card("exp-01", 1, 8)
    target["submissions"] = [{"at": at(7), "seq": 2, "snapshotted": ["train.py"]}]
    index = type("Index", (), {"cwd": "/home/ben/task", "snapshots": {}, "calls": []})()
    directory = tmp_path / "wm/cards/exp-01/snapshot"
    directory.mkdir(parents=True)
    raw = b"x=1\n"
    (directory / "train.py").write_bytes(raw)
    entry = {
        "path": "train.py",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "at": at(6),
    }
    manifest = directory / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": [entry]}))
    events, _ = build_code_events(index, [target], tmp_path)
    assert events[0]["payload"]["content"] == "x=1\n"
    assert events[0]["evidence"]["snapshot_attestation_seq"] == 2
    entry["at"] = at(9)
    manifest.write_text(json.dumps({"files": [entry]}))
    assert not build_code_events(index, [target], tmp_path)[0]


def test_input_hash_does_not_depend_on_attached_label():
    model_input = {"prefix_recipes": [{"codes": [{"content": "x=1"}]}], "generation_config": None}
    row = {"input": model_input, "output": {"avg_pass_rate": 0.1}}
    before = digest(row["input"])
    row["output"]["avg_pass_rate"] = 0.9
    assert before == digest(row["input"])
