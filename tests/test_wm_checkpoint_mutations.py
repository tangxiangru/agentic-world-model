"""Synthetic bounded mutation audits; no archived code execution or training."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.outcome_prediction.wm_checkpoint_mutations import audit_source_use


def fixture(tmp_path):
    cell = tmp_path / "cell"
    record = cell / "wm/cards/exp-01/record-02.json"
    record.parent.mkdir(parents=True)
    ledger = cell / "wm/records.jsonl"
    ledger.write_text("")
    (cell / "solve_out_sanitized.txt").write_text("")
    parent = {
        "example_id": "cell/exp-01", "card_id": "exp-01", "cell_id": "cell",
        "source_path": "/work/runs/parent", "archive_path": "/work/wm/checkpoints/exp-01",
        "consumed_path": "/work/runs/parent", "archived_at": "2026-09-05T11:00:00Z",
        "archived_setup": {"method": {"family": "sft"}},
        "evidence": {"ledger_path": str(ledger), "ledger_sha256": hashlib.sha256(b"").hexdigest(), "record_path": str(record), "record_n": 2},
    }
    child = {"example_id": "cell/exp-02", "cell_id": "cell", "first_submitted_at": "2026-09-05T12:00:00Z"}
    rows = [{"example_id": parent["example_id"], "cell_id": "cell"}, child]
    return parent, child, rows, {parent["example_id"]: parent}


def audit(parts):
    before = copy.deepcopy(parts)
    result = audit_source_use(*parts)
    assert parts == before
    return result


def add_record(parts, *, setup=None, output="/work/runs/parent", at="2026-09-05T11:30:00Z"):
    parent = parts[0]
    record = {"at": at, "card": {"card_id": "exp-01", "setup": setup or parent["archived_setup"], "result": {"output_checkpoint": output}}}
    record_path = Path(parent["evidence"]["record_path"]).with_name("record-03.json")
    record_path.write_text(json.dumps(record))
    ledger = Path(parent["evidence"]["ledger_path"])
    ledger.write_text(json.dumps({"event": "submit", "card_id": "exp-01", "record_n": 3, "at": at, "archived": parent["archive_path"]}) + "\n")
    parent["evidence"]["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()


def writer(path, at="2026-09-05T11:30:00Z", **extra):
    return {"example_id": "cell/exp-03", "cell_id": "cell", "first_submitted_at": at, "model_input": {"plan": {"setup": {"output_dir": path, "command": {"cwd": "/work"}}}}, **extra}


def write_trace(parts, command=None, name="Bash", at="2026-09-05T11:30:00Z", result_text=None):
    parent = parts[0]
    path = Path(parent["evidence"]["ledger_path"]).parent.parent / "solve_out_sanitized.txt"
    inp = {"command": command} if name == "Bash" else {"file_path": command, "content": "ignored"}
    event = {"message": {"content": [{"type": "tool_use", "id": "call-1", "name": name, "input": inp}]}}
    lines = [f"[{at}] " + json.dumps(event)]
    if result_text is not None:
        result = {"message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "content": result_text}]}}
        lines.append(f"[{at}] " + json.dumps(result))
    path.write_text("\n".join(lines) + "\n")


def test_clear_is_bounded_and_inputs_not_mutated(tmp_path):
    result = audit(fixture(tmp_path))
    assert result["status"] == "clear"
    assert result["limitations"]
    assert any(e["kind"] == "trace_checked" for e in result["evidence"])


def test_immutable_archive_avoids_mutable_source_conflict(tmp_path):
    parts = fixture(tmp_path)
    parts[0]["consumed_path"] = parts[0]["archive_path"]
    parts[2].append(writer("/work/runs/parent"))
    result = audit(parts)
    assert result["status"] == "clear"
    assert any(e["kind"] == "immutable_archive_path" for e in result["evidence"])


@pytest.mark.parametrize("field,value", [("consumed_path", "/different"), ("archived_at", "2026-09-05T12:00:01Z"), ("archived_at", "missing"), ("cell_id", "other")])
def test_invalid_context_fails_closed(tmp_path, field, value):
    parts = fixture(tmp_path); parts[0][field] = value
    assert audit(parts)["reasons"] == ["invalid_source_use_context"]


@pytest.mark.parametrize("output", ["/work/runs/parent", "/work/runs/parent/checkpoint-2", "/work/runs"])
def test_ungraded_other_card_output_overlap_is_held(tmp_path, output):
    parts = fixture(tmp_path); parts[2].append(writer(output, label=None))
    assert "other_card_proposes_overlapping_output" in audit(parts)["reasons"]


@pytest.mark.parametrize("output,at", [("/work/runs/parent2", "2026-09-05T11:30:00Z"), ("/work/runs/parent", "2026-09-05T10:00:00Z"), ("/work/runs/parent", "2026-09-05T12:00:00Z")])
def test_path_components_and_proposal_interval(tmp_path, output, at):
    parts = fixture(tmp_path); parts[2].append(writer(output, at))
    assert audit(parts)["status"] == "clear"


def test_output_argv_alias_detected(tmp_path):
    parts = fixture(tmp_path); other = writer("unrelated")
    other["model_input"]["plan"]["setup"]["command"]["argv"] = ["python", "train.py", "--output-dir=runs/parent"]
    parts[2].append(other)
    assert "other_card_proposes_overlapping_output" in audit(parts)["reasons"]


def test_child_in_place_output_is_not_a_prior_writer(tmp_path):
    parts = fixture(tmp_path)
    parts[1]["model_input"] = writer("/work/runs/parent")["model_input"]
    assert audit(parts)["status"] == "clear"


def test_earlier_proposal_later_archive_is_a_possible_writer(tmp_path):
    parts = fixture(tmp_path); other = writer("/work/runs/parent", "2026-09-05T10:00:00Z")
    parts[2].append(other)
    parts[3][other["example_id"]] = {"source_path": "/work/runs/parent", "archived_at": "2026-09-05T11:30:00Z"}
    assert "other_card_archives_overlapping_output" in audit(parts)["reasons"]


@pytest.mark.parametrize("change", ["setup", "output"])
def test_same_card_later_changed_record_held(tmp_path, change):
    parts = fixture(tmp_path)
    add_record(parts, setup={"method": "new"} if change == "setup" else None, output="/work/other" if change == "output" else "/work/runs/parent")
    assert "same_card_setup_or_output_changed_after_archive" in audit(parts)["reasons"]


def test_unchanged_resubmission_and_future_changes_do_not_hold(tmp_path):
    parts = fixture(tmp_path); add_record(parts)
    assert audit(parts)["status"] == "clear"
    add_record(parts, output="/work/other", at="2026-09-05T12:01:00Z")
    assert audit(parts)["status"] == "clear"


def test_ledger_hash_change_is_not_silently_rebound(tmp_path):
    parts = fixture(tmp_path)
    Path(parts[0]["evidence"]["ledger_path"]).write_text("[]\n")
    assert audit(parts)["reasons"] == ["mutation_ledger_hash_changed"]


def test_missing_mutable_source_trace_fails_closed(tmp_path):
    parts = fixture(tmp_path)
    parts[0]["evidence"]["trace_path"] = str(tmp_path / "not_exported.txt")
    assert audit(parts)["reasons"] == ["mutation_trace_unavailable"]


@pytest.mark.parametrize("command,name", [
    ("/work/runs/parent/generation_config.json", "Write"),
    ("/work/runs/parent/model-00001.safetensors", "Edit"),
    ("cp replacement.json /work/runs/parent/config.json", "Bash"),
    ("rm -rf /work/runs/parent", "Bash"),
    ("mv /work/runs/parent /work/moved", "Bash"),
    ("cd /work && sed -i 's/a/b/g' runs/parent/config.json", "Bash"),
    ("python -c \"open('/work/runs/parent/generation_config.json', 'w').write('{}')\"", "Bash"),
])
def test_explicit_checkpoint_mutation_attempts(tmp_path, command, name):
    parts = fixture(tmp_path); write_trace(parts, command, name)
    assert "explicit_checkpoint_mutation_attempt" in audit(parts)["reasons"]


@pytest.mark.parametrize("command,name", [
    ("cat /work/runs/parent/config.json", "Bash"),
    ("cp /work/runs/parent/config.json /work/elsewhere.json", "Bash"),
    ("/work/runs/parent/README.md", "Write"),
    ("/work/runs/parent2/config.json", "Write"),
    ("printf 'echo x > /work/runs/parent/config.json'", "Bash"),
    ("python -c \"print('comparison > /work/runs/parent')\"", "Bash"),
    ("cat <<'TXT'\nopen('/work/runs/parent/config.json','w')\nTXT", "Bash"),
])
def test_reads_and_unrelated_files_do_not_hold(tmp_path, command, name):
    parts = fixture(tmp_path); write_trace(parts, command, name)
    assert audit(parts)["status"] == "clear"


def test_labels_and_tool_result_prose_do_not_affect_decision(tmp_path):
    parts = fixture(tmp_path); parts[0]["accuracy"] = 0.9; parts[1]["label"] = {"accuracy": 0.0}
    write_trace(parts, "cat /work/runs/parent/config.json", result_text="accuracy 0.8; cp x /work/runs/parent/config.json")
    assert audit(parts)["status"] == "clear"
    parts[0]["accuracy"] = 0.0; parts[1]["label"] = {"accuracy": 1.0}
    assert audit(parts)["status"] == "clear"


def test_mutation_after_proposal_excluded(tmp_path):
    parts = fixture(tmp_path); write_trace(parts, "/work/runs/parent/config.json", "Write", at="2026-09-05T12:00:01Z")
    assert audit(parts)["status"] == "clear"


def test_equal_second_requires_matching_ledger_order(tmp_path):
    parts = fixture(tmp_path); parent, child = parts[:2]
    parent["archived_at"] = child["first_submitted_at"]
    assert audit(parts)["reasons"] == ["equal_timestamp_order_unproven"]
    child["card_id"] = "exp-02"
    ledger = Path(parent["evidence"]["ledger_path"])
    events = [
        {"card_id": "exp-01", "event": "submit", "stage": "closed", "record_n": 2, "at": parent["archived_at"]},
        {"card_id": "exp-02", "event": "submit", "stage": "plan", "record_n": 1, "at": child["first_submitted_at"], "path": "/work/wm/cards/exp-02/record-01.json"},
    ]
    ledger.write_text("\n".join(map(json.dumps, events)) + "\n")
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    parent["evidence"].update(ledger_line=1, ledger_sha256=digest)
    child["provenance"] = {"ledger_path": str(ledger), "ledger_sha256": digest, "first_record_path": "/unused/record-01.json"}
    assert audit(parts)["status"] == "clear"
    parent["evidence"]["ledger_line"] = 3
    assert audit(parts)["reasons"] == ["equal_timestamp_order_unproven"]


def test_outcome_only_event_without_versioned_record_is_not_mutation(tmp_path):
    parts = fixture(tmp_path); parent = parts[0]
    path = Path(parent["evidence"]["ledger_path"])
    path.write_text(json.dumps({"event": "outcome", "card_id": "exp-01", "at": "2026-09-05T11:30:00Z", "note": "displayed outcome only"}) + "\n")
    parent["evidence"]["ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit(parts)["status"] == "clear"
