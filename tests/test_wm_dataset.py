"""Initial recipe extraction preserves information and whole-run boundaries."""

import copy
import json

import pytest

from tools.outcome_prediction.wm_dataset import (
    classify_role,
    extract_recorder_dataset,
    stratified_run_split,
)

BASE = "fixture/base-model"
MANIFEST = "explicit_manifest.json"
SCRIPT = "/home/ben/task/train.py"
CONFIG = "/home/ben/task/generation_config.json"


def at(second):
    return f"2026-09-04T00:00:{second:02d}Z"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_card(card_id="exp-01", family="sft"):
    return {
        "card_id": card_id,
        "problem": {"statement": "Prior failure motivating a new experiment"},
        "hypothesis": {"claim": "Changing the recipe may improve accuracy"},
        "setup": {
            "base_model": BASE,
            "method": {"family": family, "learning_rate": 0.0001},
            "command": {"script": SCRIPT, "argv": ["python", SCRIPT]},
            "data": [{"source": "training-data", "n": 100}],
            "parent_checkpoint": {"origin": "base_model"},
            "output_dir": "/home/ben/task/checkpoint",
        },
        "evaluation": {"protocol": {"n": 30, "dev_set": "local-dev"}},
        "result": {"measurements": [], "execution": None, "output_checkpoint": None},
        "conclusion": {"summary": None},
    }


def write_manifest(root, cells=("cell-01",), benchmark="aime"):
    write_json(
        root / MANIFEST,
        [
            {
                "cell_id": cell,
                "scientist_model": "scientist-a",
                "benchmark": benchmark,
                "base_model": BASE,
            }
            for cell in cells
        ],
    )


def submit(root, card, second, *, number=1, stage="plan", cell="cell-01"):
    path = root / "cells" / cell / "wm/cards" / card["card_id"] / f"record-{number:02d}.json"
    write_json(path, {"at": at(second), "card": card})
    ledger = root / "cells" / cell / "wm/records.jsonl"
    entry = {
        "at": at(second),
        "event": "submit",
        "card_id": card["card_id"],
        "record_n": number,
        "stage": stage,
        "path": str(path),
    }
    old = ledger.read_text() if ledger.exists() else ""
    ledger.write_text(old + json.dumps(entry) + "\n")
    return path


def official(root, score, *, card_id="exp-01", cell="cell-01"):
    path = root / "cells" / cell / "wm_metrics" / (card_id + ".json")
    write_json(path, {"accuracy": score, "stderr": 0.01})


def extract(root, **overrides):
    args = {
        "raw_root": root,
        "manifest_name": MANIFEST,
        "source_revision": "explicit-source-revision",
        "benchmark": "aime",
        "base_model": BASE,
        "evaluation_n": 30,
        "recover_code": False,
    }
    return extract_recorder_dataset(**(args | overrides))


def test_first_plan_is_structurally_separate_from_labels_and_final_content(tmp_path):
    write_manifest(tmp_path)
    initial = make_card()
    submit(tmp_path, initial, 10)
    official(tmp_path, 0)
    final = copy.deepcopy(initial)
    final["result"] = {"measurements": [{"value": 0.9}], "output_checkpoint": "future"}
    final["conclusion"] = {"summary": "OUTCOME_SENTINEL"}
    final["setup"]["method"]["learning_rate"] = 0.9
    final_path = tmp_path / "cells/cell-01/wm/cards/exp-01/card.json"
    write_json(final_path, final)
    rows, audit = extract(tmp_path)
    row = rows[0]
    assert row["model_input"]["plan"] == {
        key: initial[key] for key in ("problem", "hypothesis", "setup", "evaluation")
    }
    assert "OUTCOME_SENTINEL" not in json.dumps(row["model_input"])
    assert row["audit"]["eligible"] is True
    assert row["audit"]["has_official_label"] is True
    assert row["label"]["accuracy"] == row["label"]["correct_count"] == 0
    assert row["audit"]["first_final_setup_changed_fields"] == ["method"]
    assert row["audit"]["output_artifact_path_mismatch"] is True
    assert row["audit"]["label_fidelity_review_required"] is True
    assert row["audit"]["label_fidelity_status"] == "provisional_unadjudicated"
    assert audit["counts"]["eligible_labeled"] == 1
    final["setup"]["method"]["learning_rate"] = 0.5
    final["result"]["measurements"][0]["value"] = 0.5
    write_json(final_path, final)
    official(tmp_path, 0.5)
    changed, _ = extract(tmp_path)
    assert changed[0]["model_input"] == row["model_input"]
    assert changed[0]["label"] != row["label"]


@pytest.mark.parametrize(
    ("stage", "result", "reason"),
    [
        ("closed", {}, "first_submission_not_plan"),
        ("plan", {"measurements": [{"metric": "accuracy", "value": 0}]}, "first_result_not_empty"),
        ("plan", {"execution": {"exit_code": 0}}, "first_result_not_empty"),
        ("plan", {"output_checkpoint": "/checkpoint"}, "first_result_not_empty"),
        ("plan", {"wall_h": 0}, "first_result_not_empty"),
    ],
)
def test_rejected_first_records_are_preserved(tmp_path, stage, result, reason):
    write_manifest(tmp_path)
    card = make_card()
    card["result"] = result
    submit(tmp_path, card, 10, stage=stage)
    rows, _ = extract(tmp_path)
    assert len(rows) == 1
    assert rows[0]["audit"]["eligible"] is False
    assert reason in rows[0]["audit"]["reasons"]
    assert "result" not in rows[0]["model_input"]["plan"]


def test_earliest_timestamp_not_filename_determines_plan(tmp_path):
    write_manifest(tmp_path)
    early = make_card()
    late = copy.deepcopy(early)
    late["setup"]["method"]["learning_rate"] = 999
    submit(tmp_path, late, 20, number=1, stage="closed")
    submit(tmp_path, early, 10, number=2, stage="plan")
    rows, _ = extract(tmp_path)
    assert rows[0]["first_submitted_at"] == at(10)
    assert rows[0]["first_stage"] == "plan"
    assert rows[0]["model_input"]["plan"]["setup"] == early["setup"]
    assert rows[0]["provenance"]["first_record_path"].endswith("record-02.json")


def test_final_only_and_metric_only_rows_never_fall_back_to_final_plan(tmp_path):
    write_manifest(tmp_path)
    write_json(tmp_path / "cells/cell-01/wm/cards/exp-01/card.json", make_card())
    official(tmp_path, 0)
    official(tmp_path, 0.1, card_id="exp-02")
    rows, audit = extract(tmp_path)
    assert len(rows) == audit["counts"]["missing_versioned_registration"] == 2
    for row in rows:
        assert row["first_submitted_at"] is None
        assert row["model_input"]["plan"] == {}
        assert row["audit"]["eligible"] is False
        assert row["audit"]["has_official_label"] is True


def test_histories_are_strict_pre_cutoff_raw_records_without_scope_collapse(tmp_path):
    write_manifest(tmp_path, cells=("cell-01", "cell-02"))
    parent = make_card("exp-01")
    submit(tmp_path, parent, 1)
    parent["result"]["measurements"] = [
        {"name": "accuracy", "value": 0, "n": 30, "scope": "temperature=.6"},
        {"metric": "mean_pass@1", "value": 0.2, "n": 150, "scope": "temperature=.7"},
    ]
    parent["evaluation"]["protocol"]["temperature"] = 0.7
    submit(tmp_path, parent, 5, number=2, stage="running")
    parent["result"]["measurements"] = [{"value": 0.9, "scope": "SAME_SECOND"}]
    submit(tmp_path, parent, 10, number=3, stage="closed")
    parent["conclusion"]["summary"] = "FUTURE_CLOSURE"
    submit(tmp_path, parent, 20, number=4, stage="closed")
    candidate = make_card("exp-02")
    submit(tmp_path, candidate, 10)
    submit(tmp_path, candidate, 25, number=2, stage="closed")
    submit(tmp_path, make_card(), 2, cell="cell-02")
    official(tmp_path, 0.8)
    rows, _ = extract(tmp_path)
    row = next(row for row in rows if row["example_id"] == "cell-01/exp-02")
    history = row["prior_observations"]
    assert [item["at"] for item in history] == [at(1), at(5)]
    assert [item["n"] for item in history[1]["measurements"]] == [30, 150]
    assert history[1]["measurements"][0]["value"] == 0
    assert history[1]["evaluation"]["protocol"]["temperature"] == 0.7
    assert history[1]["record"]["card"]["result"]["measurements"] == history[1]["measurements"]
    text = json.dumps(history)
    for forbidden in ("FUTURE_CLOSURE", "SAME_SECOND", "cell-02", '"accuracy": 0.8'):
        assert forbidden not in text
    assert "prior_observations" not in row["model_input"]


def test_task_parameters_are_explicit_and_not_gsm_defaults(tmp_path):
    write_manifest(tmp_path)
    submit(tmp_path, make_card(), 10)
    official(tmp_path, 1 / 30)
    rows, audit = extract(tmp_path)
    task = {"benchmark": "aime", "base_model": BASE, "evaluation_n": 30}
    assert audit["task"] == rows[0]["model_input"]["task"] == task
    assert rows[0]["label"]["correct_count"] == 1
    assert rows[0]["provenance"]["source_revision"] == "explicit-source-revision"
    with pytest.raises(ValueError, match="Manifest benchmark"):
        extract(tmp_path, benchmark="gsm8k")
    with pytest.raises(ValueError, match="Manifest base_model"):
        extract(tmp_path, base_model="different-base")
    with pytest.raises(ValueError, match="evaluation_n"):
        extract(tmp_path, evaluation_n=0)
    with pytest.raises(TypeError):
        extract_recorder_dataset(raw_root=tmp_path)


@pytest.mark.parametrize(
    ("family", "command", "expected"),
    [
        ("sft", "evaluate.py", "training"),
        ("weight_averaging", "unknown.py", "merge"),
        ("decode-config", "unknown.py", "decoding"),
        ("decoding", "unknown.py", "decoding"),
        ("checkpoint_selection", "unknown.py", "evaluation"),
        ("eval-only", "unknown.py", "evaluation"),
        ("other", "train_sft.py", "training"),
        ("other", "merge.py", "merge"),
        ("other", "evaluate.py", "evaluation"),
        ("other", "unknown.py --temperature .7", "decoding"),
        (None, "unknown.py", "unknown"),
    ],
)
def test_role_uses_declared_method_and_command_only(family, command, expected):
    card = make_card(family=family)
    card["setup"]["command"] = {"script": command}
    assert classify_role(card) == expected
    card["result"] = {"notes": "merge decoding sft train.py"}
    assert classify_role(card) == expected


def test_missing_code_is_unknown_not_an_execution_failure(tmp_path):
    write_manifest(tmp_path)
    card = make_card(family="decoding")
    card["setup"]["command"] = {}
    submit(tmp_path, card, 10)
    rows, _ = extract(tmp_path, recover_code=True)
    row = rows[0]
    assert row["audit"]["eligible"] is True
    assert row["role"] == "decoding"
    assert row["model_input"]["code"][0]["status"] == "unavailable"
    assert row["model_input"]["code"][0]["content"] is None
    assert row["label"]["accuracy"] is None


def trace_event(second, content, result=None):
    record = {"timestamp": at(second), "message": {"content": content}}
    if result is not None:
        record["tool_use_result"] = result
    return f"[{at(second)}] {json.dumps(record)}\n"


def trace_write(second, path, content, ident):
    request = trace_event(
        second,
        [
            {
                "type": "tool_use",
                "id": ident,
                "name": "Write",
                "input": {"file_path": path, "content": content},
            }
        ],
    )
    response = trace_event(
        second + 1,
        [{"type": "tool_result", "tool_use_id": ident, "is_error": False}],
        {"filePath": path, "content": content, "userModified": False},
    )
    return request + response


def test_code_and_declared_config_recover_only_pre_registration_versions(tmp_path):
    write_manifest(tmp_path)
    card = make_card()
    card["setup"]["command"]["argv"] += ["--generation-config", CONFIG]
    submit(tmp_path, card, 10)
    trace = trace_write(1, SCRIPT, "lr = 0.001\n", "train")
    trace += trace_write(3, CONFIG, '{"temperature": 0.7}\n', "config")
    trace += trace_write(11, SCRIPT, "FUTURE_OUTCOME = 0.9\n", "future")
    (tmp_path / "cells/cell-01/solve_out_sanitized.txt").write_text(trace)
    snapshot = tmp_path / "cells/cell-01/wm/cards/exp-01/scripts/train.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("FINAL_SNAPSHOT = 0.5\n")
    rows, _ = extract(tmp_path, recover_code=True)
    features = rows[0]["model_input"]["code"]
    assert [item["role"] for item in features] == ["training", "evaluation_config_0"]
    assert features[0]["content"] == "lr = 0.001\n"
    assert features[1]["content"] == '{"temperature": 0.7}\n'
    assert all(item["status"] == "reconstructed" for item in features)
    assert all(set(item) == {"role", "script_path", "status", "content"} for item in features)
    assert all("content" not in item for item in rows[0]["audit"]["code_provenance"])
    assert "FUTURE_OUTCOME" not in json.dumps(features)
    assert "FINAL_SNAPSHOT" not in json.dumps(features)


def test_known_fidelity_exclusion_is_benchmark_specific(tmp_path):
    write_manifest(tmp_path, cells=("r0-02",), benchmark="gsm8k")
    submit(tmp_path, make_card("exp-03"), 10, cell="r0-02")
    rows, _ = extract(tmp_path, benchmark="gsm8k", evaluation_n=1319)
    assert "known_fidelity_exclusion" in rows[0]["audit"]["reasons"]
    assert rows[0]["audit"]["known_fidelity_reason"]
    write_manifest(tmp_path, cells=("r0-02",), benchmark="aime")
    rows, _ = extract(tmp_path)
    assert rows[0]["audit"]["eligible"] is True


def test_malformed_version_is_not_silently_skipped_as_a_safe_plan(tmp_path):
    write_manifest(tmp_path)
    submit(tmp_path, make_card(), 10, number=2)
    write_json(tmp_path / "cells/cell-01/wm/cards/exp-01/record-01.json", {"at": 123})
    rows, audit = extract(tmp_path)
    assert rows[0]["audit"]["eligible"] is False
    assert "invalid_versioned_record" in rows[0]["audit"]["reasons"]
    assert len(rows[0]["audit"]["record_errors"]) == len(audit["record_errors"]) == 1


def split_rows():
    return [
        {
            "cell_id": f"{benchmark}-{scientist}-{run}",
            "example_id": f"{benchmark}-{scientist}-{run}/exp-{recipe}",
            "benchmark": benchmark,
            "scientist_model": scientist,
            "label": {"accuracy": recipe / 10},
            "eligible": recipe == 1,
        }
        for benchmark in ("gsm8k", "aime")
        for scientist in ("one", "two")
        for run in range(4)
        for recipe in (1, 2)
    ]


def test_whole_run_split_is_stratified_order_independent_and_label_blind():
    rows = split_rows()
    split = stratified_run_split(rows, test_count=1, seed="fixed-seed")
    assert len(split["test_cell_ids"]) == 4
    assert len(split["train_cell_ids"]) == 12
    assert len(split["test_example_ids"]) == 8
    assert all(len(stratum["test_cell_ids"]) == 1 for stratum in split["strata"])
    assert set(split["test_cell_ids"]).isdisjoint(split["train_cell_ids"])
    assert set(split["test_example_ids"]).isdisjoint(split["train_example_ids"])
    for row in rows:
        partition = split["cell_partition"][row["cell_id"]]
        assert row["example_id"] in split[partition + "_example_ids"]
        row["label"] = {"accuracy": 999}
        row["eligible"] = not row["eligible"]
        row["model_input"] = {"future": "changed"}
    assert stratified_run_split(list(reversed(rows)), test_count=1, seed="fixed-seed") == split


def test_split_guardrails_and_zero_holdout():
    rows = split_rows()
    split = stratified_run_split(rows, test_count=0, seed=0)
    assert split["test_cell_ids"] == []
    assert len(split["train_example_ids"]) == len(rows)
    with pytest.raises(ValueError, match="unique"):
        stratified_run_split(rows + rows[:1], test_count=1, seed=0)
    rows[1]["scientist_model"] = "different-scientist"
    with pytest.raises(ValueError, match="multiple"):
        stratified_run_split(rows, test_count=1, seed=0)
    with pytest.raises(ValueError, match="no training"):
        stratified_run_split(split_rows(), test_count=4, seed=0)
    with pytest.raises(ValueError, match="test_count"):
        stratified_run_split(split_rows(), test_count=True, seed=0)
    with pytest.raises(TypeError, match="seed"):
        stratified_run_split(split_rows(), test_count=1, seed=None)
