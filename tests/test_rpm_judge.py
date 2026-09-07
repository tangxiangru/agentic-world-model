"""Offline guards for RPM prompts, temporal evidence, and fresh judge sessions."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import rpm_judge as judge


def example(card_id, *, cell_id="cell-test", at="2026-09-04T05:00:00Z", score=0.93):
    recipe = {
        "base_model": "google/gemma-3-4b-pt",
        "method": "sft",
        "hyperparameters": {"learning_rate": 2e-5},
        "data": [{"source_ids": ["openai/gsm8k"], "n_examples": 7473}],
    }
    return {
        "example_id": f"{cell_id}/{card_id}",
        "cell_id": cell_id,
        "card_id": card_id,
        "first_submitted_at": at,
        "eligible": True,
        "prospective": True,
        "lineage_complete": True,
        "lineage": [{"card_id": card_id, "recipe": recipe}],
        "recipe": recipe,
        "recipe_text": json.dumps(recipe, sort_keys=True),
        "lineage_text": "step 1: " + json.dumps(recipe, sort_keys=True),
        "parent_ids": [],
        "data_dependency_ids": [],
        "y": score,
        "local_accuracy": 0.91,
        "parent_accuracy": 0.92,
        "scientist_model": "SECRET_SCIENTIST_ID",
        "plan_text": "POSTEXECUTION_RESULT: accuracy=0.99",
        "lineage_plan_text": "POSTEXECUTION_RESULT: adopted winner",
        "source_card": f"cells/{cell_id}/wm/cards/{card_id}/card.json",
    }


def write_record(root, row, number, at, score):
    directory = root / "cells" / row["cell_id"] / "wm/cards" / row["card_id"]
    directory.mkdir(parents=True, exist_ok=True)
    card = {
        "setup": {
            "base_model": "google/gemma-3-4b-pt",
            "method": {"family": "sft", "hyperparams": {"lr": 2e-5}},
            "parent_checkpoint": {"origin": "base_model"},
        },
        "result": {
            "execution": "completed",
            "measurements": [{"name": "accuracy", "value": score, "n": 150}],
        },
    }
    path = directory / f"record-{number:02d}.json"
    path.write_text(json.dumps({"at": at, "event": "submit", "card": card}))


def test_candidate_uses_only_canonical_dependency_sequence():
    row = example("exp-02")
    parent = example("exp-01")["recipe"]
    parent["hyperparameters"]["learning_rate"] = 3e-5
    row["lineage"].insert(0, {"card_id": "exp-01", "recipe": parent})
    payload = judge.candidate_payload(row)
    assert payload == {"recipe_sequence": [entry["recipe"] for entry in row["lineage"]]}
    encoded = json.dumps(payload)
    for forbidden in ("POSTEXECUTION", "SECRET_", "cell-test", "exp-01", "exp-02", "accuracy"):
        assert forbidden not in encoded


def test_candidate_payload_is_invariant_to_hidden_labels_and_results():
    row = example("exp-02")
    baseline = judge.candidate_payload(row)
    changed = copy.deepcopy(row)
    changed.update(y=0.03, local_accuracy=0.04, parent_accuracy=0.05)
    changed.update(plan_text="failed", lineage_plan_text="rejected", scientist_model="other")
    changed["result"] = {"execution": "failed", "accuracy": 0.03}
    changed["conclusion"] = {"decision": "reject"}
    assert judge.candidate_payload(changed) == baseline


def test_history_scores_precede_earlier_candidate_and_exclude_both_dependencies(tmp_path):
    a = example("exp-05")
    b = example("exp-06", at="2026-09-04T06:00:00Z")
    prior = example("exp-01", at="2026-09-04T00:00:00Z")
    late = example("exp-02", at="2026-09-04T01:00:00Z")
    dependent_a = example("exp-03", at="2026-09-04T02:00:00Z")
    dependent_b = example("exp-04", at="2026-09-04T03:00:00Z")
    dependent_a["lineage"].insert(0, copy.deepcopy(a["lineage"][0]))
    dependent_b["lineage"].insert(0, copy.deepcopy(b["lineage"][0]))
    other_cell = example("exp-01", cell_id="cell-other", at="2026-09-04T00:00:00Z")
    write_record(tmp_path, prior, 1, "2026-09-04T01:00:00Z", 0.25)
    write_record(tmp_path, prior, 2, "2026-09-04T05:00:00Z", 0.75)
    write_record(tmp_path, prior, 3, "2026-09-04T05:30:00Z", 0.85)
    write_record(tmp_path, late, 1, "2026-09-04T05:30:00Z", 0.95)
    write_record(tmp_path, dependent_a, 1, "2026-09-04T03:00:00Z", 0.65)
    write_record(tmp_path, dependent_b, 1, "2026-09-04T04:00:00Z", 0.66)
    write_record(tmp_path, other_cell, 1, "2026-09-04T01:00:00Z", 0.67)
    write_record(tmp_path, a, 1, "2026-09-04T04:30:00Z", 0.68)
    write_record(tmp_path, b, 1, "2026-09-04T04:30:00Z", 0.69)
    rows = [prior, late, dependent_a, dependent_b, other_cell, a, b]
    history = judge.observable_history(rows, tmp_path, a, b)
    assert len(history) == 1
    assert history[0]["observed_accuracy"] == 0.25
    assert "SECRET_" not in json.dumps(history)


def test_history_does_not_substitute_retrospective_official_accuracy(tmp_path):
    a = example("exp-02")
    b = example("exp-03", at="2026-09-04T06:00:00Z")
    prior = example("exp-01", at="2026-09-04T00:00:00Z", score=0.99)
    assert judge.observable_history([prior, a, b], tmp_path, a, b) == []


def test_history_rejects_ambiguous_same_size_measurements(tmp_path):
    a = example("exp-02")
    b = example("exp-03", at="2026-09-04T06:00:00Z")
    prior = example("exp-01", at="2026-09-04T00:00:00Z")
    write_record(tmp_path, prior, 1, "2026-09-04T01:00:00Z", 0.25)
    path = tmp_path / "cells/cell-test/wm/cards/exp-01/record-01.json"
    record = json.loads(path.read_text())
    record["card"]["result"]["measurements"].append(
        {"name": "accuracy_sampling", "value": 0.75, "n": 150}
    )
    path.write_text(json.dumps(record))
    assert judge.observable_history([prior, a, b], tmp_path, a, b) == []


def test_arms_share_candidate_and_current_run_inputs_without_hidden_outcomes():
    a, b = example("exp-01"), example("exp-02")
    b["recipe"]["hyperparameters"]["learning_rate"] = 5e-5
    train = [example("exp-01", cell_id="cell-train", score=0.33)]
    history = [{"recipe_sequence": [], "observed_accuracy": 0.4, "metric_scope": "local"}]
    within = judge.build_payload(a, b, train, history, "within_run")
    cross = judge.build_payload(a, b, train, history, "cross_run")
    bank = cross.pop("historical_training_experiments")
    groups = cross.pop("historical_training_run_groups")
    assert cross == within
    assert groups["groups"] == [[0]]
    assert "cell-train" not in json.dumps(groups)
    assert bank == [{**judge.candidate_payload(train[0]), "official_accuracy": 0.33}]
    assert "SECRET_" not in json.dumps(within)
    assert "POSTEXECUTION" not in json.dumps(within)
    assert "official_accuracy" not in json.dumps(within)
    a.update(y=0.01, local_accuracy=0.02, parent_accuracy=0.03)
    b.update(y=0.99, local_accuracy=0.98, parent_accuracy=0.97)
    assert judge.build_payload(a, b, train, history, "within_run") == within


def test_training_bank_rejects_either_candidate_cell():
    a = example("exp-01", cell_id="cell-a")
    b = example("exp-02", cell_id="cell-b")
    for cell_id in ("cell-a", "cell-b"):
        with pytest.raises(ValueError):
            judge.build_payload(a, b, [example("exp-99", cell_id=cell_id)], [], "cross_run")


def test_swapping_changes_only_candidate_assignment():
    a, b = example("exp-01"), example("exp-02")
    b["recipe"]["hyperparameters"]["learning_rate"] = 5e-5
    original = judge.build_payload(a, b, [], [], "within_run", swap=False)
    swapped = judge.build_payload(a, b, [], [], "within_run", swap=True)
    assert original["candidate_A"] == swapped["candidate_B"]
    assert original["candidate_B"] == swapped["candidate_A"]
    assert original["observed_current_run_history"] == swapped["observed_current_run_history"]


@pytest.mark.parametrize(
    "choice,p,swap,expected_p,expected_choice",
    [
        ("A", 0.8, False, 0.8, True),
        ("A", 0.8, True, 0.2, False),
        ("B", 0.2, False, 0.2, False),
        ("B", 0.2, True, 0.8, True),
    ],
)
def test_judge_predictions_return_to_canonical_orientation(
    choice, p, swap, expected_p, expected_choice
):
    raw = {"subtype": "success", "result": json.dumps({"choice": choice, "p_A": p})}
    decoded = judge.decode_prediction(raw, swapped=swap)
    assert decoded["p_a"] == pytest.approx(expected_p)
    assert decoded["choice_a"] is expected_choice


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "A", "p_A": 0.2},
        {"choice": "B", "p_A": 0.8},
        {"choice": "C", "p_A": 0.8},
        {"choice": "A", "p_A": float("nan")},
        {"choice": "A", "p_A": 1.1},
    ],
)
def test_invalid_predictions_are_not_reinterpreted_as_valid(answer):
    with pytest.raises(ValueError):
        judge.decode_prediction({"subtype": "success", "result": json.dumps(answer)})


def test_cli_is_fresh_has_no_tools_and_receives_only_payload(monkeypatch):
    calls = []
    payload = {"candidate_A": {"recipe_sequence": []}, "candidate_B": {"recipe_sequence": []}}

    def fake_run(cmd, **kwargs):
        working = Path(kwargs["cwd"])
        assert working.is_dir()
        assert list(working.iterdir()) == []
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"subtype": "success"}), stderr="")

    monkeypatch.setattr(judge.subprocess, "run", fake_run)
    judge.run_judge(payload, "fixture-model", 0.75, 123)
    judge.run_judge(payload, "fixture-model", 0.75, 123)
    assert len(calls) == 2
    assert calls[0][1]["cwd"] != calls[1][1]["cwd"]
    for cmd, kwargs in calls:
        for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            assert flag in cmd
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--system-prompt") + 1] == judge.SYSTEM
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert cmd[cmd.index("--model") + 1] == "fixture-model"
        assert "--continue" not in cmd and "--resume" not in cmd
        assert "--append-system-prompt" not in cmd
        assert json.loads(kwargs["input"]) == payload
        assert kwargs["timeout"] == 123
        assert kwargs["capture_output"] is True
        assert not Path(kwargs["cwd"]).exists()


def test_invalid_billable_call_is_saved_once_and_not_retried_on_resume(tmp_path, monkeypatch):
    payload = {"task": "fixture"}
    input_path = tmp_path / "inputs/within_run/pair-fixture.json"
    judge.write_json(input_path, payload)
    job = {
        "id": "pair-fixture",
        "arm": "within_run",
        "input": str(input_path),
        "input_sha256": judge.digest(payload),
        "swapped": False,
    }
    judge.write_json(tmp_path / "jobs.json", [job])
    judge.write_json(
        tmp_path / "protocol.json", {"system_prompt_sha256": judge.digest(judge.SYSTEM)}
    )
    (tmp_path / "hidden_labels.json").write_text("NOT JSON; THE JUDGE MUST NEVER READ THIS")
    calls = []

    def fake_judge(*args):
        calls.append(args)
        return {"subtype": "success", "result": "invalid JSON", "total_cost_usd": 0.123}

    monkeypatch.setattr(judge, "run_judge", fake_judge)
    args = SimpleNamespace(
        output_dir=tmp_path,
        arms=["within_run"],
        limit=None,
        model="fixture",
        call_budget=1.0,
        timeout=123,
        workers=1,
        total_budget=5.0,
    )
    judge.execute(args)
    output = tmp_path / "outputs/within_run/pair-fixture.json"
    before = output.read_bytes()
    stored = json.loads(before)
    assert not stored["valid"]
    assert stored["cost_usd"] == 0.123
    assert stored["raw_response"]["result"] == "invalid JSON"
    judge.execute(args)
    assert len(calls) == 1
    assert output.read_bytes() == before


@pytest.mark.parametrize("failure_mode", ["timeout", "json_without_cost"])
def test_unknown_failed_call_cost_stays_reserved_across_resume(tmp_path, monkeypatch, failure_mode):
    jobs = []
    for index in range(2):
        payload = {"task": f"fixture-{index}"}
        path = tmp_path / f"inputs/within_run/pair-{index}.json"
        judge.write_json(path, payload)
        jobs.append(
            {
                "id": f"pair-{index}",
                "arm": "within_run",
                "input": str(path),
                "input_sha256": judge.digest(payload),
                "swapped": False,
            }
        )
    judge.write_json(tmp_path / "jobs.json", jobs)
    judge.write_json(
        tmp_path / "protocol.json", {"system_prompt_sha256": judge.digest(judge.SYSTEM)}
    )
    calls = []

    def fake_judge(*args):
        calls.append(args)
        if failure_mode == "timeout":
            raise subprocess.TimeoutExpired(cmd="fixture", timeout=123)
        return {"subtype": "error_max_budget_usd", "is_error": True, "result": ""}

    monkeypatch.setattr(judge, "run_judge", fake_judge)
    args = SimpleNamespace(
        output_dir=tmp_path,
        arms=["within_run"],
        limit=None,
        model="fixture",
        call_budget=1.0,
        timeout=123,
        workers=1,
        total_budget=1.0,
    )
    judge.execute(args)
    assert len(calls) == 1
    judge.execute(args)
    assert len(calls) == 1
    output_paths = list((tmp_path / "outputs/within_run").glob("*.json"))
    assert len(output_paths) == 1
    assert json.loads(output_paths[0].read_text())["reserved_cost_usd"] == 1.0


def test_changed_system_prompt_fails_before_any_model_call(tmp_path, monkeypatch):
    judge.write_json(tmp_path / "protocol.json", {"system_prompt_sha256": "wrong-hash"})
    judge.write_json(tmp_path / "jobs.json", [])
    monkeypatch.setattr(
        judge, "run_judge", lambda *args: pytest.fail("Must fail before calling model")
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        arms=["within_run"],
        limit=None,
        model="fixture",
        call_budget=1.0,
        timeout=123,
        workers=1,
        total_budget=1.0,
    )
    with pytest.raises(ValueError, match="[Ss]ystem prompt"):
        judge.execute(args)


def test_changed_payload_fails_before_any_model_call(tmp_path, monkeypatch):
    original = {"task": "original"}
    path = tmp_path / "inputs/within_run/pair-fixture.json"
    judge.write_json(path, {"task": "edited after freezing"})
    job = {
        "id": "pair-fixture",
        "arm": "within_run",
        "input": str(path),
        "input_sha256": judge.digest(original),
        "swapped": False,
    }
    judge.write_json(tmp_path / "jobs.json", [job])
    judge.write_json(
        tmp_path / "protocol.json", {"system_prompt_sha256": judge.digest(judge.SYSTEM)}
    )
    monkeypatch.setattr(
        judge, "run_judge", lambda *args: pytest.fail("Must fail before calling model")
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        arms=["within_run"],
        limit=None,
        model="fixture",
        call_budget=1.0,
        timeout=123,
        workers=1,
        total_budget=1.0,
    )
    with pytest.raises(ValueError, match="Input changed"):
        judge.execute(args)


def test_prepared_training_banks_match_entire_disjoint_outer_fold(tmp_path):
    rows = []
    for cell in range(8):
        for number in range(2):
            row = example(f"exp-{number + 1:02d}", cell_id=f"cell-{cell}", score=0.2 + number * 0.1)
            row["recipe"]["hyperparameters"]["learning_rate"] = (cell * 2 + number + 1) * 1e-6
            rows.append(row)
    examples = tmp_path / "examples.jsonl"
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "prepared"
    args = SimpleNamespace(examples=examples, raw_root=tmp_path / "raw", output_dir=output, seed=42)
    judge.prepare(args)
    folds = json.loads((output / "folds.json").read_text())
    labels = {row["id"]: row for row in json.loads((output / "hidden_labels.json").read_text())}
    jobs = json.loads((output / "jobs.json").read_text())
    by_id = {row["example_id"]: row for row in rows}
    assert len(labels) == 8
    for fold in folds:
        train_cells = {by_id[eid]["cell_id"] for eid in fold["train_ids"]}
        test_cells = {by_id[eid]["cell_id"] for eid in fold["test_ids"]}
        assert train_cells.isdisjoint(test_cells)
    for job in jobs:
        if job["arm"] != "cross_run":
            continue
        payload = json.loads(Path(job["input"]).read_text())
        fold = folds[labels[job["id"]]["fold"]]
        expected = [
            {**judge.candidate_payload(by_id[eid]), "official_accuracy": by_id[eid]["y"]}
            for eid in fold["train_ids"]
        ]
        encode = lambda value: json.dumps(value, sort_keys=True)
        assert sorted(map(encode, payload["historical_training_experiments"])) == sorted(
            map(encode, expected)
        )
        assert "cell-" not in encode(payload)
        assert "SECRET_SCIENTIST_ID" not in encode(payload)


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("metric", 0.42, 0.42),
        ("name", 0.42, 0.42),
        ("metric", 42, 0.42),
        ("metric", True, None),
        ("metric", float("nan"), None),
    ],
)
def test_history_score_handles_recorder_metric_key_and_units(key, value, expected):
    card = {"result": {"measurements": [{key: "gsm8k_accuracy", "n": 150, "value": value}]}}
    assert judge.history_score(card) == expected


def test_training_run_groups_partition_bank_without_exposing_identity():
    a, b = example("exp-01"), example("exp-02")
    train = [
        example("exp-01", cell_id="secret-run-a"),
        example("exp-01", cell_id="secret-run-b"),
        example("exp-02", cell_id="secret-run-a"),
        example("exp-02", cell_id="secret-run-b"),
        example("exp-01", cell_id="secret-run-c"),
    ]
    payload = judge.build_payload(a, b, train, [], "cross_run")
    groups = payload["historical_training_run_groups"]["groups"]
    assert sorted(map(tuple, groups)) == [(0, 2), (1, 3), (4,)]
    assert sorted(index for group in groups for index in group) == list(range(len(train)))
    assert "secret-run" not in json.dumps(payload)
    changed = copy.deepcopy(train)
    for row in changed:
        row["y"] = 1 - row["y"]
    assert (
        judge.build_payload(a, b, changed, [], "cross_run")["historical_training_run_groups"]
        == payload["historical_training_run_groups"]
    )


def expand_compact_payload(payload):
    restored = copy.deepcopy(payload)
    catalog = restored.pop("historical_recipe_catalog")
    restored.pop("historical_recipe_encoding")
    for experiment in restored["historical_training_experiments"]:
        experiment["recipe_sequence"] = [catalog[key] for key in experiment["recipe_sequence"]]
    return restored


def test_compact_payload_roundtrip_preserves_all_examples_steps_and_metadata():
    first = {
        "method": "sft",
        "hyperparameters": {"lr": 0.00002, "seed": 0},
        "data": [{"source_ids": ["openai/gsm8k"], "missing": None}],
        "enabled": False,
    }
    reordered = dict(reversed(list(copy.deepcopy(first).items())))
    second = {"method": "merge", "merge": {"weights": [1.5, -0.5]}, "data": []}
    payload = {
        "task": "fixture",
        "candidate_A": {"recipe_sequence": [first]},
        "candidate_B": {"recipe_sequence": [second]},
        "observed_current_run_history": [{"recipe_sequence": [first], "observed_accuracy": 0.25}],
        "historical_training_run_groups": {
            "description": "anonymous",
            "groups": [[0, 2], [1], [3]],
        },
        "historical_training_experiments": [
            {"recipe_sequence": [first, second, first], "official_accuracy": 0.3},
            {"recipe_sequence": [reordered], "official_accuracy": 0.4},
            {"recipe_sequence": [second, second], "official_accuracy": 0.5},
            {"recipe_sequence": [], "official_accuracy": 0.0},
        ],
    }
    before = copy.deepcopy(payload)
    compact = judge.compact_payload(payload)
    assert payload == before
    assert len(compact["historical_recipe_catalog"]) == 2
    assert len(compact["historical_training_experiments"]) == 4
    assert [len(row["recipe_sequence"]) for row in compact["historical_training_experiments"]] == [
        3,
        1,
        2,
        0,
    ]
    for key in (
        "candidate_A",
        "candidate_B",
        "observed_current_run_history",
        "historical_training_run_groups",
    ):
        assert compact[key] == payload[key]
    serialized = json.loads(json.dumps(compact))
    assert json.dumps(expand_compact_payload(serialized), sort_keys=True) == json.dumps(
        payload, sort_keys=True
    )


def test_compact_payload_handles_empty_bank_and_within_run_payload():
    within = {"candidate_A": {"recipe_sequence": []}, "observed_current_run_history": []}
    assert judge.compact_payload(within) == within
    empty = {
        **within,
        "historical_training_experiments": [],
        "historical_training_run_groups": {"groups": []},
    }
    assert expand_compact_payload(judge.compact_payload(empty)) == empty


def resume_fixture(tmp_path, *, previous_model="fixture"):
    jobs = []
    for arm in ("within_run", "cross_run"):
        payload = {"task": arm}
        path = tmp_path / f"inputs/{arm}/pair-fixture.json"
        judge.write_json(path, payload)
        jobs.append(
            {
                "id": "pair-fixture",
                "arm": arm,
                "input": str(path),
                "input_sha256": judge.digest(payload),
                "swapped": False,
            }
        )
    judge.write_json(tmp_path / "jobs.json", jobs)
    judge.write_json(
        tmp_path / "protocol.json", {"system_prompt_sha256": judge.digest(judge.SYSTEM)}
    )
    judge.write_json(
        tmp_path / "outputs/within_run/pair-fixture.json",
        {"model_requested": previous_model, "cost_usd": 1.0, "valid": False},
    )
    return SimpleNamespace(
        output_dir=tmp_path,
        arms=["cross_run"],
        limit=None,
        model="fixture",
        call_budget=1.0,
        timeout=123,
        workers=1,
        total_budget=1.0,
    )


def test_resume_budget_includes_unselected_arms(tmp_path, monkeypatch):
    args = resume_fixture(tmp_path)
    monkeypatch.setattr(
        judge, "run_judge", lambda *args: pytest.fail("Other arm exhausted total budget")
    )
    judge.execute(args)
    assert not (tmp_path / "outputs/cross_run/pair-fixture.json").exists()


def test_resume_rejects_model_change_even_for_unselected_arm(tmp_path, monkeypatch):
    args = resume_fixture(tmp_path, previous_model="different-model")
    args.total_budget = 5.0
    monkeypatch.setattr(
        judge, "run_judge", lambda *args: pytest.fail("Mixed models must fail before call")
    )
    with pytest.raises(ValueError, match="different model"):
        judge.execute(args)
