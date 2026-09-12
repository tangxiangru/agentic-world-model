import copy
import json
import statistics

import pytest

from tools.outcome_prediction.hf_benchmark import (
    assign_groups,
    choose_representatives,
    predictor_payload,
    result_identity,
    validate_label,
)
from tools.outcome_prediction.hf_benchmark_score import predictions, score


def label():
    matrix = {str(i): [int((i + r) % 3 == 0) for r in range(10)] for i in range(30)}
    rates = [sum(v[r] for v in matrix.values()) / 30 for r in range(10)]
    return {"epochs": 10, "n_problems": 30, "per_problem": matrix,
            "per_epoch_accuracy": rates, "accuracy": statistics.mean(rates),
            "std_across_epochs": statistics.pstdev(rates)}


def test_label_recomputed_from_every_question_and_run():
    result = validate_label(label(), "aime2025")
    assert result["status"] == "complete"
    assert result["y"] == pytest.approx(1 / 3)
    corrupted = label()
    corrupted["per_problem"]["0"][0] = 0
    result = validate_label(corrupted, "aime2025")
    assert result["status"] == "invalid"
    assert "run_rates_disagree_with_question_scores" in result["errors"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d["per_problem"]["0"].pop(), "incomplete_or_nonbinary_question_runs"),
    (lambda d: d["per_epoch_accuracy"].pop(), "requires_exactly_ten_runs"),
    (lambda d: d.update(accuracy=float("nan")), "aggregate_disagrees_with_question_scores"),
    (lambda d: d["per_problem"]["0"].__setitem__(0, True), "incomplete_or_nonbinary_question_runs"),
    (lambda d: d.update(n_sample_errors=1), "sample_errors_present"),
])
def test_invalid_labels_never_get_a_target(mutation, reason):
    data = label()
    mutation(data)
    result = validate_label(data, "aime2025")
    assert result["status"] == "invalid"
    assert result["y"] is None
    assert reason in result["errors"]


def test_same_size_different_question_set_rejected():
    result = validate_label(label(), "aime2025", {"wrong" + str(i) for i in range(30)})
    assert "benchmark_question_ids_mismatch" in result["errors"]


def test_grouping_transitive_parent_and_cross_session_aliases():
    rows = [
        {"checkpoint_id": "a", "session_id": "s1", "weights_sha256": "same", "source_split": "locked_test"},
        {"checkpoint_id": "b", "session_id": "s2", "weights_sha256": "same"},
        {"checkpoint_id": "c", "session_id": "s3", "weights_sha256": "different"},
    ]
    scripts = {"c": {"session_id": "s3", "parent_checkpoint_ids": ["b"]}}
    groups = assign_groups(rows, scripts)
    assert len(groups) == 1
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["group_id"] for r in rows}) == 1


def test_real_archived_locked_split_spelling_and_location_are_preserved():
    data = {"eval_matrix_1k": {"checkpoint_id": "a", "benchmark": "aime2025",
                              "split": "locked_session_test"}}
    checkpoint, benchmark, track, split = result_identity("eval_matrix_1k/locked_test/results/a.json", data)
    assert (benchmark, track) == ("aime2025", "ptb_controlled")
    assert split == "locked_test"
    rows = [{"checkpoint_id": checkpoint, "session_id": "s1", "source_split": split}]
    assign_groups(rows, {})
    assert rows[0]["split"] == "test"
    with pytest.raises(ValueError, match="disagrees"):
        result_identity("eval_matrix_1k/results/a.json", data)


def test_alias_choice_prefers_usable_recipe_and_preserves_ten_run_label():
    common = {"track": "ptb_controlled", "benchmark": "aime2025", "weights_sha256": "w",
              "serving_sha256": "s", "exclusion_reasons": []}
    rows = [{**copy.deepcopy(common), "example_id": "a", "status": "candidate"},
            {**copy.deepcopy(common), "example_id": "b", "status": "eligible"}]
    choose_representatives(rows)
    assert rows[0]["status"] == "duplicate"
    assert rows[0]["duplicate_of"] == "b"
    assert rows[1]["status"] == "eligible"


def test_predictor_fields_do_not_copy_card_or_outcome_metadata():
    script = {"scripts": [{"path": "train.py", "role": "training", "content": "pass", "accuracy": 1}],
              "launch": {"argv": ["python", "train.py"], "dev_accuracy": 1},
              "parent_accuracy": 0.9, "provenance": {"accuracy": 1}}
    payload = predictor_payload(script, {"temperature": 0.6})
    assert set(payload) == {"experiment_scripts", "serving_config"}
    assert "accuracy" not in json.dumps(payload)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_scorer_balances_groups_and_refuses_to_reward_missing_predictions(tmp_path):
    rows = [{"example_id": str(i), "status": "eligible", "split": "test", "benchmark": "aime2025",
             "group_id": "a" if i < 2 else "b", "checkpoint_id": str(i), "weights_sha256": str(i)}
            for i in range(3)]
    write_rows(tmp_path / "audit/registry.jsonl", rows)
    write_rows(tmp_path / "labels/test.jsonl", [{"example_id": str(i), "y": 0} for i in range(3)])
    p = tmp_path / "p.jsonl"
    write_rows(p, [{"example_id": "0", "prediction": 0}, {"example_id": "1", "prediction": 0},
                   {"example_id": "2", "prediction": 1}])
    result = score(tmp_path, p)["benchmarks"]["aime2025"]
    assert result["primary_group_balanced_mae_pp"] == 50
    assert result["available_case_cell_mae_pp"] == pytest.approx(100 / 3)
    write_rows(p, [{"example_id": "0", "prediction": 0}])
    result = score(tmp_path, p)["benchmarks"]["aime2025"]
    assert result["primary_group_balanced_mae_pp"] is None
    assert result["coverage"] == pytest.approx(1 / 3)


def test_prediction_duplicates_fail_instead_of_overwriting(tmp_path):
    p = tmp_path / "p.jsonl"
    write_rows(p, [{"example_id": "a", "prediction": 0.2}] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        predictions(p)
