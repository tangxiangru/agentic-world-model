"""Ten-repeat mean-accuracy labels, using the pinned evaluator JSON schema."""

import json
import statistics
from pathlib import Path

import pytest

from tools.outcome_prediction.prefix_labels import load_label, summarize_rates


def example():
    # Every problem succeeds once: mean pass@1 is .1, observed pass@10 is 1.
    problems = {str(i): [float(run == i % 10) for run in range(10)] for i in range(30)}
    return {
        "accuracy": 0.1,
        "epochs": 10,
        "per_epoch_accuracy": [0.1] * 10,
        "std_across_epochs": 0.0,
        "n_problems": 30,
        "inspect_metrics": {"accuracy": 0.1, "stderr": 0.012},
        "per_problem": problems,
        "rescore10": {"id": "sample", "benchmark": "aime2025"},
    }


def write_result(tmp_path, data, name="sample"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def test_complete_label_is_mean_pass_at_one_not_pass_at_ten(tmp_path):
    data = example()
    result = load_label(write_result(tmp_path, data))
    assert result["status"] == "complete"
    assert result["avg_pass_rate"] == pytest.approx(0.1)
    assert result["avg_pass_rate"] != statistics.mean(max(v) for v in data["per_problem"].values())
    assert result["per_run_n"] == [30] * 10
    assert result["pooled_pass_rate"] == pytest.approx(0.1)
    assert result["eligible_for_training"]
    assert all(result["consistency_checks"].values())


def test_unequal_counts_do_not_change_definition_of_target():
    result = summarize_rates([0.0, 1.0], [10, 90])
    assert result["mean"] == 0.5
    assert result["pooled_pass_rate"] == 0.9
    assert result["equal_run_sizes"] is False


def test_zero_accuracy_is_valid(tmp_path):
    data = example()
    data.update(accuracy=0, per_epoch_accuracy=[0] * 10)
    data["per_problem"] = {str(i): [0] * 10 for i in range(30)}
    result = load_label(write_result(tmp_path, data))
    assert result["status"] == "complete"
    assert result["avg_pass_rate"] == 0


def test_missing_is_explicit_and_not_zero(tmp_path):
    result = load_label(tmp_path / "missing.json", benchmark="gsm8k")
    assert result["status"] == "missing"
    assert result["avg_pass_rate"] is None
    assert result["errors"] == ["result_file_missing"]


def test_nine_runs_are_not_a_ten_run_target(tmp_path):
    data = example()
    data["epochs"] = 9
    data["per_epoch_accuracy"] = data["per_epoch_accuracy"][:9]
    data["per_problem"] = {k: v[:9] for k, v in data["per_problem"].items()}
    result = load_label(write_result(tmp_path, data))
    assert result["status"] == "invalid"
    assert result["avg_pass_rate"] is None
    assert result["avg_available_runs_pass_rate"] == pytest.approx(0.1)
    assert "requires_exactly_ten_runs" in result["errors"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("accuracy", 0.9, "stored_accuracy_missing_or_mismatch"),
        ("epochs", 9, "stored_epochs_mismatch"),
        ("n_problems", 29, "benchmark_problem_count_mismatch"),
        ("std_across_epochs", 0.2, "stored_population_sd_missing_or_mismatch"),
        ("per_epoch_accuracy", [float("nan")] * 10, "per_epoch_accuracy_missing_or_invalid"),
        ("per_epoch_accuracy", [True] * 10, "per_epoch_accuracy_missing_or_invalid"),
    ],
)
def test_inconsistent_or_invalid_summary_is_rejected(tmp_path, field, value, error):
    data = example()
    data[field] = value
    result = load_label(write_result(tmp_path, data))
    assert result["status"] == "invalid"
    assert result["avg_pass_rate"] is None
    assert error in result["errors"]


def test_ragged_per_problem_cannot_be_aligned_by_position(tmp_path):
    data = example()
    data["per_problem"]["0"] = [0] * 9
    result = load_label(write_result(tmp_path, data))
    assert result["per_run_n"] is None
    assert "ragged_per_problem_matrix_epoch_alignment_unknown" in result["errors"]


def test_problem_scores_are_independently_checked(tmp_path):
    data = example()
    data["per_problem"]["0"][0] = 0
    result = load_label(write_result(tmp_path, data))
    assert "per_epoch_accuracy_mismatch_problem_scores" in result["errors"]
    assert result["avg_pass_rate"] is None


def test_benchmark_and_exp_id_binding_are_checked(tmp_path):
    result = load_label(write_result(tmp_path, example(), "different"), benchmark="gsm8k")
    assert "benchmark_mismatch" in result["errors"]
    assert "result_exp_id_mismatch" in result["errors"]


def test_quarantine_not_released_by_complete_numeric_label(tmp_path):
    data = example()
    exp_id = "aime-r0-11-exp-02"
    data["rescore10"]["id"] = exp_id
    result = load_label(write_result(tmp_path, data, exp_id))
    assert result["status"] == "complete"
    assert result["avg_pass_rate"] == pytest.approx(0.1)
    assert result["quarantined"]
    assert not result["eligible_for_training"]


@pytest.mark.parametrize("rates,counts", [([], None), ([1.2], None), ([0.5], [0]), ([0.5], [1, 2])])
def test_summary_rejects_invalid_values(rates, counts):
    with pytest.raises(ValueError):
        summarize_rates(rates, counts)


def test_pinned_real_schema_when_mirror_available():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7/rescore10/results/r0-29-exp-02.json"
    )
    if not path.is_file():
        pytest.skip("pinned mirror not installed")
    result = load_label(path, benchmark="gsm8k")
    assert result["status"] == "complete", result["errors"]
    assert result["n_runs"] == 10
    assert result["per_run_n"] == [1319] * 10
    assert result["avg_pass_rate"] == pytest.approx(0.6880212282031842)
    assert result["population_sd_across_runs"] == pytest.approx(0.0022807595081867056)
