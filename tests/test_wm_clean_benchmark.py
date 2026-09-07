"""Synthetic-only checks for clean benchmark splits and training boundaries."""

import copy
from typing import ClassVar

import numpy as np
import pytest

from tools.outcome_prediction import wm_clean_benchmark as benchmark
from tools.outcome_prediction import wm_clean_models


def example(identifier, cell, task="aime2025", split="train", arm="old"):
    return {
        "example_id": identifier,
        "cell_id": cell,
        "benchmark": task,
        "scientist_model": "old-scientist" if arm == "old" else "new-scientist",
        "split": split,
        "source_arm": arm,
    }


def regime_examples():
    return [
        example("g/train", "g-train", "gsm8k"),
        example("g/test", "g-test", "gsm8k", "test"),
        example("a/old-train-1", "old-train"),
        example("a/old-train-2", "old-train"),
        example("a/old-test", "old-test", split="test"),
        example("a/new-train", "new-train", arm="new"),
        example("a/new-test", "new-test", split="test", arm="new"),
    ]


def test_regimes_preserve_fixed_split_and_scientist_shift_definitions():
    rows = regime_examples()
    before = copy.deepcopy(rows)
    regimes = benchmark.make_regimes(rows)
    assert rows == before
    assert set(regimes) == {
        "session_holdout__gsm8k",
        "session_holdout__aime2025",
        "old_data_only__aime2025",
        "new_scientist_holdout__aime2025",
    }
    gsm = regimes["session_holdout__gsm8k"]
    assert set(gsm["train_ids"]) == {"g/train"}
    assert set(gsm["test_ids"]) == {"g/test"}
    aime = regimes["session_holdout__aime2025"]
    assert set(aime["train_ids"]) == {"a/old-train-1", "a/old-train-2", "a/new-train"}
    assert set(aime["test_ids"]) == {"a/old-test", "a/new-test"}
    old = regimes["old_data_only__aime2025"]
    assert set(old["train_ids"]) == {"a/old-train-1", "a/old-train-2"}
    assert set(old["test_ids"]) == set(aime["test_ids"])
    shift = regimes["new_scientist_holdout__aime2025"]
    assert set(shift["train_ids"]) == {"a/old-train-1", "a/old-train-2", "a/old-test"}
    assert set(shift["test_ids"]) == {"a/new-train", "a/new-test"}
    by_id = {row["example_id"]: row for row in rows}
    for regime in regimes.values():
        train_cells = {by_id[key]["cell_id"] for key in regime["train_ids"]}
        test_cells = {by_id[key]["cell_id"] for key in regime["test_ids"]}
        assert train_cells.isdisjoint(test_cells)
        assert all(
            by_id[key]["benchmark"] == regime["benchmark"]
            for key in regime["train_ids"] + regime["test_ids"]
        )


def test_regimes_omit_empty_training_or_test_partitions():
    assert benchmark.make_regimes([]) == {}
    assert benchmark.make_regimes([example("g/only", "one", "gsm8k")]) == {}
    assert benchmark.make_regimes([example("g/only", "one", "gsm8k", "test")]) == {}
    regimes = benchmark.make_regimes(
        [
            example("a/train", "train"),
            example("a/test", "test", split="test"),
        ]
    )
    assert "new_scientist_holdout__aime2025" not in regimes
    assert "session_holdout__gsm8k" not in regimes


def test_regimes_reject_duplicate_example_ids():
    rows = regime_examples()
    duplicate = copy.deepcopy(rows[0])
    duplicate["cell_id"] = "different-cell"
    with pytest.raises(ValueError):
        benchmark.make_regimes(rows + [duplicate])


def test_regimes_reject_session_crossing_fixed_partitions():
    rows = [
        example("g/a", "shared", "gsm8k"),
        example("g/b", "shared", "gsm8k", "test"),
    ]
    with pytest.raises(ValueError):
        benchmark.make_regimes(rows)


def test_regimes_reject_session_crossing_scientist_shift_arms():
    rows = [
        example("a/old", "shared"),
        example("a/new", "shared", arm="new"),
    ]
    with pytest.raises(ValueError):
        benchmark.make_regimes(rows)


def test_weighted_median_gives_equal_total_weight_to_each_session():
    labels = [0.1] * 5 + [0.6, 0.9]
    groups = ["large"] * 5 + ["medium", "small"]
    assert benchmark.weighted_median(labels, groups) == pytest.approx(0.6)
    assert np.median(labels) == pytest.approx(0.1)


def test_weighted_median_is_invariant_to_replication_within_a_session():
    labels = [0.1, 0.3, 0.6, 0.9]
    groups = ["first", "first", "second", "third"]
    original = benchmark.weighted_median(labels, groups)
    repeated = benchmark.weighted_median(
        [0.1, 0.3] * 7 + [0.6, 0.9],
        ["first"] * 14 + ["second", "third"],
    )
    assert original == repeated == pytest.approx(0.6)


def test_weighted_median_uses_labels_not_group_identity_values():
    labels = np.array([0.7, 0.2, 0.4])
    assert benchmark.weighted_median(labels, ["z", "a", "b"]) == pytest.approx(0.4)
    assert benchmark.weighted_median(labels, ["a", "b", "z"]) == pytest.approx(0.4)
    assert benchmark.weighted_median([0.0], ["singleton"]) == 0.0


class RecordingPredictor:
    """No estimator fitting: predictions are a known function of synthetic inputs."""

    instances: ClassVar[list] = []

    def __init__(self, name):
        self.name = name
        self.predict_calls = []
        self.__class__.instances.append(self)

    def fit(self, features, labels, references, groups):
        self.fit_features = copy.deepcopy(list(features))
        self.fit_labels = np.array(labels, dtype=float)
        self.fit_references = np.array(references, dtype=float)
        self.fit_groups = list(groups)
        return self

    def predict(self, features, references):
        self.predict_calls.append((copy.deepcopy(list(features)), list(references)))
        # The ridge fake is deliberately perfect on the synthetic training folds.
        offset = 0.0 if self.name == "ridge_delta" else 0.2
        return np.clip([row["signal"] + offset for row in features], 0, 1)

    def metadata(self):
        return {"name": self.name, "synthetic_test_double": True}


@pytest.fixture
def recording_models(monkeypatch):
    RecordingPredictor.instances = []
    monkeypatch.setattr(wm_clean_models, "CleanPredictor", RecordingPredictor)
    return RecordingPredictor.instances


def fit_examples(group_count=5):
    rows, labels = [], {}
    for index in range(group_count):
        value = 0.1 * (index + 1)
        # Unequal session sizes make ordinary means observably wrong.
        for repetition in range(3 if index == 0 else 1):
            row = example(f"train-{index}-{repetition}", f"session-{index}")
            row.update(
                {
                    "parent_reference": 0.0,
                    "reference_known": False,
                    "views": {
                        view: {"signal": value, "view_marker": float(marker)}
                        for marker, view in enumerate(
                            ("current", "parent", "history", "code_parent")
                        )
                    },
                }
            )
            rows.append(row)
            labels[row["example_id"]] = value
    test = []
    for index, (value, known, reference) in enumerate(((0.65, False, 0.0), (0.75, True, 0.8))):
        row = example(f"test-{index}", f"test-session-{index}", split="test")
        row.update(
            {
                "parent_reference": reference,
                "reference_known": known,
                "views": {
                    view: {"signal": value, "view_marker": float(marker)}
                    for marker, view in enumerate(("current", "parent", "history", "code_parent"))
                },
            }
        )
        test.append(row)
    return rows, labels, test


def test_baselines_use_equal_session_training_statistics_and_only_known_references(
    recording_models,
):
    train, labels, test = fit_examples()
    before = copy.deepcopy((train, labels, test))
    predictions, _ = benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert (train, labels, test) == before
    np.testing.assert_allclose(predictions["train_mean"], [0.3, 0.3])
    np.testing.assert_allclose(predictions["train_median"], [0.3, 0.3])
    np.testing.assert_allclose(predictions["parent_or_train_mean"], [0.3, 0.8])


def test_selection_uses_train_only_group_folds_and_records_train_winner(recording_models):
    train, labels, test = fit_examples()
    predictions, details = benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert details["selected"] == "ridge_delta"
    assert details["inner_mae"]["ridge_delta"] == pytest.approx(0.0)
    assert details["selected"] == min(details["inner_mae"], key=details["inner_mae"].get)
    assert set(details["inner_fold_of"]) == {row["cell_id"] for row in train}
    assert len(set(details["inner_fold_of"].values())) == 5
    np.testing.assert_allclose(predictions["inner_selected"], predictions["ridge_delta"])
    np.testing.assert_allclose(
        predictions["fixed_blend"],
        (predictions["regularized_delta"] + predictions["ridge_delta"]) / 2,
    )
    expected_names = set(benchmark.SPECS) | {
        "train_mean",
        "train_median",
        "parent_or_train_mean",
        "fixed_blend",
        "inner_selected",
    }
    assert set(predictions) == expected_names
    allowed_groups = {row["cell_id"] for row in train}
    for model in recording_models:
        assert set(model.fit_groups) <= allowed_groups
        assert len(set(model.fit_groups)) in (4, 5)
        assert all(set(row) == {"signal", "view_marker"} for row in model.fit_features)
        np.testing.assert_allclose(model.fit_labels, [row["signal"] for row in model.fit_features])
        fitted_signals = {row["signal"] for row in model.fit_features}
        for predicted_rows, _ in model.predict_calls:
            predicted_signals = {row["signal"] for row in predicted_rows}
            assert fitted_signals.isdisjoint(predicted_signals)
    final_fits = [model for model in recording_models if len(set(model.fit_groups)) == 5]
    assert len(final_fits) == len(benchmark.SPECS)


def test_outer_test_features_do_not_influence_inner_selection(recording_models):
    train, labels, test = fit_examples()
    _, first = benchmark.select_and_fit(train, labels, test, model_dir=None)
    changed_test = copy.deepcopy(test)
    for row in changed_test:
        for features in row["views"].values():
            features["signal"] = 0.01
            features["test_only_feature"] = 99.0
    _, second = benchmark.select_and_fit(train, labels, changed_test, model_dir=None)
    assert first["inner_mae"] == second["inner_mae"]
    assert first["selected"] == second["selected"]
    assert first["inner_fold_of"] == second["inner_fold_of"]


def test_selector_can_choose_a_trivial_baseline_and_uses_fixed_name_tie_break(
    recording_models, monkeypatch
):
    train, labels, test = fit_examples()
    labels = dict.fromkeys(labels, 0.2)
    monkeypatch.setattr(
        RecordingPredictor,
        "predict",
        lambda self, features, references: np.full(len(features), 0.9),
    )
    predictions, details = benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert details["selected"] == "parent_or_train_mean"
    for name in benchmark.BASELINES:
        assert details["inner_mae"][name] == pytest.approx(0.0)
    np.testing.assert_allclose(predictions["inner_selected"], [0.2, 0.8])


def test_each_final_model_uses_the_declared_feature_view(recording_models):
    train, labels, test = fit_examples()
    benchmark.select_and_fit(train, labels, test, model_dir=None)
    final_fits = [model for model in recording_models if len(set(model.fit_groups)) == 5]
    markers = {
        view: marker for marker, view in enumerate(("current", "parent", "history", "code_parent"))
    }
    for model, (kind, view) in zip(final_fits, benchmark.SPECS.values()):
        assert model.name == kind
        assert all(row["view_marker"] == markers[view] for row in model.fit_features)
        assert all(
            row["view_marker"] == markers[view] for rows, _ in model.predict_calls for row in rows
        )


def test_inner_fold_count_is_capped_by_available_training_sessions(recording_models):
    train, labels, test = fit_examples(group_count=3)
    _, details = benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert len(set(details["inner_fold_of"].values())) == 3
    assert all(len(set(model.fit_groups)) in (2, 3) for model in recording_models)


@pytest.mark.parametrize("change", ["extra_test_label", "missing_train_label"])
def test_fitting_rejects_label_mapping_that_is_not_exactly_training_ids(recording_models, change):
    train, labels, test = fit_examples()
    if change == "extra_test_label":
        labels[test[0]["example_id"]] = 0.95
    else:
        del labels[train[0]["example_id"]]
    with pytest.raises(ValueError):
        benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert not recording_models


def test_fitting_rejects_train_test_session_overlap_before_any_fit(recording_models):
    train, labels, test = fit_examples()
    test[0]["cell_id"] = train[0]["cell_id"]
    with pytest.raises(ValueError):
        benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert not recording_models


def test_fitting_rejects_same_example_across_train_and_test_even_with_different_cells(
    recording_models,
):
    train, labels, test = fit_examples()
    test[0]["example_id"] = train[0]["example_id"]
    assert test[0]["cell_id"] != train[0]["cell_id"]
    with pytest.raises(ValueError):
        benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert not recording_models


def test_fitting_requires_at_least_two_training_sessions(recording_models):
    train, labels, test = fit_examples(group_count=1)
    with pytest.raises(ValueError):
        benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert not recording_models


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_baseline_and_selection_validation_rejects_invalid_labels_before_fit(
    recording_models, invalid
):
    train, labels, test = fit_examples()
    labels[train[0]["example_id"]] = invalid
    with pytest.raises((ValueError, TypeError)):
        benchmark.select_and_fit(train, labels, test, model_dir=None)
    assert not recording_models
