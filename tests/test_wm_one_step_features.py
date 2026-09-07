"""Synthetic feature-boundary tests; no real labels or predictors are loaded."""

import copy
import math

import pytest

from tools.outcome_prediction import wm_one_step_features as adapter


def recipe(*, epochs=2, lr=1e-5, family="sft", code=None):
    return {
        "plan": {
            "setup": {
                "method": {
                    "family": family,
                    "hyperparams": {"epochs": epochs, "lr": lr},
                },
                "command": {
                    "argv": ["python", "train.py"],
                    "script": "/task/train.py",
                    "cwd": "/task",
                },
                "data": [],
            },
        },
        "code": []
        if code is None
        else [
            {
                "status": "reconstructed",
                "role": "training",
                "script_path": "/task/train.py",
                "content": code,
            }
        ],
    }


def example(*, history=None, kind="measured", accuracy=0.5, current=None):
    return {
        "example_id": "secret-session/exp-3",
        "cell_id": "secret-session",
        "benchmark": "private-benchmark",
        "scientist_model": "private-scientist",
        "parent": {"accuracy": accuracy, "kind": kind, "reference_known": True},
        "model_input": recipe() if current is None else current,
        "history": [] if history is None else history,
        "history_complete_to_base": kind == "published_base",
    }


def ancestor(epochs, *, status="screened"):
    return {
        "example_id": "ignored-ancestor",
        "recipe_status": status,
        "model_input": recipe(epochs=epochs),
    }


@pytest.mark.parametrize("mode,width", [("parent", 3), ("current", 72), ("history", 151)])
def test_mode_width_is_fixed_finite_and_nested(mode, width):
    empty = adapter.features(example(current=recipe(epochs=None, lr=None)), mode)
    rich = adapter.features(example(history=[ancestor(1), ancestor(2)]), mode)
    assert len(empty) == len(rich) == width
    assert empty.keys() == rich.keys()
    assert all(type(value) is float and math.isfinite(value) for value in rich.values())
    assert (
        adapter.features(example(), "parent").items()
        <= adapter.features(example(), "current").items()
    )
    assert (
        adapter.features(example(), "current").items()
        <= adapter.features(example(), "history").items()
    )


@pytest.mark.parametrize("score", [0, 0.0, 0.25, 1, 1.0])
def test_valid_parent_score_including_zero_is_preserved(score):
    assert adapter.features(example(accuracy=score), "parent")["parent.accuracy"] == score


@pytest.mark.parametrize(
    "score",
    [None, True, False, "0.5", -0.01, 1.01, float("nan"), float("inf"), -float("inf"), {}, []],
)
def test_missing_invalid_or_nonfinite_parent_accuracy_is_not_imputed(score):
    with pytest.raises((TypeError, ValueError)):
        adapter.features(example(accuracy=score))


def test_parent_reference_type_is_explicit_and_producer_identity_is_optional():
    observed = adapter.features(example(), "parent")
    base = example(kind="published_base")
    base["parent"].pop("reference_known")
    supplied = adapter.features(base, "parent")
    assert observed["parent.reference.measured"] == 1
    assert observed["parent.reference.published_base"] == 0
    assert supplied["parent.reference.measured"] == 0
    assert supplied["parent.reference.published_base"] == 1
    assert observed["parent.accuracy"] == supplied["parent.accuracy"]


@pytest.mark.parametrize("path_kind", ["immutable_archive", "archived_source"])
def test_legacy_export_measured_parent_contract(path_kind):
    row = example()
    row["parent"].pop("kind")
    row["parent"]["path_kind"] = path_kind
    assert adapter.features(row, "parent")["parent.reference.measured"] == 1


@pytest.mark.parametrize(
    "parent",
    [
        None,
        {},
        {"accuracy": 0.2},
        {"accuracy": 0.2, "kind": "unknown"},
        {"accuracy": 0.2, "kind": "measured", "reference_known": False},
        {"accuracy": 0.2, "kind": "measured", "reference_known": 1},
    ],
)
def test_unidentified_or_unverified_reference_rejected(parent):
    row = example()
    row["parent"] = parent
    with pytest.raises((TypeError, ValueError)):
        adapter.features(row)


class Forbidden:
    def __deepcopy__(self, memo):
        raise AssertionError("Forbidden field copied")

    def __iter__(self):
        raise AssertionError("Forbidden field iterated")

    def __str__(self):
        raise AssertionError("Forbidden field stringified")


def test_parent_ablation_does_not_inspect_current_recipe_or_history():
    row = example()
    row["model_input"] = row["history"] = Forbidden()
    assert len(adapter.features(row, "parent")) == 3


def test_current_ablation_does_not_inspect_history():
    row = example()
    expected = adapter.features(row, "current")
    row["history"] = row["history_complete_to_base"] = Forbidden()
    assert adapter.features(row, "current") == expected


def test_target_labels_all_ancestor_scores_ids_and_free_prose_are_ignored():
    row = example(history=[ancestor(1), ancestor(4)])
    expected = adapter.features(row, "history")
    for key in (
        "example_id",
        "cell_id",
        "benchmark",
        "scientist_model",
        "label",
        "result",
        "accuracy",
        "delta_accuracy",
        "prior_observations",
    ):
        row[key] = Forbidden()
    row["parent"]["producer_id"] = row["parent"]["consumed_path"] = Forbidden()
    for item in row["history"]:
        for key in ("example_id", "label", "accuracy", "result", "parent", "prior_observations"):
            item[key] = Forbidden()
    for source in [row["model_input"], *(h["model_input"] for h in row["history"])]:
        source["result"] = source["prior_observations"] = Forbidden()
        source["plan"]["hypothesis"] = source["plan"]["result"] = Forbidden()
        setup = source["plan"]["setup"]
        setup["parent_checkpoint"] = setup["budget"] = setup["result"] = Forbidden()
        setup["method"]["hyperparams"].update(
            accuracy=Forbidden(), score=Forbidden(), notes=Forbidden()
        )
    assert adapter.features(row, "history") == expected


def test_irrelevant_numeric_metadata_and_dataset_yields_do_not_change_features():
    row = example()
    row["model_input"]["plan"]["setup"]["data"] = [{"source": "gsm8k"}]
    expected = adapter.features(row)
    row["model_input"]["plan"]["setup"]["data"][0].update(
        n_examples=999999, accuracy=0.999, result={"score": 0.999}
    )
    row["model_input"]["plan"]["setup"]["method"]["hyperparams"].update(
        seed=987654, run_id=12, final_accuracy=0.999
    )
    assert adapter.features(row) == expected


def test_current_configuration_and_planned_data_quota_are_used():
    row = example(
        current=recipe(
            code="from transformers import TrainingArguments\na=TrainingArguments(learning_rate=3e-5,num_train_epochs=7)"
        )
    )
    row["model_input"]["plan"]["setup"]["data"] = [
        {
            "source": "gsm8k",
            "build_command": "python build.py --n 2048",
            "selection": "Keep correct solutions.",
        }
    ]
    result = adapter.features(row)
    assert result["current.config.learning_rate"] == 3e-5
    assert result["current.config.epochs"] == 7
    assert result["current.data.quota.requested_n"] == 2048
    assert result["current.data.policy.evidence.correctness_filter"] == 1


def test_mutable_snapshot_code_and_unrelated_code_constants_are_not_inputs():
    row = example(
        current=recipe(
            code="from transformers import TrainingArguments\na=TrainingArguments(num_train_epochs=999)"
        )
    )
    row["model_input"]["code"][0]["status"] = "snapshot_only"
    assert adapter.features(row) == adapter.features(example())
    clean = example(current=recipe(code="unrelated_id=1234\naccuracy=.8\nprint(accuracy)"))
    changed = copy.deepcopy(clean)
    changed["model_input"]["code"][0]["content"] = "unrelated_id=9876\naccuracy=.1\nprint(accuracy)"
    assert adapter.features(clean) == adapter.features(changed)


def test_consistent_path_renaming_does_not_encode_path_names():
    row = example(
        current=recipe(
            code="from transformers import TrainingArguments\na=TrainingArguments(num_train_epochs=7)"
        )
    )
    expected = adapter.features(row)
    command = row["model_input"]["plan"]["setup"]["command"]
    command.update(argv=["python", "renamed.py"], script="/elsewhere/renamed.py", cwd="/elsewhere")
    row["model_input"]["code"][0]["script_path"] = "/elsewhere/renamed.py"
    assert adapter.features(row) == expected


def test_missing_configuration_is_distinct_from_a_known_zero():
    unknown = adapter.features(example(current=recipe(lr=None)))
    zero = adapter.features(example(current=recipe(lr=0)))
    assert unknown["current.config.learning_rate"] == zero["current.config.learning_rate"] == 0
    assert unknown["current.observed.config.learning_rate"] == 0
    assert zero["current.observed.config.learning_rate"] == 1


def test_all_ancestors_contribute_to_fixed_mean_max_latest_and_decay():
    result = adapter.features(example(history=[ancestor(1), ancestor(3), ancestor(5)]), "history")
    prefix = "config.epochs"
    assert result["history.mean." + prefix] == 3
    assert result["history.max." + prefix] == 5
    assert result["history.latest." + prefix] == 5
    assert result["history.decay." + prefix] == pytest.approx((1 * 0.25 + 3 * 0.5 + 5) / 1.75)
    assert result["history.observed_fraction." + prefix] == 1
    assert result["history.steps"] == 3


def test_quarantined_latest_is_not_backfilled_from_an_older_recipe():
    quarantined = {
        "recipe_status": "quarantined",
        "model_input": Forbidden(),
        "accuracy": Forbidden(),
    }
    result = adapter.features(example(history=[ancestor(4), quarantined]), "history")
    assert result["history.latest.config.epochs"] == 0
    assert result["history.latest_observed.config.epochs"] == 0
    assert result["history.mean.config.epochs"] == 4
    assert result["history.observed_fraction.config.epochs"] == 0.5
    assert result["history.screened_steps"] == 1
    assert result["history.latest_screened"] == 0


def test_decay_preserves_distance_across_quarantined_gaps():
    result = adapter.features(
        example(history=[ancestor(1), ancestor(999, status="quarantined"), ancestor(5)]), "history"
    )
    assert result["history.decay.config.epochs"] == pytest.approx((0.25 + 5) / 1.25)
    assert result["history.max.config.epochs"] == 5


def test_latest_screened_recipe_with_missing_setting_does_not_backfill_setting():
    result = adapter.features(example(history=[ancestor(4), ancestor(None)]), "history")
    assert result["history.latest_screened"] == 1
    assert result["history.latest.config.epochs"] == 0
    assert result["history.latest_observed.config.epochs"] == 0
    assert result["history.mean.config.epochs"] == 4


def test_history_order_changes_recency_but_not_mean_or_max():
    a = adapter.features(example(history=[ancestor(1), ancestor(5)]), "history")
    b = adapter.features(example(history=[ancestor(5), ancestor(1)]), "history")
    assert a["history.mean.config.epochs"] == b["history.mean.config.epochs"]
    assert a["history.max.config.epochs"] == b["history.max.config.epochs"]
    assert a["history.latest.config.epochs"] != b["history.latest.config.epochs"]
    assert a["history.decay.config.epochs"] != b["history.decay.config.epochs"]


def test_base_without_history_uses_fixed_zero_history_and_explicit_completeness():
    row = example(kind="published_base")
    row.pop("history")
    row.pop("history_complete_to_base")
    result = adapter.features(row, "history")
    assert result["history.steps"] == 0
    assert result["history.complete_to_base"] == 1
    assert result["history.mean.config.epochs"] == 0
    assert result["history.observed_fraction.config.epochs"] == 0


@pytest.mark.parametrize(
    "history",
    [
        None,
        "history",
        [{"recipe_status": "unknown"}],
        [{"recipe_status": "screened", "model_input": None}],
        [None],
    ],
)
def test_unknown_history_contract_is_rejected(history):
    row = example()
    row["history"] = history
    with pytest.raises((TypeError, ValueError)):
        adapter.features(row, "history")


def test_published_base_with_checkpoint_ancestors_is_rejected():
    with pytest.raises(ValueError):
        adapter.features(example(kind="published_base", history=[ancestor(1)]), "history")


@pytest.mark.parametrize("mode", [None, "absolute", "all", 1])
def test_unknown_feature_mode_is_rejected(mode):
    with pytest.raises(ValueError):
        adapter.features(example(), mode)


def test_extractor_nonfinite_values_are_not_silently_used(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "extract_config_features",
        lambda source: ({"codecfg.learning_rate": float("inf")}, {}),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.features(example())


def test_features_do_not_mutate_input():
    row = example(history=[ancestor(1), ancestor(3)])
    before = copy.deepcopy(row)
    adapter.features(row, "history")
    assert row == before
