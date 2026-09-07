from copy import deepcopy

import numpy as np
import pytest

from tools.outcome_prediction.wm_small_benchmark import (
    SPECS,
    bootstrap_gain,
    evaluate,
    fit_one,
)
from tools.outcome_prediction.wm_small_data import (
    build_examples,
    cohort,
    combine,
    folds,
    numeric,
    safe_text,
    step_features,
)


def inventory(identifier="r/exp-01", first_stage="plan", day=1):
    return {
        "example_id": identifier,
        "first_stage": first_stage,
        "first_submitted_at": f"2026-01-{day:02}T00:00:00Z",
        "initial_eligible": True,
        "eligible": False,
        "label": {"accuracy": 0.4, "official_metric": {"accuracy": 0.4}},
        "model_input": {
            "plan": {
                "setup": {
                    "method": {
                        "family": "sft",
                        "hyperparams": {
                            "lr": 1e-5,
                            "epochs": 2,
                            "batch_size": 4,
                            "grad_accum": 2,
                            "max_steps": 100,
                            "max_seq_len": 512,
                        },
                    },
                    "data": [{"source": "gsm8k", "n_examples": 42}],
                }
            },
            "code": [
                {
                    "status": "reconstructed",
                    "content": "# score 0.9\nx = SFTTrainer(learning_rate=0.0002)\n",
                }
            ],
        },
    }


def card(identifier="r/exp-01", parents=None):
    cell, cid = identifier.split("/")
    return {
        "example_id": identifier,
        "cell_id": cell,
        "card_id": cid,
        "benchmark": "gsm8k",
        "scientist_model": "scientist",
        "official_accuracy": 0.4,
        "lineage_complete": True,
        "parents": parents or [],
        "plan_stage_recorded": True,
        "setup_changed": False,
        "plan_at": "2026-01-02T00:00:00Z",
        "recipe": [{"card_id": cid, "relation": "target"}],
    }


@pytest.mark.parametrize("value", [True, False, "score=0.8", "5 epochs", float("inf"), None])
def test_numeric_rejects_nonnumeric(value):
    assert numeric(value) is None


def test_numeric_preserves_scientific_settings():
    assert numeric("1e-5") == 1e-5


def test_target_and_outcome_prose_not_features():
    a = inventory()
    b = deepcopy(a)
    b["label"] = {"accuracy": 0.999}
    b["model_input"]["plan"]["hypothesis"] = "accuracy will be 0.999"
    b["model_input"]["plan"]["setup"]["progress"] = {"total": 100000}
    b["model_input"]["plan"]["setup"]["data"][0]["n_examples"] = 999999
    b["model_input"]["plan"]["setup"]["method"]["hyperparams"]["other"] = "greedy boxed 99%"
    b["model_input"]["code"][0]["content"] += "# the future score will be 0.99\nscore=0.999\n"
    assert step_features(a) == step_features(b)
    assert "0.999" not in safe_text(step_features(b), [])


def test_closed_current_is_masked_but_known_history_can_be_read():
    row = inventory(first_stage="closed")
    assert step_features(row) == {"proposal.available": 0.0}
    assert step_features(row, completed_history=True)["hp.lr"] == 1e-5


def test_plan_with_existing_results_is_also_masked():
    row = inventory()
    row["audit"] = {"reasons": ["first_result_not_empty"]}
    assert step_features(row) == {"proposal.available": 0.0}


def test_static_code_setting_and_plan_setting_separate():
    f = step_features(inventory())
    assert f["hp.lr"] == 1e-5
    assert f["code.setting.lr"] == 0.0002
    assert f["code.calls.SFTTrainer"] == 1


def test_unavailable_code_not_read():
    row = inventory()
    row["model_input"]["code"][0]["status"] = "blocked"
    f = step_features(row)
    assert f["code.calls.SFTTrainer"] == 0
    assert f["code.setting.lr"] is None


def test_cohort_base_is_explicit_assumed_reference():
    rows, excluded = cohort([card()])
    assert not excluded
    assert rows[0]["parent_reference"] == 0.045
    assert rows[0]["reference_source"] == "legacy_base_constant_not_verified"


def test_cohort_self_parent_rejected():
    with pytest.raises(ValueError, match="Self parent"):
        cohort([card(parents=["exp-01"])])


def test_missing_parent_not_imputed_as_base():
    rows, excluded = cohort([card(parents=["exp-99"])])
    assert not rows
    assert excluded[0]["reason"] == "missing_parent_label"


def test_parent_zero_valid_and_multireference_explicit():
    first, second, child = card(), card("r/exp-02"), card("r/exp-03", ["exp-01", "exp-02"])
    first["official_accuracy"] = 0
    rows, _ = cohort([first, second, child])
    assert rows[-1]["parent_reference"] == 0.2
    assert rows[-1]["reference_source"] == "mean_declared_parent_scores"


def test_folds_do_not_depend_on_label_values():
    cards = [card(f"r{i}/exp-01") for i in range(16)]
    expected = folds(cards)
    for row in cards:
        row["official_accuracy"] = 0.999
    assert folds(cards) == expected


def test_self_data_ancestor_omitted_and_no_history_scores():
    c = card()
    c["recipe"].insert(0, {"card_id": "exp-01", "relation": "data"})
    selected, _ = cohort([c])
    inv = inventory()
    rows, labels = build_examples([c], {inv["example_id"]: inv}, selected)
    assert not rows[0]["history_ids"]
    assert labels == {"r/exp-01": 0.4}
    assert not rows[0]["audit"]["history_scores_used"]
    assert rows[0]["audit"]["initial_fidelity_eligible"]
    assert not rows[0]["audit"]["fidelity_reviewed_eligible"]


def test_future_ancestor_recipe_omitted():
    c = card()
    c["recipe"].insert(0, {"card_id": "exp-02", "relation": "data"})
    selected, _ = cohort([c])
    inv, future = inventory(day=2), inventory("r/exp-02", day=3)
    rows, _ = build_examples([c], {inv["example_id"]: inv, future["example_id"]: future}, selected)
    assert not rows[0]["history_ids"]
    assert rows[0]["audit"]["omitted_history"][0]["reason"] == "not_available_before_proposal"


def test_context_views_and_width_do_not_scale_with_steps():
    current = {"hp.lr": 1e-5}
    history = [{"relation": "weights", "features": {"hp.lr": 2e-5}}]
    a = combine(current, history, history, 0.4, False, "current")
    b = combine(current, history, history, 0.4, False, "history")
    c = combine(current, history, history * 30, 0.4, False, "history")
    assert not any(k.startswith("history.") for k in a)
    assert b.keys() == c.keys()
    assert b["action_minus_recent.hp.lr"] == -1e-5


def test_metrics_equal_sessions_not_cards():
    rows = [{"cell_id": "a"}] * 3 + [{"cell_id": "b"}]
    result = evaluate(rows, [0, 0, 0, 1], [0, 0, 0, 0])
    assert result["mae"] == 0.5
    assert result["pooled_card_mae"] == 0.25


def test_bootstrap_identical_forecasts_zero_gain():
    rows = [{"cell_id": "a"}, {"cell_id": "b"}]
    result = bootstrap_gain(rows, [0.2, 0.8], [0.3, 0.5], [0.3, 0.5], repeats=20)
    assert result["relative_mae_reduction"] == 0
    assert result["ci95"] == [0, 0]


def test_fit_one_never_fits_heldout_targets_and_uses_matching_embedding_indices(monkeypatch):
    import tools.outcome_prediction.wm_small_models as module

    observed = {}

    class Fake:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, features, labels, reference, groups, embeddings=None):
            observed["fit"] = (features, list(labels), list(reference), groups, embeddings.tolist())
            return self

        def predict(self, features, reference, embeddings=None):
            observed["predict"] = (features, list(reference), embeddings.tolist())
            return np.zeros(len(features))

    monkeypatch.setattr(module, "SmallPredictor", Fake)
    rows = [
        {"views": {"history": {"v": i}}, "parent_reference": 0.1 * i, "cell_id": str(i)}
        for i in range(4)
    ]
    fit_one(
        rows,
        np.array([0.2, 0.3, 0.4, 0.99]),
        np.array([0, 2]),
        np.array([3]),
        SPECS["embedding_ridge_delta"],
        10,
        np.arange(8).reshape(4, 2),
    )
    assert observed["fit"][1] == [0.2, 0.4]
    assert observed["fit"][4] == [[0, 1], [4, 5]]
    assert observed["predict"][2] == [[6, 7]]
