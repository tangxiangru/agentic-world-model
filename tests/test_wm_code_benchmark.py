from copy import deepcopy

import numpy as np
import pytest

from tools.outcome_prediction.wm_code_benchmark import (
    PRIMARY,
    SPECS,
    build_views,
    checked_features,
    coverage_report,
    history_features,
    paired_comparison,
    positive_input,
)
from tools.outcome_prediction.wm_small_benchmark import evaluate


def source(identifier, day, lr=0.01, stage="plan"):
    return {
        "example_id": identifier,
        "cell_id": identifier.split("/")[0],
        "first_submitted_at": f"2026-01-{day:02d}T00:00:00Z",
        "first_stage": stage,
        "audit": {"reasons": []},
        "model_input": {
            "plan": {
                "hypothesis": "result will be 0.99",
                "setup": {
                    "method": {"family": "sft", "hyperparams": {"lr": lr, "other": "0.99"}},
                    "command": {
                        "argv": ["python", "train.py"],
                        "script": "train.py",
                        "env": {"TOKEN": "not-a-real-token"},
                    },
                    "data": [
                        {
                            "source": "gsm8k",
                            "n_examples": 999,
                            "build_command": ["python", "build.py", "--n", "100"],
                        }
                    ],
                    "progress": {"total": 1000},
                },
            },
            "code": [{"role": "training", "status": "reconstructed", "content": "x=1"}],
            "known_previous_checkpoints": [{"score": 0.99}],
        },
        "label": {"accuracy": 0.8},
    }


def example(identifier="r/exp-02", parents=None, history=None, available=True):
    return {
        "example_id": identifier,
        "cell_id": identifier.split("/")[0],
        "benchmark": "gsm8k",
        "parent_reference": 0.2,
        "from_base": False,
        "parent_ids": parents or [],
        "history_ids": history or [],
        "fold": 0,
        "views": {
            view: {"parent_reference": 0.2, "old": i}
            for i, view in enumerate(("reference", "current", "parent", "history"))
        },
        "audit": {"proposal_features_available": available, "fidelity_reviewed_eligible": True},
    }


def extractor(payload):
    return {"lr": payload["plan"]["setup"]["method"]["hyperparams"]["lr"], "flag": 1.0}, {
        "static": True
    }


def test_positive_projection_strips_outcomes_generic_counts_and_env():
    row = source("r/exp-01", 1)
    p = positive_input(row)
    assert set(p) == {"plan", "code"}
    assert set(p["plan"]) == {"setup"}
    assert "other" not in p["plan"]["setup"]["method"]["hyperparams"]
    assert "env" not in p["plan"]["setup"]["command"]
    assert "n_examples" not in p["plan"]["setup"]["data"][0]
    assert "progress" not in p["plan"]["setup"]


def test_unreconstructed_code_never_enters_extractors():
    row = source("r/exp-01", 1)
    row["model_input"]["code"].extend(
        [
            {"status": "blocked", "content": "score=0.99"},
            {"status": "unavailable", "content": "score=0.98"},
        ]
    )
    assert len(positive_input(row)["code"]) == 1


def test_views_keep_original_baselines_and_inputs_unmodified():
    original = example()
    rows = {"r/exp-02": source("r/exp-02", 2)}
    before = deepcopy(original)
    views, _ = build_views([original], rows, extractor, extractor)
    assert original == before
    for name, base in [
        ("v1_current", "current"),
        ("v1_parent", "parent"),
        ("v1_history", "history"),
    ]:
        assert views[0]["views"][name] == original["views"][base]
    assert "extra.current.config.lr" in views[0]["views"]["config_current"]
    assert not any("extra.current.data" in k for k in views[0]["views"]["config_current"])
    assert len(SPECS) == 9 and PRIMARY == "code_parent"


def test_parent_and_full_history_are_different_and_chronological():
    original = example(
        parents=["r/exp-02"], history=["r/exp-02", "r/exp-01"], identifier="r/exp-03"
    )
    rows = {f"r/exp-0{i}": source(f"r/exp-0{i}", i, lr=i / 100) for i in (1, 2, 3)}
    output, _ = build_views([original], rows, extractor, extractor)
    parent = output[0]["views"]["code_parent"]
    history = output[0]["views"]["code_history"]
    assert parent["extra.history.config.steps"] == 1
    assert history["extra.history.config.steps"] == 2
    assert parent["extra.history.config.latest.lr"] == 0.02
    assert history["extra.history.config.mean.lr"] == pytest.approx(0.015)
    assert history["extra.history.config.action_change.lr"] == pytest.approx(0.01)


def test_masked_current_still_can_have_parent_code():
    original = example(parents=["r/exp-01"], history=["r/exp-01"], available=False)
    rows = {"r/exp-01": source("r/exp-01", 1), "r/exp-02": source("r/exp-02", 2, stage="closed")}
    output, audit = build_views([original], rows, extractor, extractor)
    features = output[0]["views"]["code_parent"]
    assert features["extra.current.config.available"] == 0
    assert "extra.current.config.lr" not in features
    assert features["extra.history.config.latest.lr"] == 0.01
    assert features["extra.history.config.action_change.lr"] is None
    assert audit["r/exp-02::current"]["masked"]


@pytest.mark.parametrize("ancestor", ["r/exp-02", "s/exp-01", "missing/exp-01", "r/exp-03"])
def test_unsafe_history_rejected(ancestor):
    original = example(history=[ancestor])
    rows = {
        "r/exp-02": source("r/exp-02", 2),
        "r/exp-03": source("r/exp-03", 3),
        "s/exp-01": source("s/exp-01", 1),
    }
    with pytest.raises(ValueError, match="Unsafe recipe history"):
        build_views([original], rows, extractor, extractor)


def test_duplicate_history_rejected():
    original = example(history=["r/exp-01", "r/exp-01"])
    rows = {"r/exp-02": source("r/exp-02", 2), "r/exp-01": source("r/exp-01", 1)}
    with pytest.raises(ValueError, match="Duplicate"):
        build_views([original], rows, extractor, extractor)


def test_own_labels_and_previous_observations_do_not_affect_features():
    original = example()
    rows = {"r/exp-02": source("r/exp-02", 2)}
    expected, _ = build_views([original], rows, extractor, extractor)
    rows["r/exp-02"]["label"] = {"accuracy": 0.01}
    rows["r/exp-02"]["prior_observations"] = [{"accuracy": 0.99}]
    rows["r/exp-02"]["model_input"]["plan"]["result"] = {"accuracy": 0.98}
    actual, _ = build_views([original], rows, extractor, extractor)
    assert actual == expected


def test_missing_latest_is_not_replaced_by_older_value():
    f = history_features({"lr": 0.03}, [{"lr": 0.01}, {"lr": None}])
    assert f["latest.lr"] is None
    assert f["action_change.lr"] is None
    assert f["mean.lr"] == 0.01


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "accuracy=0.9", [], {}])
def test_numeric_feature_boundary(bad):
    with pytest.raises(ValueError):
        checked_features({"x": bad})


def test_paired_comparison_matches_point_metrics_and_reproducible():
    rows = [{"cell_id": "a"}, {"cell_id": "a"}, {"cell_id": "b"}]
    y, p, b = np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.3, 0.7]), np.array([0.5, 0.6, 0.2])
    result = paired_comparison(rows, y, p, b, repeats=200)
    assert result == paired_comparison(rows, y, p, b, repeats=200)
    ep, eb = evaluate(rows, y, p), evaluate(rows, y, b)
    assert result["relative_mae_reduction"]["estimate"] == pytest.approx(1 - ep["mae"] / eb["mae"])
    assert result["relative_rmse_reduction"]["estimate"] == pytest.approx(
        1 - ep["rmse"] / eb["rmse"]
    )
    assert result["r2_increase"]["estimate"] == pytest.approx(ep["r2"] - eb["r2"])


def test_paired_comparison_rejects_truncation():
    with pytest.raises(ValueError):
        paired_comparison([{"cell_id": "a"}], [0.1], [0.2], [])


def test_coverage_counts_presence_and_nonzero_separately():
    original = example()
    output, _ = build_views([original], {"r/exp-02": source("r/exp-02", 2)}, extractor, extractor)
    c = coverage_report(output)["gsm8k"]
    assert c["examples"] == c["usable_proposals"] == 1
    assert c["fields"]["extra.current.config.lr"]["nonzero"] == 1
