import copy

import pytest

from tools.outcome_prediction import wm_delta_siblings as delta


def step(example_id="run/child"):
    return {
        "example_id": example_id,
        "cell_id": "run",
        "benchmark": "gsm8k",
        "eligible": True,
        "first_submitted_at": "2026-01-02T00:00:00Z",
        "task": {"benchmark": "gsm8k", "base_model": "model", "evaluation_n": 100},
        "model_input": {
            "plan": {
                "setup": {
                    "method": {"family": "sft", "hyperparams": {"lr": 1e-5, "epochs": 2}},
                    "data": [{"source": "openai/gsm8k", "n_examples": 9876}],
                }
            },
            "code": [{"status": "reconstructed", "content": "SFTTrainer()\n# accuracy 0.999"}],
        },
        "audit": {"output_artifact_comparison": {"final_output_checkpoint": "/weights/child"}},
        "label": {"accuracy": 0.6, "evaluation_n": 100, "official_metric": {"accuracy": 0.6}},
    }


def graph_node(example_id, parents=(), closure=None):
    return {
        "is_merge_declared": False,
        "parents": list(parents),
        "topological_closure": closure or [example_id],
        "closure_has_unresolved_dependencies": False,
    }


def edge(parent="run/parent", artifact="/weights/parent"):
    return {
        "kind": "weights",
        "internal": False,
        "producer_id": parent,
        "artifact": artifact,
        "card_only": False,
        "resolution_status": "time_qualified_declared_artifact",
    }


def fixture():
    parent = step("run/parent")
    parent["first_submitted_at"] = "2026-01-01T00:00:00Z"
    parent["audit"]["output_artifact_comparison"]["final_output_checkpoint"] = "/weights/parent"
    a, b = step("run/a"), step("run/b")
    b["first_submitted_at"] = "2026-01-03T00:00:00Z"
    rows = {r["example_id"]: r for r in [parent, a, b]}
    nodes = {"run/parent": graph_node("run/parent")}
    for key in ("run/a", "run/b"):
        nodes[key] = graph_node(key, [edge()], ["run/parent", key])
    return (
        rows,
        nodes,
        {"cell_partition": {"run": "test"}},
        {k: True for k in rows},
        {"run/parent": delta.timestamp("2026-01-01T12:00:00Z")},
    )


def test_signed_delta_and_sibling_algebra():
    examples, _ = delta.build_inputs(*fixture())
    pairs, _ = delta.sibling_pairs(examples)
    labels = {"run/a": 0.4, "run/b": 0.7}
    predictions = {"run/a": -0.2, "run/b": 0.1}
    result = delta.metrics(examples, pairs, labels, predictions)
    assert result["delta_mae"] == pytest.approx(0)
    assert result["pair_accuracy"] == 1
    assert result["pairs"][0]["true_delta_a"] < 0
    assert (labels["run/a"] - 0.6) - (labels["run/b"] - 0.6) == pytest.approx(
        labels["run/a"] - labels["run/b"]
    )


def test_parent_carry_is_fair_tie_not_id_tiebreak():
    examples, _ = delta.build_inputs(*fixture())
    pairs, _ = delta.sibling_pairs(examples)
    result = delta.metrics(examples, pairs, {"run/a": 0.4, "run/b": 0.8}, {"run/a": 0, "run/b": 0})
    assert result["pair_accuracy"] == 0.5
    assert result["chosen_accuracy"] == pytest.approx(0.6)
    assert result["regret"] == pytest.approx(0.2)


def test_true_ties_are_counted_but_not_ranking_successes():
    examples, _ = delta.build_inputs(*fixture())
    pairs, _ = delta.sibling_pairs(examples)
    result = delta.metrics(
        examples, pairs, {"run/a": 0.5, "run/b": 0.5}, {"run/a": 0.2, "run/b": 0.1}
    )
    assert result["true_ties"] == 1
    assert result["pair_accuracy"] is None
    assert result["regret"] == 0


def test_child_outcomes_do_not_change_input_or_pair_cohort():
    original = fixture()
    examples, _ = delta.build_inputs(*original)
    poisoned = copy.deepcopy(original)
    for key in ("run/a", "run/b"):
        poisoned[0][key]["label"] = {"accuracy": object()}
        poisoned[0][key]["prior_observations"] = object()
        poisoned[0][key]["model_input"]["known_previous_checkpoints"] = object()
        poisoned[0][key]["model_input"]["plan"]["hypothesis"] = object()
    again, _ = delta.build_inputs(*poisoned)
    assert examples == again
    assert delta.sibling_pairs(examples) == delta.sibling_pairs(again)


def test_feature_boundary_preserves_numbers_and_ignores_yields_comments():
    row = step()
    original = delta.positive_features(row)
    row["model_input"]["plan"]["setup"]["data"][0]["n_examples"] = 123
    row["model_input"]["plan"]["setup"]["progress"] = {"total": 999}
    row["model_input"]["code"][0]["content"] = "SFTTrainer()\n# GRPOTrainer() accuracy0.1"
    assert original == delta.positive_features(row)
    assert original["hp.lr"] == 1e-5
    assert original["code.syntax.SFTTrainer"] == 1
    assert original["code.syntax.GRPOTrainer"] == 0


@pytest.mark.parametrize("artifact", ["/weights/parent/checkpoint-100", "/weights/parent-greedy"])
def test_parent_card_is_not_enough_for_artifact_matching(artifact):
    f = fixture()
    f[1]["run/a"]["parents"][0]["artifact"] = artifact
    examples, excluded = delta.build_inputs(*f)
    assert "run/a" not in {r["example_id"] for r in examples}
    assert any(e["reason"] == "parent_score_artifact_path_mismatch" for e in excluded)


def test_missing_parent_score_is_not_zero_or_median():
    f = fixture()
    f[3]["run/parent"] = False
    examples, excluded = delta.build_inputs(*f)
    assert not examples
    assert any(e["reason"] == "parent_official_grade_unavailable" for e in excluded)


def test_zero_parent_score_is_valid():
    f = fixture()
    f[0]["run/parent"]["label"]["accuracy"] = 0.0
    f[0]["run/parent"]["label"]["official_metric"]["accuracy"] = 0.0
    examples, _ = delta.build_inputs(*f)
    assert len(examples) == 2
    assert all(r["parent_accuracy"] == 0 for r in examples)


def test_child_before_parent_close_is_not_used():
    f = fixture()
    f[4]["run/parent"] = delta.timestamp("2026-01-02T12:00:00Z")
    examples, _ = delta.build_inputs(*f)
    assert [r["example_id"] for r in examples] == ["run/b"]


def test_history_is_identical_despite_sibling_proposal_times():
    examples, _ = delta.build_inputs(*fixture())
    assert examples[0]["history"] == examples[1]["history"]
    assert examples[0]["history_sha256"] == examples[1]["history_sha256"]
    assert len(delta.sibling_pairs(examples)[0]) == 1


def test_different_history_or_parent_never_paired():
    examples, _ = delta.build_inputs(*fixture())
    examples[0]["parent_key"] = "another_artifact"
    assert not delta.sibling_pairs(examples)[0]
    examples[0]["parent_key"] = examples[1]["parent_key"]
    examples[0]["history_sha256"] = "different_history"
    assert not delta.sibling_pairs(examples)[0]


def test_rival_dependency_disqualifies_pair():
    examples, _ = delta.build_inputs(*fixture())
    examples[1]["candidate_closure"].append(examples[0]["example_id"])
    pairs, excluded = delta.sibling_pairs(examples)
    assert not pairs
    assert excluded[0]["reason"] == "one_candidate_is_dependency_of_other"


def test_generator_is_not_student_parent():
    node = graph_node("x", [dict(edge(), kind="generated_data")])
    with pytest.raises(ValueError, match="not_exactly_one_weight_parent"):
        delta.parent_edge(node)


def test_no_synthetic_base_accuracy():
    node = graph_node(
        "x", [dict(edge(), producer_id=None, resolution_status="declared_base_model")]
    )
    with pytest.raises(ValueError, match="base_parent_score_not_recorded"):
        delta.parent_edge(node)


def test_typed_history_relations():
    nodes = {
        "parent": graph_node(
            "parent", [edge("weights"), dict(edge("generator"), kind="generated_data")]
        ),
        "weights": graph_node("weights"),
        "generator": graph_node("generator"),
    }
    assert delta.history_relations("parent", nodes) == {
        "parent": "parent",
        "weights": "weights",
        "generator": "generated_data",
    }


def test_benchmark_and_run_mismatch_rejected():
    for kind in ("task", "cell_id"):
        f = fixture()
        f[0]["run/parent"][kind] = "different"
        f[2]["cell_partition"]["different"] = "train"
        examples, _ = delta.build_inputs(*f)
        assert not examples


def test_invalid_official_parent_grade_rejected():
    f = fixture()
    f[0]["run/parent"]["label"]["accuracy"] = float("nan")
    examples, _ = delta.build_inputs(*f)
    assert not examples


def test_metrics_weight_runs_not_number_of_pairs():
    examples = [
        {"example_id": k, "cell_id": k[0], "parent_accuracy": 0.5}
        for k in ("a1", "a2", "a3", "b1", "b2")
    ]
    pairs = [
        {"pair_id": str(i), "a": a, "b": b, "cell_id": a[0]}
        for i, (a, b) in enumerate([("a1", "a2"), ("a1", "a3"), ("b1", "b2")])
    ]
    labels = {"a1": 0.8, "a2": 0.4, "a3": 0.4, "b1": 0.8, "b2": 0.4}
    pred = {"a1": 0.2, "a2": 0, "a3": 0, "b1": 0, "b2": 0.2}
    assert delta.metrics(examples, pairs, labels, pred)["pair_accuracy"] == 0.5
