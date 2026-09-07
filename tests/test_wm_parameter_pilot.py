"""Limited pilot forbids heldout targets and non-whitelisted feature content."""

import copy
import json

import pytest

from tools.outcome_prediction.wm_parameter_pilot import (
    declared_closure,
    load_train_inventory,
    parameter_features,
    prepare_cohort,
    recipe_features,
    run_pilot,
)


def fixture(n=8):
    rows, nodes, partition = [], {}, {}
    for i in range(n):
        cell = f"train-{i}"
        partition[cell] = "train"
        for j in range(2):
            identity = f"{cell}/exp-{j}"
            rows.append(
                {
                    "example_id": identity,
                    "cell_id": cell,
                    "role": "training",
                    "audit": {"eligible": True},
                    "label": {"accuracy": (i + j) / (n + 1)},
                    "model_input": {
                        "task": {"base_model": "Qwen/Qwen3-4B-Base", "benchmark": "aime2025"},
                        "plan": {
                            "setup": {
                                "method": {
                                    "hyperparams": {
                                        "lr": 1e-5 * (j + 1),
                                        "epochs": 1,
                                        "top_p": 0.95,
                                    }
                                }
                            }
                        },
                    },
                }
            )
            rows[-1]["model_input"]["plan"]["setup"]["data"] = [
                {"source": "open-r1/OpenR1-Math-220k"}
            ]
            nodes[identity] = {
                "cell_id": cell,
                "base_model": "Qwen/Qwen3-4B-Base",
                "closure_has_unresolved_dependencies": False,
                "operation_expansion_required": False,
                "topological_closure": [identity],
                "parents": [
                    {
                        "kind": "weights",
                        "dependency_scope": "requires_producer_weights",
                        "resolution_status": "declared_base_model",
                        "producer_id": None,
                    }
                ],
            }
    partition["heldout"] = "test"
    return rows, {"cell_partition": partition}, {"nodes": nodes}


class PoisonTestRow(dict):
    def __getitem__(self, key):
        assert key == "cell_id", f"Forbidden TEST access: {key}"
        return super().__getitem__(key)

    def get(self, key, default=None):
        raise AssertionError(f"Forbidden TEST get: {key}")


def test_test_rows_filtered_before_any_feature_or_label_access():
    rows, split, graph = fixture()
    expected = prepare_cohort(rows, split, graph)
    rows.append(PoisonTestRow(cell_id="heldout", label=object(), model_input=object()))
    assert prepare_cohort(rows, split, graph) == expected


def test_test_json_label_not_decoded_even_before_identity_field(tmp_path):
    rows, split, _ = fixture(1)
    path = tmp_path / "inventory.jsonl"
    # Deliberately invalid label value: loading this TEST object would fail.
    heldout = '{"label": NO_DECODE_ALLOWED, "cell_id": "heldout", "model_input": {"x": 3}}\n'
    path.write_text(heldout + json.dumps(rows[0]) + "\n")
    loaded, skipped = load_train_inventory(path, split)
    assert loaded == rows[:1]
    assert skipped == {"heldout": 1}


def test_whitelist_invariant_to_scores_prose_code_counts_progress_and_ids():
    rows, _, graph = fixture(1)
    source = rows[0]
    expected = parameter_features(source)
    source["label"] = {"accuracy": 0.99}
    source["scientist_model"] = "SECRET_SCIENTIST"
    source["card_id"] = "SECRET_CARD"
    source["prior_observations"] = [{"accuracy": 0.8}]
    source["model_input"].update(
        code=[{"content": "parent accuracy=0.9"}], known_previous_checkpoints={"accuracy": 0.9}
    )
    source["model_input"]["plan"]["hypothesis"] = {"claim": "parent scored 0.9"}
    setup = source["model_input"]["plan"]["setup"]
    setup.update(
        data=[{"n_examples": 999999, "source": "SECRET_DATA"}],
        progress={"total": 300},
        budget={"planned_h": 9},
    )
    setup["method"]["hyperparams"].update(
        other="accuracy=0.9", precision="bf16 because parent scored 0.9"
    )
    assert parameter_features(source) == expected
    features = recipe_features([source["example_id"]], {source["example_id"]: source}, graph)
    assert "SECRET_" not in json.dumps(features)
    assert source["example_id"] not in json.dumps(features)


def test_final_zero_label_valid_and_ancestor_labels_not_features():
    rows, split, graph = fixture(1)
    cohort, _ = prepare_cohort(rows, split, graph)
    assert cohort[0]["target"] == 0.0
    rows[0]["label"] = object()
    features = recipe_features([rows[0]["example_id"]], {r["example_id"]: r for r in rows}, graph)
    assert features["step_0.lr"] == 1e-5


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"card_only": True}, "card_only_dependency"),
        ({"dependency_scope": "data_builder_only"}, "data_builder_only_requires_safe_expansion"),
        ({"kind": "generated_data"}, "external_data_operation_not_modeled"),
        ({"artifact_variant": "checkpoint-100"}, "artifact_variant_training_prefix_unresolved"),
        ({"resolution_status": "unresolved"}, "unresolved_edge"),
    ],
)
def test_unsafe_typed_dependencies_excluded(change, reason):
    rows, _, graph = fixture(1)
    parent, target = [r["example_id"] for r in rows]
    node = graph["nodes"][target]
    node["topological_closure"] = [parent, target]
    node["parents"] = [
        {
            "kind": "weights",
            "dependency_scope": "requires_producer_weights",
            "resolution_status": "time_qualified_declared_artifact",
            "producer_id": parent,
            "artifact_variant": "final",
            **change,
        }
    ]
    _, reasons = declared_closure(target, {r["example_id"]: r for r in rows}, graph)
    assert reason in reasons


def test_complete_ordered_closure_required_and_preserved():
    rows, _, graph = fixture(1)
    parent, target = [r["example_id"] for r in rows]
    node = graph["nodes"][target]
    node["topological_closure"] = [parent, target]
    node["parents"] = [
        {
            "kind": "weights",
            "dependency_scope": "requires_producer_weights",
            "resolution_status": "time_qualified_declared_artifact",
            "producer_id": parent,
            "artifact_variant": "final",
        }
    ]
    indexed = {r["example_id"]: r for r in rows}
    closure, reasons = declared_closure(target, indexed, graph)
    assert not reasons and closure == [parent, target]
    features = recipe_features(closure, indexed, graph)
    assert features["step_0.lr"] == 1e-5 and features["step_1.lr"] == 2e-5
    node["topological_closure"] = [target]
    assert "incomplete_or_ambiguous_declared_closure" in declared_closure(target, indexed, graph)[1]


def test_cv_disjoint_train_runs_and_metrics_reproducible_without_test_use():
    rows, split, graph = fixture()
    before = copy.deepcopy(rows)
    rows.append(PoisonTestRow(cell_id="heldout"))
    result = run_pilot(rows, split, graph)
    assert rows[:-1] == before
    assert result["test_scored"] is False
    assert len(result["predictions_private"]) == 16
    for fold in result["folds_private"]:
        assert set(fold["train_cells"]).isdisjoint(fold["validation_cells"])
        assert "heldout" not in fold["train_cells"] + fold["validation_cells"]
    for metrics in result["metrics"]["aime2025"].values():
        assert metrics["runs"] == 8
        assert metrics["selection_runs_with_two_or_more_candidates"] == 8
        assert 0 <= metrics["selection_regret"] <= 1
    assert run_pilot(rows, split, graph) == result


def test_fold_feature_vocabulary_fitted_only_on_its_training_rows():
    rows, split, graph = fixture()
    rows[0]["model_input"]["plan"]["setup"]["method"]["hyperparams"]["temperature"] = 0.7
    result = run_pilot(rows, split, graph)
    fold = next(f for f in result["folds_private"] if rows[0]["cell_id"] in f["validation_cells"])
    assert not any(
        "temperature" in feature
        for feature in fold["fold_train_feature_vocabularies"]["ridge_full"]["feature_names"]
    )


def test_r0_11_exp04_self_generated_data_rejected_despite_empty_graph_parents():
    rows, split, graph = fixture(1)
    source = rows[0]
    source["model_input"]["plan"]["setup"]["data"] = [
        {
            "source": "synthetic:self (RFT rounds 1+2 from sft_out_e3 and sft_out_rft) + HF openai/gsm8k train gold",
            "build_command": ["python", "merge_data.py", "--rft", "rft_data.jsonl,rft_data2.jsonl"],
        }
    ]
    graph["nodes"][source["example_id"]]["parents"] = []
    cohort, omitted = prepare_cohort(rows, split, graph)
    assert source["example_id"] not in {r["example_id"] for r in cohort}
    reasons = next(r["reasons"] for r in omitted if r["example_id"] == source["example_id"])
    assert "internal_self_generated_data_operation_unmodeled" in reasons
    assert "data_builder_model_or_rft_dependency_unmodeled" in reasons
    assert "missing_declared_weight_origin" in reasons
