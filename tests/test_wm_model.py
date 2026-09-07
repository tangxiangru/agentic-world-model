import copy

import pytest

from tools.outcome_prediction import wm_model
from tools.outcome_prediction.wm_model import RecipeWorldModel, feature_document, public_payload


def row(cell, number, accuracy, benchmark="gsm8k"):
    return {
        "example_id": f"{cell}/exp-{number:02d}",
        "cell_id": cell,
        "label": {"accuracy": accuracy},
        "audit": {"eligible": True},
        "model_input": {
            "task": {"benchmark": benchmark, "evaluation_n": 1319 if benchmark == "gsm8k" else 30},
            "plan": {"setup": {"method": "sft", "lr": 1e-5 * number, "data": "math examples"}},
            "code": [{"content": f"train(epochs={number})"}],
        },
        "prior_observations": [{"accuracy": 0.3, "n": 30, "protocol": "local"}],
    }


def train_model():
    rows = [row(c, n, n / 10) for c in ("train-a", "train-b") for n in range(1, 7)]
    test = row("test-c", 7, 0.9)
    test["model_input"]["plan"]["setup"]["data"] = "heldoutuniquetoken"
    return RecipeWorldModel().fit(
        [*rows, test], train_cell_ids=["train-a", "train-b"], forbidden_cell_ids=["test-c"]
    ), test


def test_feature_boundary_ignores_target_outcomes_and_metadata():
    original = row("test", 2, 0.1)
    changed = copy.deepcopy(original)
    changed.update(label={"accuracy": 0.99}, audit={"eligible": False}, scientist_model="secret")
    assert public_payload(original) == public_payload(changed)
    assert feature_document(public_payload(original)) == feature_document(public_payload(changed))
    assert "prior_observations" in public_payload(original)


@pytest.mark.parametrize("field", ["result", "conclusion", "label", "y", "audit"])
def test_obvious_outcomes_in_public_payload_rejected(field):
    target = row("test", 2, 0.1)
    target["model_input"][field] = {"accuracy": 0.9}
    with pytest.raises(ValueError, match="Outcome/audit"):
        public_payload(target)


def test_post_execution_plan_section_rejected():
    target = row("test", 2, 0.1)
    target["model_input"]["plan"]["result"] = {"accuracy": 0.9}
    with pytest.raises(ValueError, match="first four"):
        public_payload(target)


def test_feature_document_preserves_dataset_sources_and_decimal_settings():
    payload = public_payload(row("a", 2, 0.1))
    payload["plan"]["setup"]["data"] = [{"source": "meta-math/MetaMathQA", "lr": 0.00001}]
    document = feature_document(payload)
    assert "meta-math/MetaMathQA" in document
    assert "1e-05" in document


def test_scientific_evidence_retained_while_reconstruction_provenance_removed():
    payload = public_payload(row("a", 2, 0.1))
    payload["plan"]["problem"] = {"evidence": [{"observation": "prior stopping failure"}]}
    payload["plan"]["setup"]["data"] = {"source_path": "scientific-data.jsonl"}
    payload["code"][0]["evidence"] = {"provenance_token": "not-a-feature"}
    document = feature_document(payload)
    assert "prior stopping failure" in document
    assert "scientific-data.jsonl" in document
    assert "not-a-feature" not in document


def test_whole_test_run_never_enters_fit_or_vocabulary():
    model, test = train_model()
    assert model.training_manifest["fitted_train_cell_ids"] == ["train-a", "train-b"]
    assert "heldoutuniquetoken" not in model.models["gsm8k"]["vectorizer"].vocabulary_
    prediction = model.predict(public_payload(test))
    assert 0 <= prediction["predicted_official_accuracy"] <= 1
    assert all(n["cell_id"] != "test-c" for n in prediction["nearest_training_examples"])
    test["label"]["accuracy"] = 0.0
    assert prediction == model.predict(public_payload(test))


def test_overlapping_run_split_fails():
    with pytest.raises(ValueError, match="disjoint"):
        RecipeWorldModel().fit(
            [row("same", 1, 0.1)], train_cell_ids=["same"], forbidden_cell_ids=["same"]
        )


def test_zero_is_valid_and_unlabeled_and_ineligible_are_excluded():
    rows = [row("a", 1, 0.0), row("a", 2, None), row("a", 3, 0.9)]
    rows[2]["audit"]["eligible"] = False
    model = RecipeWorldModel().fit(rows, train_cell_ids=["a"])
    assert len(model.training_manifest["training_examples"]) == 1
    assert model.predict(public_payload(rows[0]))["predicted_official_accuracy"] == 0.0


def test_no_implicit_cross_domain_fallback():
    model, _ = train_model()
    with pytest.raises(ValueError, match="No trained WM"):
        model.predict(public_payload(row("new", 1, 0.1, benchmark="aime2025")))


def test_model_artifact_requires_hash_and_refuses_overwrite(tmp_path):
    model, test = train_model()
    path = tmp_path / "wm.joblib"
    fingerprint = model.save(path)
    restored = RecipeWorldModel.load(path, expected_sha256=fingerprint)
    assert restored.predict(public_payload(test)) == model.predict(public_payload(test))
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        RecipeWorldModel.load(path, expected_sha256="0" * 64)


def test_loading_rejects_changed_feature_contract(tmp_path, monkeypatch):
    model, _ = train_model()
    path = tmp_path / "wm.joblib"
    fingerprint = model.save(path)
    monkeypatch.setattr(wm_model, "feature_contract_hash", lambda: "changed-feature-contract")
    with pytest.raises(ValueError, match="feature contract"):
        RecipeWorldModel.load(path, expected_sha256=fingerprint)


def test_server_rejects_extra_config_before_loading_any_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid config must fail before artifact load")

    monkeypatch.setattr(RecipeWorldModel, "load", forbidden)
    with pytest.raises(ValueError, match="config"):
        wm_model.serve({"hidden_labels": "must not be accepted"})
