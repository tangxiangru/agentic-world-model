import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.outcome_prediction import wm_final_model as wm
from tools.outcome_prediction import wm_recipe_input as ri


def payload(rate=1e-5, *, benchmark="aime2025", tag="example"):
    steps = []
    for index in range(2):
        step = {
            "step_id": f"exp{index + 1:02d}",
            "role": "training",
            "parents": [],
            "plan": {
                "hypothesis": {"claim": f"Preserve reasoning features {tag}"},
                "setup": {
                    "lr": rate * (index + 1),
                    "epochs": 2,
                    "warmup": 0,
                    "command": f"python train.py --learning-rate {rate} --seed 42",
                },
            },
            "code": [
                {
                    "role": "training",
                    "script_path": "train.py",
                    "status": "reconstructed",
                    "content": "weight_decay = 0.0123\nn = 8604\n",
                }
            ],
        }
        if index:
            step["parents"] = [
                {"step_id": "exp01", "kind": "weights", "artifact": "ckpts/exp01/checkpoint-606"}
            ]
        steps.append(step)
    return {
        "schema": ri.SCHEMA,
        "task": {"benchmark": benchmark, "base_model": "fixture/base", "evaluation_n": 30},
        "recipe": {"steps": steps, "final_step_id": "exp02"},
    }


def approved(p):
    return {
        "status": "approved",
        "payload_sha256": ri.digest(p),
        "step_reviews": {
            step["step_id"]: {
                "payload_sha256": ri.digest(step),
                "reviewer": "synthetic fixture only",
                "evidence_sha256": "a" * 64,
                "outcome_free": True,
                "executable_recipe_preserved": True,
            }
            for step in p["recipe"]["steps"]
        },
    }


def fixture_rows(benchmark="aime2025"):
    rows = []
    for run in range(4):
        for recipe in range(2):
            p = payload((recipe + 1) * 1e-5, benchmark=benchmark, tag=f"uniquetag{run}")
            rows.append(
                {
                    "example_id": f"{benchmark}-run-{run}/target-{recipe}",
                    "cell_id": f"{benchmark}-run-{run}",
                    "model_input": p,
                    "content_review": approved(p),
                    "label": {"accuracy": recipe * 0.2 + run * 0.01},
                }
            )
    return rows


def split(rows):
    return {"train_cell_ids": sorted({r["cell_id"] for r in rows}), "test_cell_ids": ["test-run"]}


@pytest.fixture(scope="module")
def fitted():
    rows = fixture_rows()
    return wm.train(rows, split(rows))


def test_numeric_and_ordered_ancestor_features_preserved():
    p = payload()
    document = wm.feature_document(p)
    for value in ("1e-05", "2e-05", "0.0123", "8604", "checkpoint-606"):
        assert value in document
    features = wm.structured_features(p)
    assert features["step[000].plan.setup.lr"] == 1e-5
    assert features["step[001].plan.setup.lr"] == 2e-5
    assert features["step[000].plan.setup.warmup"] == 0
    assert features["step[000].plan.setup.warmup.present"] == 1
    assert features["step[001].depth"] == 1
    assert features["step[001].parent.weights=step_000"] == 1
    assert any(value == 0.0123 for value in features.values())
    changed = copy.deepcopy(p)
    changed["recipe"]["steps"][0]["plan"]["setup"]["lr"] = 5e-5
    assert wm.structured_features(changed) != features
    vectorizer = wm._vectorizer().fit([document])
    assert "1e-05" in vectorizer.vocabulary_
    assert "0.0123" in vectorizer.vocabulary_


@pytest.mark.parametrize(
    ("field_path", "feature_path"),
    [
        (("data", 0, "n_examples"), "step[000].plan.setup.data[0].n_examples"),
        (("progress", "total"), "step[000].plan.setup.progress.total"),
    ],
)
def test_omitted_observed_metadata_is_not_encoded_as_a_zero_budget(field_path, feature_path):
    original = payload()
    setup = original["recipe"]["steps"][0]["plan"]["setup"]
    setup["data"] = [{"source": "synthetic fixture", "n_examples": 0}]
    setup["progress"] = {"unit": "optimizer_step", "total": 0}
    explicit = wm.structured_features(original)
    assert explicit[feature_path] == 0.0
    assert explicit[feature_path + ".present"] == 1.0

    absent = copy.deepcopy(original)
    parent = absent["recipe"]["steps"][0]["plan"]["setup"]
    for part in field_path[:-1]:
        parent = parent[part]
    del parent[field_path[-1]]
    ri.validate_full_recipe(absent)
    missing = wm.structured_features(absent)
    assert feature_path not in missing
    assert feature_path + ".present" not in missing
    assert wm.feature_document(absent) != wm.feature_document(original)

    # Only metadata is absent: the scientific recipe and ancestor DAG survive.
    assert missing["step[000].plan.setup.lr"] == explicit["step[000].plan.setup.lr"]
    assert missing["step[001].parent.weights=step_000"] == 1.0
    assert absent["recipe"]["steps"][0]["code"] == original["recipe"]["steps"][0]["code"]
    assert (
        absent["recipe"]["steps"][0]["plan"]["setup"]["command"]
        == original["recipe"]["steps"][0]["plan"]["setup"]["command"]
    )


def test_step_id_relabeling_is_simultaneous_and_structure_preserving():
    p = payload()
    p["recipe"]["steps"][0]["step_id"] = "step_001"
    p["recipe"]["steps"][1]["step_id"] = "step_000"
    p["recipe"]["steps"][1]["parents"][0]["step_id"] = "step_001"
    p["recipe"]["final_step_id"] = "step_000"
    canonical = wm.canonical_payload(p)
    ri.validate_full_recipe(canonical)
    assert canonical["recipe"]["steps"][1]["parents"][0]["step_id"] == "step_000"
    assert canonical["recipe"]["final_step_id"] == "step_001"


def test_numeric_step_ids_never_rewrite_coincident_hyperparameter_values():
    p = payload()
    p["recipe"]["steps"][0]["step_id"] = "1"
    p["recipe"]["steps"][1]["parents"][0]["step_id"] = "1"
    p["recipe"]["steps"][0]["plan"]["setup"]["command"] = "train --epochs 1 --lr 1e-5"
    p["recipe"]["steps"][0]["plan"]["setup"]["numeric_string"] = "1"
    canonical = wm.canonical_payload(p)
    assert canonical["recipe"]["steps"][0]["plan"]["setup"]["numeric_string"] == "1"
    assert (
        canonical["recipe"]["steps"][0]["plan"]["setup"]["command"] == "train --epochs 1 --lr 1e-5"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing-review",
        "not-approved",
        "payload-changed",
        "step-changed",
        "legacy",
        "ancestor-result",
    ],
)
def test_unapproved_or_legacy_inputs_refused_before_fit(change):
    rows = fixture_rows()
    row = rows[0]
    if change == "missing-review":
        del row["content_review"]
    elif change == "not-approved":
        row["content_review"]["status"] = "pending"
    elif change == "payload-changed":
        row["model_input"]["recipe"]["steps"][0]["plan"]["setup"]["lr"] = 8e-5
    elif change == "step-changed":
        row["content_review"]["step_reviews"]["exp01"]["payload_sha256"] = "0" * 64
    elif change == "legacy":
        row["model_input"] = {"task": {}, "plan": {}, "prior_observations": []}
    else:
        row["model_input"]["recipe"]["steps"][0]["plan"]["setup"]["results"] = {"accuracy": 0}
        row["content_review"] = approved(row["model_input"])
    with pytest.raises((ValueError, TypeError)):
        wm.train(rows, split(rows))


def test_heldout_rows_filtered_before_any_feature_score_or_review_access(monkeypatch):
    class Poison(dict):
        def __getitem__(self, key):
            if key != "cell_id":
                raise AssertionError("Held-out row was inspected")
            return "test-run"

        def get(self, *args):
            raise AssertionError("Held-out row was inspected")

    rows = fixture_rows()
    expected = wm.train(rows, split(rows))
    actual = wm.train([Poison(), *rows], split(rows))
    assert (
        actual.training_manifest["training_data_sha256"]
        == expected.training_manifest["training_data_sha256"]
    )
    assert (
        actual.training_manifest["benchmark_train_cv"]
        == expected.training_manifest["benchmark_train_cv"]
    )


@pytest.mark.parametrize(
    "change",
    ["overlap", "unknown", "missing-train", "duplicate-target", "invalid-label", "missing-label"],
)
def test_invalid_partition_or_training_targets_rejected(change):
    rows = fixture_rows()
    partition = split(rows)
    if change == "overlap":
        partition["test_cell_ids"].append(partition["train_cell_ids"][0])
    elif change == "unknown":
        rows.append({"cell_id": "undeclared"})
    elif change == "missing-train":
        partition["train_cell_ids"].append("not-present")
    elif change == "duplicate-target":
        rows[1]["example_id"] = rows[0]["example_id"]
    elif change == "invalid-label":
        rows[0]["label"]["accuracy"] = True
    else:
        rows[0]["label"]["accuracy"] = None
    with pytest.raises(ValueError):
        wm.train(rows, partition)


def test_insufficient_train_groups_does_not_silently_reduce_cv():
    rows = fixture_rows()[:6]
    with pytest.raises(ValueError, match="four"):
        wm.train(rows, split(rows))


def test_explicit_empty_training_runs_keep_the_frozen_split_unchanged():
    rows = fixture_rows()
    partition = split(rows)
    partition["train_cell_ids"].append("known-empty-run")
    partition["empty_train_cell_ids"] = ["known-empty-run"]
    model = wm.train(rows, partition)
    manifest = model.training_manifest
    assert "known-empty-run" in manifest["train_cell_ids"]
    assert "known-empty-run" not in manifest["fitted_train_cell_ids"]
    assert manifest["empty_train_cell_ids"] == ["known-empty-run"]
    assert manifest["forbidden_cell_ids"] == ["test-run"]


@pytest.mark.parametrize(
    "empty", [["test-run"], ["unlisted"], ["aime2025-run-0"], ["aime2025-run-0"] * 2]
)
def test_false_empty_training_declarations_rejected(empty):
    rows = fixture_rows()
    partition = {**split(rows), "empty_train_cell_ids": empty}
    with pytest.raises(ValueError):
        wm.train(rows, partition)


def test_fold_assignment_uses_cell_ids_not_scores_or_features():
    rows = fixture_rows()
    expected = wm._folds(rows)
    for row in rows:
        row["label"]["accuracy"] = 1 - row["label"]["accuracy"]
        row["model_input"] = "not inspected"
    assert wm._folds(rows) == expected
    assert set(expected.values()) == {0, 1, 2, 3}


def test_cv_feature_fit_scope_and_champion_selection(fitted):
    report = fitted.training_manifest["benchmark_train_cv"]["aime2025"]
    assert len(report["fold_audits"]) == 12
    all_ids = set(report["ordered_training_ids"])
    for fold in report["fold_audits"]:
        train_ids, val_ids = set(fold["fit_example_ids"]), set(fold["validation_example_ids"])
        assert train_ids.isdisjoint(val_ids) and train_ids | val_ids == all_ids
        assert {i.split("/")[0] for i in train_ids}.isdisjoint(i.split("/")[0] for i in val_ids)
    metrics = report["candidate_metrics"]
    expected = min(
        wm.VARIANTS,
        key=lambda name: (
            metrics[name]["run_macro_selection_regret"],
            metrics[name]["run_macro_mae"],
            wm.VARIANTS.index(name),
        ),
    )
    assert report["champion"] == expected
    assert not fitted.training_manifest["test_data_used"]
    assert not fitted.training_manifest["five_percentage_point_improvement_demonstrated"]


@pytest.mark.parametrize("name", wm.VARIANTS)
def test_variant_preprocessors_fit_only_training_payloads(name):
    train_payloads = [
        payload(tag="trainwordalpha"),
        payload(2e-5, tag="trainwordbeta"),
        payload(3e-5),
    ]
    model = wm._Variant(name).fit(train_payloads, np.array([0.1, 0.2, 0.3]), np.ones(3))
    heldout = payload(9e-3, tag="unseenheldoutword")
    heldout["recipe"]["steps"][0]["plan"]["setup"]["heldout_setting"] = 99999
    if hasattr(model, "text"):
        assert "unseenheldoutword" not in model.text.vocabulary_
        before = dict(model.text.vocabulary_)
    if hasattr(model, "structured"):
        assert not any("heldout_setting" in key for key in model.structured.vocabulary_)
        scaler_before = model.scaler.mean_.copy()
    prediction = model.predict([heldout])
    assert prediction.shape == (1,) and 0 <= prediction[0] <= 1
    if hasattr(model, "text"):
        assert before == model.text.vocabulary_
    if hasattr(model, "structured"):
        np.testing.assert_array_equal(scaler_before, model.scaler.mean_)


def test_equal_run_weights_and_all_recipes_selection_regret():
    rows = [
        {"cell_id": "one", "example_id": "one/a"},
        {"cell_id": "one", "example_id": "one/b"},
        {"cell_id": "two", "example_id": "two/c"},
    ]
    weights = wm._weights(rows)
    assert weights[0] + weights[1] == weights[2]
    metrics = wm._selection_metrics(rows, np.array([0.0, 0.8, 0.2]), np.array([0.5, 0.4, 0.2]))
    assert metrics["run_macro_selection_regret"] == pytest.approx(0.4)
    assert metrics["run_macro_mae"] == pytest.approx(0.225)
    assert metrics["multi_recipe_runs"] == 1


def test_prediction_has_final_training_neighbors_not_ancestor_observations(fitted):
    prediction = fitted.predict(payload(), neighbors=2)
    assert 0 <= prediction["predicted_official_accuracy"] <= 1
    assert prediction["diagnostic_uncertainty"]["calibrated"] is False
    assert len(prediction["nearest_training_examples"]) == 2
    for neighbor in prediction["nearest_training_examples"]:
        assert set(neighbor) == {
            "example_id",
            "cell_id",
            "final_official_accuracy",
            "text_similarity",
        }
    assert "prior_observations" not in json.dumps(prediction)
    assert "known_previous_checkpoints" not in json.dumps(prediction)
    with pytest.raises(ValueError, match="benchmark"):
        fitted.predict(payload(benchmark="unknown"))


def test_benchmarks_fit_and_retrieve_separately():
    rows = fixture_rows("aime2025") + fixture_rows("gsm8k")
    model = wm.train(rows, split(rows))
    assert set(model.models) == {"aime2025", "gsm8k"}
    assert set(model.training_manifest["benchmark_train_cv"]) == {"aime2025", "gsm8k"}
    neighbors = model.predict(payload(benchmark="gsm8k"))["nearest_training_examples"]
    assert all(item["cell_id"].startswith("gsm8k-") for item in neighbors)


def test_hash_pinned_roundtrip_and_source_contract(fitted, tmp_path, monkeypatch):
    path = tmp_path / "wm.joblib"
    model_hash = fitted.save(path)
    loaded = wm.FinalRecipeWorldModel.load(path, expected_sha256=model_hash)
    assert loaded.predict(payload()) == fitted.predict(payload())
    with pytest.raises(FileExistsError):
        fitted.save(path)
    with pytest.raises(ValueError, match="hash"):
        wm.FinalRecipeWorldModel.load(path, expected_sha256="0" * 64)
    monkeypatch.setattr(wm, "feature_contract_hash", lambda: "0" * 64)
    with pytest.raises(ValueError, match="contract"):
        wm.FinalRecipeWorldModel.load(path, expected_sha256=model_hash)


def test_source_contract_changes_when_numeric_spec_changes(monkeypatch):
    before = wm.feature_contract_hash()
    monkeypatch.setitem(wm.SPEC, "ridge_alpha", 99)
    assert wm.feature_contract_hash() != before


def test_serve_requires_approved_fixed_payloads(monkeypatch, fitted):
    from tools.outcome_prediction import wm_evidence

    captured = {}

    class Store:
        def __init__(self, root, files, *, audit_log, candidates, predict):
            captured.update({"candidates": candidates, "predict": predict})

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(wm.FinalRecipeWorldModel, "load", lambda *args, **kwargs: fitted)
    monkeypatch.setattr(wm_evidence, "EvidenceStore", Store)
    monkeypatch.setattr(wm_evidence, "serve_stdio", lambda store: None)
    p = payload()
    config = {
        "model_path": "unused",
        "model_sha256": "a" * 64,
        "candidate_payloads": {"candidate-a": {"model_input": p, "content_review": approved(p)}},
        "evidence_root": "unused",
        "files": {},
        "audit_log": "unused",
    }
    wm.serve(config)
    assert captured["candidates"] == ("candidate-a",)
    assert 0 <= captured["predict"]("candidate-a")["predicted_official_accuracy"] <= 1
    with pytest.raises(KeyError):
        captured["predict"]("unapproved")
    config["candidate_payloads"]["candidate-a"] = p
    with pytest.raises(ValueError, match="review"):
        wm.serve(config)


def test_module_serve_loads_imported_model_and_initializes_mcp(fitted, tmp_path):
    """Exercise the actual -m boundary without Claude or any prediction request."""
    model_path = tmp_path / "wm.joblib"
    model_hash = fitted.save(model_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    audit = tmp_path / "audit.jsonl"
    p = payload(tag="synthetic-startup-only")
    config = {
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "candidate_payloads": {
            "synthetic-candidate": {"model_input": p, "content_review": approved(p)}
        },
        "evidence_root": str(evidence),
        "files": {},
        "audit_log": str(audit),
    }
    config_path = tmp_path / "server.json"
    config_path.write_text(json.dumps(config))
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "synthetic-test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    wire = "\n".join(json.dumps(message) for message in messages) + "\n"
    repository = Path(wm.__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.outcome_prediction.wm_final_model",
            "serve",
            "--config",
            str(config_path),
        ],
        input=wire,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={"PATH": os.defpath, "PYTHONPATH": str(repository), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[0]["result"]["capabilities"] == {"tools": {}}
    tool_specs = responses[1]["result"]["tools"]
    assert [tool["name"] for tool in tool_specs] == [
        "list_evidence",
        "read_evidence",
        "search_evidence",
        "predict_candidate",
    ]
    assert tool_specs[-1]["inputSchema"]["properties"]["candidate_id"]["enum"] == [
        "synthetic-candidate"
    ]
    assert audit.read_bytes() == b""  # Initialize/list never dispatch or predict.
    assert wm._sha(model_path.read_bytes()) == model_hash


def test_training_api_does_not_apply_optional_terminal_only_supervision_filter():
    # Prefix targets are permitted as separate full recipes unless the user
    # explicitly chooses terminal-only supervision. Inputs never get their scores.
    rows = fixture_rows()
    for row in rows[::2]:
        p = row["model_input"]
        p["recipe"]["steps"] = p["recipe"]["steps"][:1]
        p["recipe"]["final_step_id"] = "exp01"
        row["content_review"] = approved(p)
    model = wm.train(rows, split(rows))
    assert len(model.training_manifest["training_examples"]) == 8
