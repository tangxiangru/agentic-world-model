"""Synthetic reference-cohort tests; no network or predictor fitting."""

import copy
import hashlib
import io
import json

import pytest

from tools.outcome_prediction import wm_delta_reference_data as data


def public_source():
    payload = {"zeroshot": {key: {bench: score} for bench, (_, key, score) in data.BASES.items()}}
    body = json.dumps(payload)
    return {"url": data.PUBLIC_URL, "body": body, "sha256": hashlib.sha256(body.encode()).hexdigest()}


def fixture():
    targets, labels, decisions, registry = [], {}, {}, {}
    for card, benchmark, cell, score, mode in (
        ("measured", "gsm8k", "train-cell", 0.5, "single"),
        ("base-aime", "aime2025", "train-cell", 0.0, "base"),
        ("base-gsm", "gsm8k", "test-cell", 0.1, "base"),
        ("missing-parent", "gsm8k", "test-cell", 0.8, "single"),
        ("missing-target", "gsm8k", "test-cell", None, "base"),
    ):
        key = cell + "/" + card
        model = data.BASES[benchmark][0]
        targets.append({"example_id": key, "cell_id": cell, "benchmark": benchmark,
                        "model_input": {"plan": {"setup": {}}, "code": []}})
        labels[key] = {"accuracy": score, "official_metric": {"accuracy": score}, "evaluation_n": 30}
        decisions[key] = {"eligible": True, "one_step_eligible": mode == "single" and card == "measured",
                          "checkpoint_inputs": {"status": mode, "inputs": [{"path": model, "base_model": model}]}}
        registry[key] = {"status": "official_archive_correlated", "accuracy": score,
                         "cell_id": cell, "benchmark": benchmark, "archived_setup": {"base_model": model}}
    measured = copy.deepcopy(targets[:1])
    measured[0].update(parent={"accuracy": 0.0, "reference_known": True, "producer_id": "train-cell/parent",
                               "path_kind": "archived_source"}, history=[])
    registry["train-cell/parent"] = {"status": "official_archive_correlated", "accuracy": 0.0,
                                      "cell_id": "train-cell", "benchmark": "gsm8k"}
    measured_labels = {measured[0]["example_id"]: {**labels[measured[0]["example_id"]], "delta_accuracy": 0.5}}
    return [targets, labels, measured, measured_labels, decisions, registry,
            {"train-cell": "train", "test-cell": "test"}, public_source()]


def test_combines_only_known_scores_preserves_zero_and_split():
    args = fixture()
    untouched = copy.deepcopy(args)
    result = data.build_bundle(*args)
    assert args == untouched
    assert result["summary"]["combined"] == {
        "total": 3, "train": 2, "test": 1, "by_benchmark": {"gsm8k": 2, "aime2025": 1},
        "target_zero": 1, "parent_zero": 1,
    }
    assert result["split"]["cell_partition"] == args[-2]
    for row in result["inputs"]:
        label = result["labels"][row["example_id"]]
        assert label["delta_accuracy"] == label["accuracy"] - row["parent"]["accuracy"]
        if row["reference_kind"] == "published_base":
            assert row["parent"]["kind"] == "published_base"
            assert row["parent"]["evaluation_n"] is None
            assert row["history"] == [] and row["history_complete_to_base"] is True
        else:
            assert row["parent"]["kind"] == "measured"
            assert row["parent"]["accuracy"] == 0


@pytest.mark.parametrize("value", [None, False, True, float("nan"), float("inf"), -0.01, 1.01, "0.2"])
def test_invalid_target_or_parent_excluded(value):
    args = fixture()
    args[1]["train-cell/base-aime"]["accuracy"] = value
    args[1]["train-cell/base-aime"]["official_metric"]["accuracy"] = value
    args[2][0]["parent"]["accuracy"] = value
    result = data.build_bundle(*args)
    assert set(result["labels"]) == {"test-cell/base-gsm"}


def test_missing_target_key_excluded_not_backfilled():
    args = fixture()
    del args[1]["train-cell/base-aime"]["accuracy"]
    assert "train-cell/base-aime" not in data.build_bundle(*args)["labels"]


@pytest.mark.parametrize("change", ["model", "archived_model", "multiple", "split", "cell", "target_score", "parent_score", "delta", "duplicate"])
def test_identity_conflicts_fail_closed(change):
    args = fixture()
    key = "train-cell/base-aime"
    if change == "model":
        args[4][key]["checkpoint_inputs"]["inputs"][0]["base_model"] = "Qwen/Qwen3-4B"
    elif change == "archived_model":
        args[5][key]["archived_setup"]["base_model"] = "wrong/base"
    elif change == "multiple":
        args[4][key]["checkpoint_inputs"]["inputs"].append({"path": "another"})
    elif change == "split":
        args[6]["train-cell"] = "validation"
    elif change == "cell":
        args[5][key]["cell_id"] = "test-cell"
    elif change == "target_score":
        args[5][key]["accuracy"] = 0.1
    elif change == "parent_score":
        args[5]["train-cell/parent"]["accuracy"] = 0.1
    elif change == "delta":
        args[3]["train-cell/measured"]["delta_accuracy"] = 0.4
    elif change == "duplicate":
        args[0].append(copy.deepcopy(args[0][0]))
    with pytest.raises(ValueError):
        data.build_bundle(*args)


@pytest.mark.parametrize("change", ["url", "body", "value"])
def test_public_provenance_and_expected_values_pinned(change):
    source = public_source()
    if change == "url":
        source["url"] = "https://unrelated.invalid/baselines.json"
    elif change == "body":
        source["body"] += " "
    else:
        body = json.loads(source["body"])
        body["zeroshot"]["gemma-3-4b-pt"]["gsm8k"] = 0.9
        source["body"] = json.dumps(body)
        source["sha256"] = hashlib.sha256(source["body"].encode()).hexdigest()
    with pytest.raises(ValueError):
        data.validate_public_source(source)


def frozen_fixture(tmp_path, monkeypatch):
    args = fixture()
    source = tmp_path / "source"
    (source / "private").mkdir(parents=True)
    for name, value in (("private/decisions.json", args[4]), ("private/registry.json", args[5]),
                        ("split.json", {"cell_partition": args[6]}), ("manifest.json", {"fixture": True})):
        data.original.write(source / name, value)

    def load_original(directory, partition, *, cohort):
        assert directory == source
        rows, labels = args[:2] if cohort == "target" else args[2:4]
        chosen = [e for e in rows if args[6][e["cell_id"]] == partition]
        return copy.deepcopy(chosen), {e["example_id"]: copy.deepcopy(labels[e["example_id"]]) for e in chosen}

    monkeypatch.setattr(data.original, "load_partition", load_original)
    monkeypatch.setattr(data, "urlopen", lambda url, timeout: io.BytesIO(args[-1]["body"].encode()))
    output = tmp_path / "expanded"
    data.freeze(source, output)
    return source, output


def replace_fixture(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_frozen_loader_cohorts_and_no_overwrite(tmp_path, monkeypatch):
    source, output = frozen_fixture(tmp_path, monkeypatch)
    assert len(data.load_partition(output, "train")[0]) == 2
    assert len(data.load_partition(output, "test")[0]) == 1
    assert len(data.load_partition(output, "train", cohort="measured_parent")[0]) == 1
    assert len(data.load_partition(output, "test", cohort="measured_parent")[0]) == 0
    assert len(data.load_partition(output, "train", cohort="published_base")[0]) == 1
    with pytest.raises(FileExistsError):
        data.freeze(source, output)


@pytest.mark.parametrize("change", ["missing_label", "wrong_delta", "split", "kind", "source", "missing_dependency", "policy"])
def test_loader_rejects_tampering_even_if_artifact_hash_updated(tmp_path, monkeypatch, change):
    source, output = frozen_fixture(tmp_path, monkeypatch)
    if change == "source":
        replace_fixture(source / "manifest.json", {"changed": True})
    elif change == "missing_dependency":
        manifest = data.original.read(output / "manifest.json")
        del manifest["sources"][str(source / "manifest.json")]
        replace_fixture(output / "manifest.json", manifest)
    else:
        filename = {"missing_label": "labels.json", "wrong_delta": "labels.json", "split": "split.json",
                    "kind": "inputs.json", "policy": "policy.json"}[change]
        value = data.original.read(output / filename)
        if change == "missing_label":
            del value["train-cell/base-aime"]["accuracy"]
        elif change == "wrong_delta":
            value["train-cell/base-aime"]["delta_accuracy"] = 0
        elif change == "split":
            value["cell_partition"]["train-cell"] = "test"
        elif change == "kind":
            value[0]["reference_kind"] = "published_base"
        else:
            value["no_missing_score_imputation"] = False
        replace_fixture(output / filename, value)
        manifest = data.original.read(output / "manifest.json")
        manifest["files"][filename] = data.original.sha(output / filename)
        replace_fixture(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        data.load_partition(output, "train")
