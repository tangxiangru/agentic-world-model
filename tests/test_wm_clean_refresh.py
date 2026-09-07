"""Synthetic strict-refresh contract tests; no archived scores or network calls."""

import copy
import hashlib
import json

import pytest

from tools.outcome_prediction.wm_clean_refresh import (
    freeze_dataset,
    load_partition,
    project_example,
    screen_row,
)


def clean_row(cell="fixture-01", card="exp-01", accuracy=0.25):
    output = "/task/checkpoints/" + card
    return {
        "example_id": cell + "/" + card,
        "cell_id": cell,
        "card_id": card,
        "benchmark": "aime2025",
        "scientist_model": "fixture-scientist",
        "first_submitted_at": "2026-09-05T13:48:09Z",
        "first_stage": "plan",
        "eligible": True,
        "role": "training",
        "model_input": {
            "task": {"benchmark": "aime2025", "base_model": "fixture/base"},
            "plan": {
                "problem": {"statement": "PROBLEM_SENTINEL"},
                "hypothesis": {"claim": "HYPOTHESIS_SENTINEL"},
                "setup": {
                    "base_model": "fixture/base",
                    "command": {
                        "argv": ["python", "train.py", "--lr", "0.0001"],
                        "script": "/task/train.py",
                        "cwd": "/task",
                        "env": {"PRIVATE_SENTINEL": "discard"},
                        "log": "PRIVATE_LOG_SENTINEL",
                    },
                    "method": {
                        "family": "sft",
                        "framework": "trl",
                        "hyperparams": {"lr": 0.0001, "epochs": 1, "other": "HP_PROSE_SENTINEL"},
                    },
                    "data": [{"source": "public-training-data", "n_examples": 999999}],
                    "parent_checkpoint": {"origin": "base_model", "path": "PARENT_SENTINEL"},
                    "output_dir": output,
                    "progress": {"total": 12345, "unit": "steps"},
                },
                "evaluation": {"protocol": {"n": 30}, "comparator": {"value": 0.99}},
            },
            "code": [
                {
                    "role": "training",
                    "script_path": "/task/train.py",
                    "status": "reconstructed",
                    "content": "learning_rate=0.0001\n",
                    "blockers": ["CODE_AUDIT_SENTINEL"],
                    "sha256": "CODE_HASH_SENTINEL",
                }
            ],
        },
        "label": {
            "accuracy": accuracy,
            "official_metric": {"accuracy": accuracy},
            "stderr": 0.1,
            "evaluation_n": 30,
        },
        "prior_observations": [{"measurements": [{"value": 0.88}], "text": "HISTORY_SENTINEL"}],
        "audit": {
            "eligible": True,
            "eligibility_status": "provisional_plan_stage_screen",
            "reasons": [],
            "warnings": [],
            "first_missing_fields": [],
            "has_official_label": True,
            "label_fidelity_review_required": True,
            "label_fidelity_status": "provisional_unadjudicated",
            "label_fidelity_flags": [],
            "first_final_setup_changed": False,
            "first_final_setup_changed_fields": [],
            "output_artifact_path_mismatch": False,
            "output_artifact_comparison": {
                "first_declared_output_dir": output,
                "final_output_checkpoint": output,
                "comparison": "literal path comparison only",
            },
            "record_errors": [],
            "code_provenance": [],
        },
        "provenance": {
            "first_record_sha256": "1" * 64,
            "card_sha256": "2" * 64,
            "official_metric_sha256": "3" * 64,
            "ledger_sha256": "4" * 64,
            "source_revision": "fixture-revision",
        },
    }


def assert_rejected(row, previous=None):
    result = screen_row(row, previous=previous)
    assert result["eligible"] is False
    assert isinstance(result["reasons"], list) and result["reasons"]
    assert all(isinstance(reason, str) and reason for reason in result["reasons"])
    return result


def test_complete_clean_labeled_row_is_accepted_without_mutation():
    row = clean_row()
    original = copy.deepcopy(row)
    result = screen_row(row)
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert isinstance(result["previous_review_status"], str)
    assert row == original


@pytest.mark.parametrize("accuracy", [0, 0.0, 1 / 30, 0.5, 0.999, 1])
def test_selection_does_not_discard_valid_low_or_zero_scores(accuracy):
    result = screen_row(clean_row(accuracy=accuracy))
    assert result["eligible"] is True
    assert result["reasons"] == []


@pytest.mark.parametrize(
    "label",
    [
        None,
        {},
        {"accuracy": 0.2},
        {"accuracy": 0.2, "official_metric": {}},
        {"accuracy": True, "official_metric": {"accuracy": True}},
        {"accuracy": "0.2", "official_metric": {"accuracy": 0.2}},
        {"accuracy": 0.2, "official_metric": {"accuracy": "0.2"}},
        {"accuracy": 0.2, "official_metric": {"accuracy": 0.3}},
        {"accuracy": -0.1, "official_metric": {"accuracy": -0.1}},
        {"accuracy": 1.1, "official_metric": {"accuracy": 1.1}},
        {"accuracy": float("nan"), "official_metric": {"accuracy": float("nan")}},
        {"accuracy": float("inf"), "official_metric": {"accuracy": float("inf")}},
    ],
)
def test_missing_invalid_or_mismatched_official_labels_are_rejected(label):
    row = clean_row()
    row["label"] = label
    assert_rejected(row)


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_row_eligibility_must_be_explicit_boolean_true(value):
    row = clean_row()
    row["eligible"] = value
    assert_rejected(row)


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_structural_audit_eligibility_must_be_explicit_boolean_true(value):
    row = clean_row()
    row["audit"]["eligible"] = value
    assert_rejected(row)


@pytest.mark.parametrize("audit", [None, {}, [], "unknown"])
def test_missing_or_unknown_audit_fails_closed(audit):
    row = clean_row()
    row["audit"] = audit
    assert_rejected(row)


@pytest.mark.parametrize(
    "field", ["label", "eligible", "audit", "first_submitted_at", "first_stage"]
)
def test_missing_required_screen_fields_fail_closed(field):
    row = clean_row()
    del row[field]
    assert_rejected(row)


@pytest.mark.parametrize(
    "timestamp", [None, "", "invalid", "2026-09-05T13:48:09", 12345, "2026-19-99T00:00:00Z"]
)
def test_first_submission_requires_nonempty_valid_timezone_timestamp(timestamp):
    row = clean_row()
    row["first_submitted_at"] = timestamp
    assert_rejected(row)


@pytest.mark.parametrize("stage", [None, "", "closed", "running", "unknown", "PLAN"])
def test_only_first_plan_stage_is_eligible(stage):
    row = clean_row()
    row["first_stage"] = stage
    assert_rejected(row)


def test_first_plan_missing_fields_are_not_silently_ignored():
    row = clean_row()
    row["audit"]["first_missing_fields"] = ["setup.command.argv"]
    assert_rejected(row)


def test_structural_reasons_cannot_be_overridden_by_true_eligibility_flag():
    row = clean_row()
    row["audit"]["reasons"] = ["first_result_not_empty"]
    assert_rejected(row)


@pytest.mark.parametrize(
    "flag",
    [
        "first_final_setup_changed",
        "declared_output_checkpoint_path_differs",
        "declared_or_produced_output_unknown",
        "final_card_missing",
        "future_unknown_fidelity_flag",
    ],
)
def test_any_label_fidelity_flag_quarantines_the_row(flag):
    row = clean_row()
    row["audit"]["label_fidelity_flags"] = [flag]
    assert_rejected(row)


@pytest.mark.parametrize(
    "comparison",
    [
        None,
        {},
        {"first_declared_output_dir": "/task/checkpoint"},
        {"final_output_checkpoint": "/task/checkpoint"},
        {"first_declared_output_dir": "", "final_output_checkpoint": ""},
        {"first_declared_output_dir": "/task/checkpoint", "final_output_checkpoint": "/task/other"},
    ],
)
def test_missing_or_mismatched_final_artifact_binding_fails_closed(comparison):
    row = clean_row()
    row["audit"]["output_artifact_comparison"] = comparison
    # Other audit booleans remain optimistically false: the literal evidence
    # must still be checked rather than trusting a precomputed empty flag list.
    assert_rejected(row)


def test_missing_binding_object_fails_closed():
    row = clean_row()
    del row["audit"]["output_artifact_comparison"]
    assert_rejected(row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_final_setup_changed", True),
        ("first_final_setup_changed_fields", ["method"]),
        ("output_artifact_path_mismatch", True),
    ],
)
def test_conflicting_fidelity_signals_fail_closed_even_without_flags(key, value):
    row = clean_row()
    row["audit"][key] = value
    assert_rejected(row)


def test_previously_rejected_identical_source_never_becomes_eligible():
    row = clean_row()
    previous = copy.deepcopy(row)
    previous["eligible"] = False
    previous["audit"]["eligible"] = False
    previous["audit"]["reasons"] = ["reviewed_artifact_binding_failure"]
    before = copy.deepcopy(previous)
    result = assert_rejected(row, previous)
    assert result["previous_review_status"]
    assert previous == before


def test_previously_rejected_changed_source_remains_quarantined():
    row = clean_row()
    previous = copy.deepcopy(row)
    previous["audit"]["eligible"] = False
    unchanged = assert_rejected(row, previous)
    row["provenance"]["card_sha256"] = "a" * 64
    changed = assert_rejected(row, previous)
    assert changed["previous_review_status"] != unchanged["previous_review_status"]
    assert any("review" in reason for reason in changed["reasons"])


@pytest.mark.parametrize(
    "fingerprint",
    [
        "first_record_sha256",
        "card_sha256",
        "official_metric_sha256",
        "ledger_sha256",
    ],
)
def test_partial_previous_fingerprints_do_not_claim_source_matched(fingerprint):
    row = clean_row()
    previous = copy.deepcopy(row)
    previous["audit"]["eligible"] = False
    matched = assert_rejected(row, previous)
    del previous["provenance"][fingerprint]
    partial = assert_rejected(row, previous)
    assert partial["previous_review_status"] != matched["previous_review_status"]
    assert any("review" in reason for reason in partial["reasons"])


def test_previous_rejection_does_not_depend_on_score_magnitude():
    previous = clean_row(accuracy=0)
    previous["audit"]["eligible"] = False
    decisions = [assert_rejected(clean_row(accuracy=v), previous) for v in (0, 0.5, 1)]
    assert decisions[0] == decisions[1] == decisions[2]


def test_changed_reconstructed_code_provenance_requires_review_of_previous_rejection():
    row = clean_row()
    row["audit"]["code_provenance"] = [{"trace_sha256": "a" * 64}]
    previous = copy.deepcopy(row)
    previous["audit"]["eligible"] = False
    matched = assert_rejected(row, previous)
    row["audit"]["code_provenance"][0]["trace_sha256"] = "b" * 64
    changed = assert_rejected(row, previous)
    assert changed["previous_review_status"] != matched["previous_review_status"]
    assert any("review" in reason for reason in changed["reasons"])


def test_projection_is_positive_input_only_and_does_not_mutate_inventory():
    row = clean_row()
    original = copy.deepcopy(row)
    projected = project_example(row)
    assert set(projected) == {
        "example_id",
        "cell_id",
        "benchmark",
        "scientist_model",
        "model_input",
    }
    assert set(projected["model_input"]) == {"plan", "code"}
    assert set(projected["model_input"]["plan"]) == {"setup"}
    text = json.dumps(projected)
    for sentinel in (
        "HISTORY_SENTINEL",
        "PROBLEM_SENTINEL",
        "HYPOTHESIS_SENTINEL",
        "PRIVATE_SENTINEL",
        "PRIVATE_LOG_SENTINEL",
        "PARENT_SENTINEL",
        "HP_PROSE_SENTINEL",
        "CODE_AUDIT_SENTINEL",
        "CODE_HASH_SENTINEL",
    ):
        assert sentinel not in text
    for excluded in ("label", "result", "prior_observations", "audit", "provenance"):
        assert excluded not in projected
    assert "n_examples" not in text and "999999" not in text
    assert projected["model_input"]["plan"]["setup"]["method"]["hyperparams"]["lr"] == 0.0001
    assert row == original


def test_projection_drops_unreconstructed_snapshots():
    row = clean_row()
    row["model_input"]["code"].append(
        {
            "role": "training",
            "status": "snapshot",
            "script_path": "/task/later.py",
            "content": "LATER_CODE_SENTINEL",
        }
    )
    assert "LATER_CODE_SENTINEL" not in json.dumps(project_example(row))


def test_projection_invariant_to_outcomes_history_and_posthoc_audit():
    row = clean_row()
    expected = project_example(row)
    row["label"] = {"accuracy": 0.99, "official_metric": {"accuracy": 0.99}}
    row["prior_observations"] = [{"measurements": [{"value": 0.01}]}]
    row["audit"]["reasons"] = ["irrelevant_for_projection"]
    row["provenance"]["card_sha256"] = "f" * 64
    row["model_input"]["plan"]["result"] = {"accuracy": 0.99}
    assert project_example(row) == expected


def test_projection_returns_deep_copies():
    row = clean_row()
    projected = project_example(row)
    projected["model_input"]["plan"]["setup"]["method"]["hyperparams"]["lr"] = 99
    projected["model_input"]["code"][0]["content"] = "changed"
    assert row["model_input"]["plan"]["setup"]["method"]["hyperparams"]["lr"] == 0.0001
    assert row["model_input"]["code"][0]["content"] == "learning_rate=0.0001\n"


def bundle_rows():
    rows = [
        clean_row(f"fixture-{n:02}", f"exp-{j:02}", accuracy=(n + j) / 30)
        for n in range(1, 5)
        for j in range(1, 3)
    ]
    rows[1]["label"] = None
    rows[3]["audit"]["label_fidelity_flags"] = ["first_final_setup_changed"]
    rows[5]["audit"]["eligible"] = False
    return rows


def test_bundle_loaders_only_return_clean_labeled_targets_and_keep_sessions_whole(tmp_path):
    rows = bundle_rows()
    output = tmp_path / "bundle"
    freeze_dataset(rows, output, test_count=1, source_metadata={"fixture": True})
    train, train_labels = load_partition(output, "train")
    test, test_labels = load_partition(output, "test")
    expected = {r["example_id"] for r in rows if screen_row(r)["eligible"]}
    assert set(train_labels) | set(test_labels) == expected
    assert not (set(train_labels) & set(test_labels))
    assert set(train_labels) == {r["example_id"] for r in train}
    assert set(test_labels) == {r["example_id"] for r in test}
    assert not ({r["cell_id"] for r in train} & {r["cell_id"] for r in test})
    assert {r["cell_id"] for r in test}
    for example in train + test:
        assert "label" not in example and "audit" not in example
    assert all(
        type(v) in (int, float) and 0 <= v <= 1 for v in (train_labels | test_labels).values()
    )


def test_zero_official_labels_survive_both_clean_partition_loaders(tmp_path):
    rows = [clean_row(f"fixture-{n:02}", accuracy=0) for n in range(1, 5)]
    output = tmp_path / "bundle"
    freeze_dataset(rows, output, test_count=1)
    for partition in ("train", "test"):
        examples, labels = load_partition(output, partition)
        assert examples and labels
        assert all(value == 0 for value in labels.values())


def test_bundle_keeps_full_inventory_private_but_labels_separate(tmp_path):
    rows = bundle_rows()
    output = tmp_path / "bundle"
    freeze_dataset(rows, output, test_count=1)
    inventory = [
        json.loads(line) for line in (output / "private/inventory.jsonl").read_text().splitlines()
    ]
    assert {r["example_id"] for r in inventory} == {r["example_id"] for r in rows}
    inputs = json.loads((output / "clean_inputs.json").read_text())
    labels = json.loads((output / "clean_labels.json").read_text())
    assert {r["example_id"] for r in inputs} == set(labels)
    assert all("label" not in r and "prior_observations" not in r for r in inputs)
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "private").stat().st_mode & 0o777 == 0o700
    for path in output.rglob("*"):
        if path.is_file():
            assert path.stat().st_mode & 0o777 == 0o600


def test_split_is_identity_based_order_invariant_and_assigned_before_filtering(tmp_path):
    rows = bundle_rows()
    output_a, output_b = tmp_path / "a", tmp_path / "b"
    freeze_dataset(rows, output_a, test_count=1)
    changed = list(reversed(copy.deepcopy(rows)))
    for row in changed:
        row["label"] = {"accuracy": 1.0, "official_metric": {"accuracy": 1.0}}
        row["audit"]["label_fidelity_flags"] = []
        row["audit"]["eligible"] = True
    freeze_dataset(changed, output_b, test_count=1)
    a = json.loads((output_a / "split.json").read_text())
    b = json.loads((output_b / "split.json").read_text())
    assert a["cell_partition"] == b["cell_partition"]
    assert set(a["cell_partition"]) == {r["cell_id"] for r in rows}
    assert len(a["test_cell_ids"]) == 1


def test_previously_rejected_rows_are_not_reintroduced_by_freeze(tmp_path):
    rows = [clean_row(f"fixture-{n:02}") for n in range(1, 5)]
    previous = copy.deepcopy(rows[0])
    previous["audit"]["eligible"] = False
    output = tmp_path / "bundle"
    freeze_dataset(rows, output, previous_rows=[previous], test_count=1)
    labels = {}
    for partition in ("train", "test"):
        labels.update(load_partition(output, partition)[1])
    assert rows[0]["example_id"] not in labels


def test_freeze_refuses_existing_output_and_duplicate_identity(tmp_path):
    rows = bundle_rows()
    output = tmp_path / "bundle"
    freeze_dataset(rows, output, test_count=1)
    with pytest.raises(FileExistsError):
        freeze_dataset(rows, output, test_count=1)
    with pytest.raises(ValueError):
        freeze_dataset(rows + [rows[0]], tmp_path / "duplicate", test_count=1)
    with pytest.raises(ValueError):
        freeze_dataset(
            rows, tmp_path / "duplicate_previous", previous_rows=[rows[0], rows[0]], test_count=1
        )


@pytest.mark.parametrize("partition", ["validation", "all", "", None])
def test_loader_rejects_unknown_partitions(tmp_path, partition):
    with pytest.raises(ValueError):
        load_partition(tmp_path, partition)


@pytest.mark.parametrize(
    "filename",
    [
        "clean_inputs.json",
        "clean_labels.json",
        "decisions.json",
        "split.json",
        "policy.json",
        "private/inventory.jsonl",
    ],
)
def test_loader_rejects_any_hashed_artifact_modification(tmp_path, filename):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    path = output / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        load_partition(output, "train")


def replace_and_rehash(output, filename, value):
    """Adversarial fixture: checksum-valid but semantically inconsistent bundle."""
    path = output / filename
    path.write_text(json.dumps(value))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_loader_checks_input_label_identity_even_after_rehash(tmp_path):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    labels = json.loads((output / "clean_labels.json").read_text())
    labels.pop(next(iter(labels)))
    replace_and_rehash(output, "clean_labels.json", labels)
    with pytest.raises(ValueError):
        load_partition(output, "train")


def test_loader_checks_duplicate_input_ids_even_after_rehash(tmp_path):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    inputs = json.loads((output / "clean_inputs.json").read_text())
    inputs.append(copy.deepcopy(inputs[0]))
    replace_and_rehash(output, "clean_inputs.json", inputs)
    with pytest.raises(ValueError):
        load_partition(output, "train")


@pytest.mark.parametrize("eligible", [False, None, "true", 1])
def test_loader_requires_explicit_clean_boolean_decision_after_rehash(tmp_path, eligible):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    inputs = json.loads((output / "clean_inputs.json").read_text())
    decisions = json.loads((output / "decisions.json").read_text())
    decisions[inputs[0]["example_id"]]["eligible"] = eligible
    replace_and_rehash(output, "decisions.json", decisions)
    with pytest.raises(ValueError):
        load_partition(output, "train")


def test_loader_rejects_unlabeled_target_even_after_rehash(tmp_path):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    labels = json.loads((output / "clean_labels.json").read_text())
    labels[next(iter(labels))] = None
    replace_and_rehash(output, "clean_labels.json", labels)
    with pytest.raises(ValueError):
        load_partition(output, "train")


def test_loader_rejects_cross_partition_session_overlap_after_rehash(tmp_path):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    split = json.loads((output / "split.json").read_text())
    split["test_cell_ids"].append(split["train_cell_ids"][0])
    replace_and_rehash(output, "split.json", split)
    with pytest.raises(ValueError):
        load_partition(output, "train")


def test_loader_rejects_missing_clean_partition_id_after_rehash(tmp_path):
    output = tmp_path / "bundle"
    freeze_dataset(bundle_rows(), output, test_count=1)
    split = json.loads((output / "split.json").read_text())
    split["clean_train_example_ids"].append("fixture-nonexistent/exp-99")
    replace_and_rehash(output, "split.json", split)
    with pytest.raises(ValueError):
        load_partition(output, "train")
