"""Synthetic cleanup contract tests; no fitting, archived execution, or network."""

import copy
import json

import pytest

from tools.outcome_prediction import wm_one_step_data as data


def fixture(tmp_path, card="exp-01", score=0.0, cell="fixture-01"):
    directory = tmp_path / cell / "wm" / "cards" / card
    directory.mkdir(parents=True, exist_ok=True)
    runtime = "/home/ben/task"
    output = runtime + "/runs/" + card
    setup = {
        "base_model": "fixture/base",
        "parent_checkpoint": {"path": "fixture/base"},
        "command": {"argv": ["python", "train.py"], "script": runtime + "/train.py", "cwd": runtime},
        "output_dir": output,
        "method": {"family": "sft", "hyperparams": {"lr": 0.001}},
        "data": [{"source": "training-only"}],
    }
    first_at, archived_at = "2026-09-05T10:00:00Z", "2026-09-05T11:00:00Z"
    first = {"event": "submit", "at": first_at, "card": {"card_id": card, "setup": setup, "result": {}}}
    archived = {
        "event": "submit", "at": archived_at,
        "card": {"card_id": card, "setup": setup, "result": {"execution": "completed", "output_checkpoint": output}},
    }
    for name, value in (("record-01.json", first), ("record-02.json", archived), ("card.json", archived["card"])):
        data.write(directory / name, value)
    ledger = directory.parent.parent / "records.jsonl"
    event = {
        "event": "submit", "stage": "closed", "at": archived_at, "card_id": card,
        "record_n": 2, "path": runtime + "/wm/cards/" + card + "/record-02.json",
        "archived": runtime + "/wm/checkpoints/" + card,
    }
    with ledger.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    metric = tmp_path / cell / "wm_metrics" / (card + ".json")
    metric.parent.mkdir(exist_ok=True)
    data.write(metric, {"accuracy": score})
    log = tmp_path / cell / "output.log"
    with log.open("a") as stream:
        stream.write("card checkpoint evaluation: " + card + " (max_tokens='')\n")
    row = {
        "example_id": cell + "/" + card, "cell_id": cell, "card_id": card,
        "benchmark": "aime2025", "scientist_model": "fixture-scientist",
        "first_submitted_at": first_at, "first_stage": "plan", "eligible": True,
        "model_input": {"task": {"base_model": "fixture/base"}, "plan": {"setup": setup}, "code": []},
        "label": {"accuracy": score, "evaluation_n": 30, "official_metric": {"accuracy": score}},
        "audit": {"eligible": True, "reasons": [], "first_missing_fields": []},
        "provenance": {},
    }
    for name, path in (("card", directory / "card.json"), ("first_record", directory / "record-01.json"), ("ledger", ledger), ("official_metric", metric)):
        row["provenance"].update({name + "_path": str(path), name + "_sha256": data.sha(path)})
    return row


def exact(row):
    return {"status": "exact", "target_output": row["audit"]["final_output_checkpoint"]}


def single(path):
    return {"status": "single", "inputs": [{"path": path}], "evidence": [], "reasons": []}


def child_of(parent, *, cell=None):
    return {
        "example_id": (cell or parent["cell_id"]) + "/exp-99",
        "cell_id": cell or parent["cell_id"], "benchmark": parent["benchmark"],
        "first_submitted_at": "2026-09-05T12:00:00Z",
        "label": {"evaluation_n": 30},
    }


def test_zero_is_a_valid_official_score_and_parent(tmp_path):
    row = fixture(tmp_path, score=0)
    binding = data.archive_binding(row)
    assert binding["status"] == "official_archive_correlated"
    reference = data.parent_reference(child_of(row), single(binding["source_path"]), {row["example_id"]: binding})
    assert reference["reference_known"] is True
    assert reference["accuracy"] == 0


@pytest.mark.parametrize("missing", ["null_accuracy", "absent_accuracy", "absent_label"])
def test_missing_score_cannot_supply_a_target_or_parent_reference(tmp_path, missing):
    row = fixture(tmp_path, score=None)
    if missing == "absent_accuracy":
        del row["label"]["accuracy"]
    elif missing == "absent_label":
        del row["label"]
    binding = data.archive_binding(row)
    assert binding["status"] == "archive_without_valid_grade"
    assert "accuracy" not in binding
    assert not data.target_decision(row, binding, exact)["eligible"]
    reference = data.parent_reference(
        child_of(row), single(binding["source_path"]), {row["example_id"]: binding},
    )
    assert reference["reference_known"] is False
    assert "accuracy" not in reference


def test_first_archive_not_latest_mutable_output(tmp_path):
    row = fixture(tmp_path)
    card_path = data.Path(row["provenance"]["card_path"])
    card = data.read(card_path)
    card["result"]["output_checkpoint"] = "/home/ben/task/final_model"
    card_path.write_text(json.dumps(card))
    row["provenance"]["card_sha256"] = data.sha(card_path)
    archived = data.archive_binding(row)
    assert archived["source_path"].endswith("runs/exp-01")
    assert archived["latest_card_output"].endswith("final_model")
    decision = data.target_decision(row, archived, exact)
    assert decision["eligible"]
    assert decision["target_binding"]["target_output"] == archived["source_path"]


def test_later_archive_resubmission_does_not_rebind_score(tmp_path):
    row = fixture(tmp_path)
    ledger_path = data.Path(row["provenance"]["ledger_path"])
    event = data.load_jsonl(ledger_path)[0]
    event.update(at="2026-09-05T13:00:00Z", record_n=3)
    with ledger_path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    row["provenance"]["ledger_sha256"] = data.sha(ledger_path)
    assert data.archive_binding(row)["evidence"]["record_n"] == 2


@pytest.mark.parametrize("field", ["card", "ledger", "first_record", "official_metric"])
def test_raw_source_hashes_verified(tmp_path, field):
    row = fixture(tmp_path)
    path = data.Path(row["provenance"][field + "_path"])
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Changed raw source"):
        data.archive_binding(row)


def test_parent_recipe_not_required_for_valid_parent_score(tmp_path):
    row = fixture(tmp_path, score=0.2)
    row.update(eligible=False, first_stage="closed")
    row["audit"].update(eligible=False, reasons=["known_fidelity_exclusion"])
    binding = data.archive_binding(row)
    assert not data.target_decision(row, binding, exact)["eligible"]
    reference = data.parent_reference(child_of(row), single(binding["source_path"]), {row["example_id"]: binding})
    assert reference["reference_known"]


@pytest.mark.parametrize("status,reason", [
    ("base", "base_official_protocol_score_unverified"),
    ("multiple", "multiple_checkpoint_inputs_not_a_single_parent_delta"),
    ("unresolved", "checkpoint_input_unresolved"),
])
def test_unknown_parent_never_gets_zero_or_mean(tmp_path, status, reason):
    row = fixture(tmp_path)
    result = data.parent_reference(row, {"status": status, "inputs": []}, {})
    assert result["reference_known"] is False
    assert result["reasons"] == [reason]
    assert "accuracy" not in result


def test_no_cross_session_or_self_label_join(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    registry = {row["example_id"]: binding}
    assert not data.parent_reference(child_of(row, cell="different-session"), single(binding["source_path"]), registry)["reference_known"]
    row["first_submitted_at"] = "2026-09-05T12:00:00Z"
    assert not data.parent_reference(row, single(binding["source_path"]), registry)["reference_known"]


@pytest.mark.parametrize("at", ["2026-09-05T10:30:00Z", "2026-09-05T11:00:00Z"])
def test_parent_archive_must_strictly_precede_proposal(tmp_path, at):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    child = child_of(row)
    child["first_submitted_at"] = at
    assert not data.parent_reference(child, single(binding["source_path"]), {row["example_id"]: binding})["reference_known"]


def test_reused_path_not_chosen_by_nearest_time(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    other = copy.deepcopy(binding)
    other.update(example_id="fixture-01/exp-02", archived_at="2026-09-05T11:15:00Z")
    registry = {binding["example_id"]: binding, other["example_id"]: other}
    result = data.parent_reference(child_of(row), single(binding["source_path"]), registry)
    assert result["reasons"] == ["ambiguous_reused_checkpoint_path"]


def test_root_directory_not_equal_to_final_checkpoint(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    binding["source_path"] += "/final"
    result = data.parent_reference(child_of(row), single(binding["source_path"].removesuffix("/final")), {row["example_id"]: binding})
    assert result["reasons"] == ["no_exact_scored_archive"]


def test_parent_evaluation_size_must_match(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    binding["evaluation_n"] = 100
    result = data.parent_reference(child_of(row), single(binding["source_path"]), {row["example_id"]: binding})
    assert result["reasons"] == ["parent_child_evaluation_protocol_mismatch"]


def test_planned_save_path_recovery_does_not_clear_recipe_change(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    binding["archived_setup"]["method"]["hyperparams"]["lr"] = 0.3
    result = data.target_decision(row, binding, lambda _: {"status": "planned_save_path"})
    assert not result["eligible"]
    assert "proposal_to_archived_setup_changed" in result["reasons"]


def test_known_hold_and_missing_first_fields_remain_quarantined(tmp_path):
    row = fixture(tmp_path)
    binding = data.archive_binding(row)
    row["example_id"] = "aime-r0-20/exp-09"
    row["audit"]["first_missing_fields"] = ["setup.data"]
    result = data.target_decision(row, binding, exact)
    assert "unresolved_prior_target_binding_hold" in result["reasons"]
    assert "incomplete_first_registration" in result["reasons"]


def test_export_has_real_delta_and_no_own_label_in_features(tmp_path):
    parent = fixture(tmp_path, score=0.2)
    child = fixture(tmp_path, card="exp-02", score=0.3)
    # Both fixture records share one ledger; refresh its expected provenance.
    parent["provenance"]["ledger_sha256"] = child["provenance"]["ledger_sha256"]
    child["first_submitted_at"] = "2026-09-05T11:05:00Z"
    record_path = data.Path(child["provenance"]["card_path"]).parent / "record-02.json"
    record = data.read(record_path)
    record["at"] = "2026-09-05T11:30:00Z"
    record_path.write_text(json.dumps(record))
    ledger_path = data.Path(child["provenance"]["ledger_path"])
    events = data.load_jsonl(ledger_path)
    events[1]["at"] = "2026-09-05T11:30:00Z"
    ledger_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    for row in (parent, child):
        row["provenance"]["ledger_sha256"] = data.sha(ledger_path)
    parent_path = parent["model_input"]["plan"]["setup"]["output_dir"]
    def parse(row):
        return single(parent_path) if row["card_id"] == "exp-02" else {"status": "base", "inputs": []}
    bundle = data.build(
        [parent, child], {"cell_partition": {"fixture-01": "test"}}, exact, parse,
        mutation_resolver=lambda *_: {"status": "clear", "reasons": []},
    )
    assert len(bundle["target_inputs"]) == 2
    assert len(bundle["one_step_inputs"]) == 1
    features = bundle["one_step_inputs"][0]
    assert features["parent"]["accuracy"] == 0.2
    assert "label" not in features
    assert "0.3" not in json.dumps(features)
    assert bundle["one_step_labels"][child["example_id"]]["delta_accuracy"] == pytest.approx(0.1)
    assert features["history"][0]["example_id"] == parent["example_id"]
    assert features["history_complete_to_base"]


def test_build_rejects_duplicate_identity_and_missing_split(tmp_path):
    row = fixture(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        data.build([row, row], {"cell_partition": {}}, exact, lambda _: {})
    with pytest.raises(ValueError, match="Every session"):
        data.build([row], {"cell_partition": {}}, exact, lambda _: {})


def test_equal_timestamp_may_use_same_ledger_sequence(tmp_path):
    row = fixture(tmp_path)
    parent = data.archive_binding(row)
    child = child_of(row)
    child.update(card_id="exp-99", first_submitted_at=parent["archived_at"])
    child["provenance"] = copy.deepcopy(row["provenance"])
    child["provenance"]["first_record_path"] = "/unused/cards/exp-99/record-01.json"
    ledger_path = data.Path(row["provenance"]["ledger_path"])
    event = {
        "event": "submit", "stage": "plan", "at": parent["archived_at"],
        "card_id": "exp-99", "path": "/home/ben/task/wm/cards/exp-99/record-01.json",
    }
    with ledger_path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    child["provenance"]["ledger_sha256"] = data.sha(ledger_path)
    assert data.parent_precedes_proposal(parent, child)
    parent["evidence"]["ledger_line"] = 3
    assert not data.parent_precedes_proposal(parent, child)


def test_late_archive_distinguished_from_absent_grade(tmp_path):
    row = fixture(tmp_path)
    parent = data.archive_binding(row)
    child = child_of(row)
    child["first_submitted_at"] = "2026-09-05T10:55:00Z"
    result = data.parent_reference(child, single(parent["source_path"]), {row["example_id"]: parent})
    assert result["reasons"] == ["matching_archive_not_proven_before_proposal"]
    assert result["candidate_ids"] == [row["example_id"]]


def minimal_bundle(tmp_path):
    row = fixture(tmp_path / "raw")
    archived = data.archive_binding(row)
    key, parent_id = row["example_id"], row["cell_id"] + "/exp-00"
    parent = {
        **copy.deepcopy(archived), "example_id": parent_id, "accuracy": 0.1,
        "archived_at": "2026-09-05T09:00:00Z", "card_id": "exp-00",
        "source_path": "/home/ben/task/runs/exp-00", "archive_path": "/home/ben/task/wm/checkpoints/exp-00",
    }
    example = data.project_example(row)
    example["parent"] = {
        "producer_id": parent_id, "reference_known": True, "accuracy": 0.1,
        "evaluation_n": 30, "consumed_path": parent["source_path"], "path_kind": "archived_source",
        "official_grade_availability": "retrospective_observed_parent_assumption",
    }
    label = {**copy.deepcopy(row["label"]), "delta_accuracy": -0.1}
    values = {
        "private/registry.json": {key: archived, parent_id: parent},
        "private/decisions.json": {key: {
            "eligible": True, "one_step_eligible": True, "proposal_at": row["first_submitted_at"],
            "parent_reference": {"status": "official_archive_correlated", **copy.deepcopy(example["parent"])},
        }},
        "target_inputs.json": [data.project_example(row)],
        "target_labels.json": {key: row["label"]},
        "one_step_inputs.json": [example],
        "one_step_labels.json": {key: label},
        "split.json": {"cell_partition": {row["cell_id"]: "train"}},
        "summary.json": {}, "policy.json": {},
    }
    root = tmp_path / "bundle"
    (root / "private").mkdir(parents=True)
    for name, value in values.items():
        data.write(root / name, value)
    data.write(root / "manifest.json", {
        "files": {name: data.sha(root / name) for name in values},
        "sources": {row["provenance"]["card_path"]: row["provenance"]["card_sha256"]},
    })
    return root


def test_loader_separates_sessions_and_labels(tmp_path):
    root = minimal_bundle(tmp_path)
    examples, labels = data.load_partition(root, "train")
    assert len(examples) == len(labels) == 1
    assert data.load_partition(root, "test") == ([], {})
    assert "label" not in examples[0]
    assert labels[examples[0]["example_id"]]["delta_accuracy"] == -0.1


@pytest.mark.parametrize("partition", ["train", "test"])
@pytest.mark.parametrize("score_owner", ["target", "parent"])
@pytest.mark.parametrize("missing", ["null", "absent"])
def test_delta_loader_rejects_missing_accuracy_in_every_partition(
    tmp_path, partition, score_owner, missing,
):
    root = minimal_bundle(tmp_path)
    inputs = data.read(root / "one_step_inputs.json")
    labels = data.read(root / "one_step_labels.json")
    key = inputs[0]["example_id"]
    score = labels[key] if score_owner == "target" else inputs[0]["parent"]
    if missing == "null":
        score["accuracy"] = None
    else:
        del score["accuracy"]
    # Rehash deliberately: semantic checks must reject missing scores even when
    # the file-integrity check passes, including in the requested test partition.
    values = {
        "one_step_inputs.json": inputs,
        "one_step_labels.json": labels,
        "split.json": {"cell_partition": {inputs[0]["cell_id"]: partition}},
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    manifest = data.read(root / "manifest.json")
    manifest["files"] = {name: data.sha(root / name) for name in manifest["files"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.load_partition(root, partition)


@pytest.mark.parametrize("name", sorted(data.REQUIRED_FILES))
def test_loader_rejects_modified_or_uncovered_bundle_file(tmp_path, name):
    root = minimal_bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = data.read(manifest_path)
    del manifest["files"][name]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete frozen manifest"):
        data.load_partition(root, "train")


def test_loader_detects_changed_content(tmp_path):
    root = minimal_bundle(tmp_path)
    with (root / "one_step_inputs.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Changed frozen data"):
        data.load_partition(root, "train")


def test_loader_detects_changed_sources(tmp_path):
    root = minimal_bundle(tmp_path)
    path = data.Path(next(iter(data.read(root / "manifest.json")["sources"])))
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Changed frozen source"):
        data.load_partition(root, "train")


@pytest.mark.parametrize("change", [
    "missing_parent", "wrong_delta", "cross_session", "self_label", "quarantined",
    "wrong_path", "wrong_parent_evaluation_n", "different_bound_reference", "foreign_parent_benchmark",
])
def test_loader_rejects_semantically_invalid_rows_even_with_matching_file_hash(tmp_path, change):
    root = minimal_bundle(tmp_path)
    inputs, labels = data.read(root / "one_step_inputs.json"), data.read(root / "one_step_labels.json")
    registry = data.read(root / "private/registry.json")
    decisions = data.read(root / "private/decisions.json")
    key, parent_id = inputs[0]["example_id"], inputs[0]["parent"]["producer_id"]
    if change == "missing_parent":
        inputs[0]["parent"]["reference_known"] = False
    elif change == "wrong_delta":
        labels[key]["delta_accuracy"] = 0.8
    elif change == "cross_session":
        registry[parent_id]["cell_id"] = "heldout-session"
    elif change == "self_label":
        inputs[0]["parent"]["producer_id"] = key
    elif change == "quarantined":
        decisions[key]["eligible"] = False
    elif change == "wrong_path":
        inputs[0]["parent"]["consumed_path"] = "/home/ben/task/unscored-checkpoint"
    elif change == "wrong_parent_evaluation_n":
        registry[parent_id]["evaluation_n"] = 100
    elif change == "different_bound_reference":
        decisions[key]["parent_reference"]["producer_id"] = "other-parent"
    else:
        registry[parent_id]["benchmark"] = "different-task"
    for name, value in (("one_step_inputs.json", inputs), ("one_step_labels.json", labels), ("private/registry.json", registry), ("private/decisions.json", decisions)):
        (root / name).write_text(json.dumps(value))
    manifest = data.read(root / "manifest.json")
    manifest["files"] = {name: data.sha(root / name) for name in manifest["files"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.load_partition(root, "train")


def test_relative_directory_does_not_bypass_manifest_containment(tmp_path, monkeypatch):
    root = minimal_bundle(tmp_path)
    manifest = data.read(root / "manifest.json")
    manifest["files"]["../outside.json"] = "invalid"
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Invalid manifest path"):
        data.load_partition("bundle", "train")


def test_static_code_requires_hashed_preproposal_provenance(tmp_path):
    row = fixture(tmp_path)
    code = {"role": "training", "script_path": "/home/ben/task/train.py", "status": "reconstructed", "content": "pass\n"}
    row["model_input"]["code"] = [code]
    assert data.code_provenance_issues(row) == ["code_reconstruction_provenance_missing"]
    proof = {
        "role": code["role"], "script_path": code["script_path"], "status": "reconstructed",
        "sha256": data.hashlib.sha256(code["content"].encode()).hexdigest(),
        "cutoff": row["first_submitted_at"], "blockers": [],
        "evidence": {"at": "2026-09-05T09:30:00Z", "envelope_at": "2026-09-05T09:30:00Z"},
    }
    row["audit"]["code_provenance"] = [proof]
    assert data.code_provenance_issues(row) == []
    proof["evidence"]["at"] = "2026-09-05T10:00:00Z"
    assert data.code_provenance_issues(row) == ["code_not_bound_to_preproposal_version"]
    proof["evidence"]["at"] = "2026-09-05T09:30:00Z"
    code["content"] = "post_proposal_replacement\n"
    assert data.code_provenance_issues(row) == ["code_not_bound_to_preproposal_version"]


def test_modified_archives_cannot_supply_targets_or_parent_scores(tmp_path):
    row = fixture(tmp_path, cell="aime2-r0-21", card="exp-02")
    binding = data.archive_binding(row)
    assert binding["status"] == "unresolved"
    assert binding["reasons"] == ["known_archive_integrity_hold"]
    assert not data.target_decision(row, binding, exact)["eligible"]
    result = data.parent_reference(child_of(row), single(binding["source_path"]), {row["example_id"]: binding})
    assert result["reference_known"] is False
