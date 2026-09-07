import copy
import hashlib
import json

import pytest

from tools.outcome_prediction import wm_compile as compiler
from tools.outcome_prediction import wm_data_export as export
from tools.outcome_prediction import wm_final_prepare as prepare
from tools.outcome_prediction import wm_lineage as lineage


def fixture(tmp_path):
    split = {
        "train_cell_ids": ["r0-01", "r0-02"],
        "test_cell_ids": ["r0-90", "r0-91"],
        "cell_partition": {"r0-01": "train", "r0-02": "train", "r0-90": "test", "r0-91": "test"},
    }
    rows = []
    for cell, number in [("r0-01", 1), ("r0-02", 1), ("r0-90", 1), ("r0-90", 2)]:
        rows.append(
            {
                "example_id": f"{cell}/exp-{number:02d}",
                "cell_id": cell,
                "card_id": f"exp-{number:02d}",
                "first_submitted_at": f"2026-01-01T0{number}:00:00Z",
                "model_input": {
                    "task": {"benchmark": "gsm8k", "base_model": "org/base", "evaluation_n": 1319},
                    "plan": {
                        "setup": {
                            "base_model": "org/base",
                            "method": {"family": "sft"},
                            "parent_checkpoint": {"path": "org/base"},
                        }
                    },
                    "code": [],
                },
                "audit": {"code_provenance": []},
                "label": {"accuracy": 0.5, "official_metric": {"accuracy": 0.5}},
            }
        )
    graph = lineage.build_graph(rows)
    draft = prepare.assemble(rows, graph, split)
    reviews = {
        q["payload_sha256"]: {
            "payload_sha256": q["payload_sha256"],
            "reviewer": "synthetic fixture",
            "evidence_sha256": "a" * 64,
            "outcome_free": True,
            "executable_recipe_preserved": True,
        }
        for q in draft["review_queue"]
        if q["source_example_id"] != "r0-02/exp-01"
    }
    prepared = prepare.assemble(rows, graph, split, whole_step_reviews=reviews)
    bundle = tmp_path / "source"
    bundle.mkdir()
    hashes = {}
    for key in ("drafts", "review_queue", "audit"):
        path = bundle / (key + ".json")
        path.write_text(json.dumps(prepared[key]))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = tmp_path / "inventory.jsonl"
    # Invalid TEST label JSON is deliberately never decoded by the export.
    inventory.write_text(
        "\n".join(json.dumps(r) for r in rows[:2]) + '\n{"cell_id":"r0-90","label":INVALID}\n'
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    (bundle / "provenance.json").write_text(
        json.dumps(
            {
                "files_sha256": hashes,
                "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            }
        )
    )
    return bundle, inventory, split_path


def test_export_joins_only_train_labels_and_preserves_empty_runs(tmp_path):
    paths = fixture(tmp_path)
    output = tmp_path / "export"
    summary = export.build(*paths, output)
    assert summary == {
        "train_recipes_with_final_official_labels": 1,
        "test_input_recipes": 2,
        "current_input_comparison_runs": 1,
        "frozen_train_runs": 2,
        "frozen_test_runs": 2,
    }
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert records[0]["label"] == {"accuracy": 0.5}
    test = json.loads((output / "test_inputs.json").read_text())
    assert all(set(row) == set(export.WRAPPER_KEYS) and "label" not in row for row in test.values())
    cases = json.loads((output / "case_index.json").read_text())
    assert not cases["study_frozen"] and not cases["cohort_finalized"]
    assert len(cases["cases"]) == 2 and cases["cases"][1]["candidate_ids"] == []
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["test_inventory_rows_skipped_before_record_decode"] == 1
    assert len(coverage["run_coverage"]) == 4
    assert json.loads((output / "split.json").read_text())["empty_train_cell_ids"] == ["r0-02"]
    proof = json.loads((output / "provenance.json").read_text())
    for name, expected in proof["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "train.jsonl").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export.build(*paths, output)


@pytest.mark.parametrize("target", ["inventory", "split", "drafts"])
def test_stale_input_hash_rejected_before_output(tmp_path, target):
    bundle, inventory, split = fixture(tmp_path)
    path = {"inventory": inventory, "split": split, "drafts": bundle / "drafts.json"}[target]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        export.build(bundle, inventory, split, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unapproved_test_content_cannot_be_exported(tmp_path):
    bundle, inventory, split = fixture(tmp_path)
    path = bundle / "drafts.json"
    rows = json.loads(path.read_text())
    selected = next(r for r in rows if r["cell_id"] == "r0-90")
    selected["content_review"] = copy.deepcopy(selected["content_review"])
    selected["content_review"]["status"] = "needs_review"
    path.write_text(json.dumps(rows))
    proof_path = bundle / "provenance.json"
    proof = json.loads(proof_path.read_text())
    proof["files_sha256"]["drafts.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="not approved"):
        export.build(bundle, inventory, split, tmp_path / "out")


def test_assembled_split_binding_is_checked(tmp_path):
    bundle, inventory, split = fixture(tmp_path)
    path = bundle / "audit.json"
    audit = json.loads(path.read_text())
    assert audit["split_sha256"] == compiler.digest(json.loads(split.read_text()))
    audit["split_sha256"] = "f" * 64
    path.write_text(json.dumps(audit))
    proof_path = bundle / "provenance.json"
    proof = json.loads(proof_path.read_text())
    proof["files_sha256"]["audit.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="split changed"):
        export.build(bundle, inventory, split, tmp_path / "out")
